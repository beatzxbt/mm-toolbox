"""WebSocket connection pool management."""

import asyncio
import inspect
import logging
from collections.abc import Callable
from heapq import nlargest
from typing import Any, Self
from inspect import get_annotations

import xxhash
import numpy as np
from msgspec import Struct
from msgspec.json import decode as json_decode

from mm_toolbox.ringbuffer.numeric import NumericRingBuffer, _contains_u64, _insert_u64
from mm_toolbox.time.time import time_s
from mm_toolbox.websocket.connection import (
    ConnectionState,
    WsConnection,
    WsConnectionConfig,
    _ReceiveBuffer,
)

UINT64_MASK = (1 << 64) - 1
_logger = logging.getLogger(__name__)


class WsPoolConfig(Struct):
    """Configuration for WebSocket connection pool."""

    num_connections: int
    evict_interval_s: int
    hash_capacity: int = 16_384

    def __post_init__(self) -> None:
        """Validate pool configuration parameters."""
        if self.num_connections <= 1:
            raise ValueError(
                f"Invalid num_connections; expected >1 but got {self.num_connections}"
            )
        if self.evict_interval_s <= 0:
            raise ValueError(
                f"Invalid eviction interval; expected >0 but got "
                f"{self.evict_interval_s}s"
            )
        if self.hash_capacity <= 0:
            raise ValueError(
                f"Invalid hash_capacity; expected >0 but got {self.hash_capacity}"
            )

    @classmethod
    def default(cls) -> "WsPoolConfig":
        """Create default pool configuration."""
        return cls(
            num_connections=5,
            evict_interval_s=60,
            hash_capacity=16_384,
        )


class WsPool:
    """Manages a pool of fast WebSocket connections."""

    def __init__(
        self,
        config: WsConnectionConfig,
        on_message: Callable[[bytes], None] | None = None,
        pool_config: WsPoolConfig | None = None,
    ) -> None:
        """Initialize a pool of WebSocket connections.

        Args:
            config (WsConnectionConfig): Base connection configuration.
            on_message (Callable[[bytes], None], optional): Single-argument
                callback invoked for each unique received message.
            pool_config (WsPoolConfig, optional): Pool-specific settings.
                Defaults to ``WsPoolConfig.default()``.

        Raises:
            ValueError: If ``on_message`` does not accept exactly one ``bytes``
                argument.

        """
        self._config: WsConnectionConfig = config
        self._pool_config: WsPoolConfig = pool_config or WsPoolConfig.default()
        self._ringbuffer = _ReceiveBuffer(128)
        self._hashes: NumericRingBuffer = NumericRingBuffer(
            max_capacity=self._pool_config.hash_capacity,
            dtype=np.uint64,
            disable_async=True,
        )

        # Verify the signature of the on_message, must be a single bytes arg
        if on_message is not None:
            sig = inspect.signature(on_message)
            if len(sig.parameters) != 1:
                raise ValueError(
                    f"Invalid on_message signature; expected a single bytes "
                    f"argument but got {sig}"
                )
            param = next(iter(sig.parameters.values()))
            try:
                type_hints = get_annotations(on_message, eval_str=False)
            except Exception:
                type_hints = {}
            param_type = type_hints.get(param.name, param.annotation)
            if param_type == "bytes":
                param_type = bytes
            if param_type is not bytes:
                raise ValueError(
                    f"Invalid on_message signature; expected a single bytes "
                    f"argument but got {sig}"
                )

        self._on_message: Callable[[bytes], None] = (
            on_message or self.__default_on_message
        )

        self._conns: dict[int, WsConnection] = {}
        self._fast_conn: WsConnection | None = None
        self._pool_state: ConnectionState = ConnectionState.DISCONNECTED
        self._should_stop = False  # Lightweight stop signal
        self._eviction_task: asyncio.Task[None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._started = False
        self._generation = 0
        self._opening_tasks: set[asyncio.Task[WsConnection]] = set()
        self._replacement_tasks: set[asyncio.Task[None]] = set()

    def __default_on_message(self, msg: bytes) -> None:
        """Default callback for processing WebSocket messages. For convenience
        in debugging, this just prints the json."""
        try:
            print(json_decode(msg))
        except Exception as e:
            print(f"Error decoding message: {e}")

    async def _timed_operations(self) -> None:
        """Restore missing sockets and periodically replace slow live sockets."""
        next_eviction_time = time_s() + self._pool_config.evict_interval_s
        while not self._should_stop:
            for conn_id, conn in list(self._conns.items()):
                if not conn.is_connected():
                    self._conns.pop(conn_id, None)
            self._update_fast_connection()
            if self._config.auto_reconnect:
                if time_s() >= next_eviction_time:
                    connected = list(self._conns.values())
                    restart_count = (
                        self._pool_config.num_connections // 2
                        if self._pool_config.num_connections >= 4
                        else 1
                    )
                    for conn in nlargest(
                        restart_count, connected, key=lambda c: c.get_latency_ms()
                    ):
                        self._conns.pop(conn.get_config().conn_id, None)
                        conn.close()
                    self._update_fast_connection()
                    next_eviction_time = time_s() + self._pool_config.evict_interval_s
                self._schedule_replacements(self._pool_config.num_connections)
            await asyncio.sleep(0.1)

    def _update_fast_connection(self) -> None:
        """Refresh the cached fastest connection by latency.

        Scans all connected sockets and stores the one with the lowest
        ``get_latency_ms()`` value.

        """
        if not self._conns:
            self._fast_conn = None
            return
        self._fast_conn = min(
            (conn for conn in self._conns.values() if conn.is_connected()),
            key=lambda x: x.get_latency_ms(),
            default=None,
        )

    def _schedule_replacements(self, count: int) -> None:
        """Keep at most one retry task per missing connection."""
        if count <= 0 or self._should_stop or not self._config.auto_reconnect:
            return
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            same_loop = asyncio.get_running_loop() is loop
        except RuntimeError:
            same_loop = False
        if not same_loop:
            loop.call_soon_threadsafe(self._schedule_replacements, count)
            return
        missing = (
            self._pool_config.num_connections
            - self.get_connection_count()
            - len(self._replacement_tasks)
        )
        for _ in range(min(count, max(0, missing))):
            task = loop.create_task(self._replace_connection())
            self._replacement_tasks.add(task)
            task.add_done_callback(self._replacement_tasks.discard)

    async def _replace_connection(self) -> None:
        """Retry a missing socket with bounded exponential backoff."""
        delay = 1.0
        while not self._should_stop and self._config.auto_reconnect:
            try:
                await self._open_new_conn()
                return
            except Exception as exc:
                _logger.warning("WebSocket replacement failed; retrying: %s", exc)
                await asyncio.sleep(delay)
                delay = min(delay * 2.0, 30.0)

    @staticmethod
    def _hash_payload(msg: bytes) -> int:
        """Build a stable 64-bit hash key for a payload."""
        return (xxhash.xxh64_intdigest(msg) ^ (len(msg) << 1)) & UINT64_MASK

    def _is_seen_hash(self, msg: bytes) -> bool:
        """Check and update hash history for an incoming payload.

        We intentionally use the internal uint64 helper APIs from
        ``ringbuffer.numeric`` instead of fused-type ``contains/insert`` calls.
        This avoids Python-side fused dispatch edge cases and gives a stable
        typed path for high-frequency hash-history checks.

        Returns:
            bool: True when msg hash already exists in history, else False.
        """
        key = self._hash_payload(msg)
        if _contains_u64(self._hashes, key):
            return True
        _insert_u64(self._hashes, key)
        return False

    def _send_data_now(self, msg: bytes, only_fastest: bool) -> None:
        """Send data using the current pool snapshot.

        Args:
            msg (bytes): Payload to send.
            only_fastest (bool): Whether to send only through the fastest connection.

        Returns:
            None: This method does not return a value.
        """
        if only_fastest:
            fast_conn = self._fast_conn
            if fast_conn is None or not fast_conn.is_connected():
                self._update_fast_connection()
                fast_conn = self._fast_conn
            if fast_conn is None:
                raise RuntimeError("No live connections")
            # A transport error may follow a write. Propagate it without
            # retrying an order whose delivery is uncertain.
            fast_conn.send_data(msg)
        else:
            connected = [conn for conn in self._conns.values() if conn.is_connected()]
            if not connected:
                raise RuntimeError("No live connections")
            for conn in connected:
                conn.send_data(msg)

    async def _open_new_conn(self) -> WsConnection:
        """Return one live socket or raise, including when shutdown races open."""
        if self._should_stop:
            raise RuntimeError("WebSocket pool is closed")
        generation = self._generation
        config = WsConnectionConfig.default(
            self._config.wss_url,
            on_connect=self._config.on_connect,
            auto_reconnect=self._config.auto_reconnect,
            max_frame_size=self._config.max_frame_size,
            latency_ping_interval_ms=self._config.latency_ping_interval_ms,
        )
        conn = await WsConnection.new(ringbuffer=self._ringbuffer, config=config)
        if (
            self._should_stop
            or generation != self._generation
            or not conn.is_connected()
        ):
            conn.close()
            raise RuntimeError("WebSocket pool closed or disconnected during open")
        self._conns[config.conn_id] = conn
        self._update_fast_connection()
        return conn

    def set_on_connect(self, on_connect: list[bytes]) -> None:
        """Update the messages sent when new connections open.

        Args:
            on_connect (list[bytes]): Payloads to transmit on connect.

        """
        self._config.on_connect = on_connect
        for conn in self._conns.values():
            conn.set_on_connect(on_connect)

    def send_data(self, msg: bytes, only_fastest: bool = True) -> None:
        """Sends a payload through all WebSocket connections in the pool.

        In instances of subscribing/unsubscribing to new topics, it is
        desired to send the message to all connections.

        If sending something latency sensitive like an Order, it should
        only be sent through one connection.
        """
        if not self._started or self._should_stop:
            raise RuntimeError("Connection not running")
        if self.get_connection_count() == 0:
            raise RuntimeError("No live connections")

        self._send_data_now(msg, only_fastest)

    @classmethod
    async def new(
        cls,
        config: WsConnectionConfig,
        on_message: Callable[[bytes], None] | None = None,
        pool_config: WsPoolConfig | None = None,
    ) -> Self:
        """Starts all WebSocket connections in the pool."""
        pool = cls(
            config=config,
            on_message=on_message,
            pool_config=pool_config,
        )
        return pool

    def get_state(self) -> ConnectionState:
        """Return the current aggregate pool state.

        Returns:
            ConnectionState: Overall pool state.

        """
        if self._started and not self._should_stop:
            if self.get_connection_count():
                return ConnectionState.CONNECTED
            if self._config.auto_reconnect:
                return ConnectionState.CONNECTING
            return ConnectionState.DISCONNECTED
        return self._pool_state

    def get_config(self) -> WsConnectionConfig:
        """Return the base connection configuration.

        Returns:
            WsConnectionConfig: Configuration shared by all pool connections.

        """
        return self._config

    def get_connection_count(self) -> int:
        """Return the number of currently connected sockets.

        Returns:
            int: Active connection count.

        """
        return sum(1 for conn in self._conns.values() if conn.is_connected())

    def get_latency_ms(self) -> float:
        """Return the lowest latency among active connections.

        Returns:
            float: Minimum latency in milliseconds, or ``0.0`` when empty.

        """
        if not self._conns:
            return 0.0
        latencies = [
            conn.get_latency_ms()
            for conn in self._conns.values()
            if conn.is_connected()
        ]
        return min(latencies) if latencies else 0.0

    def get_seq_id(self) -> int:
        """Return the highest sequence ID across active connections.

        Returns:
            int: Maximum ``seq_id``, or ``0`` when empty.

        """
        if not self._conns:
            return 0
        seq_ids = [
            conn.get_seq_id() for conn in self._conns.values() if conn.is_connected()
        ]
        return max(seq_ids) if seq_ids else 0

    def get_dropped_messages(self) -> int:
        """Return all shared receive-buffer overwrites over this pool's lifetime.

        Includes startup, evicted and disconnected listeners, before deduplication.
        Closing or reentering the pool never resets the cumulative count.
        """
        return self._ringbuffer.get_dropped_messages()

    def close(self) -> None:
        """Shut down the pool, cancel eviction, and close all connections."""
        self._should_stop = True
        self._started = False
        self._generation += 1
        for task in (*self._opening_tasks, *self._replacement_tasks):
            task.cancel()
        self._opening_tasks.clear()
        self._replacement_tasks.clear()
        self._pool_state = ConnectionState.DISCONNECTED
        if self._eviction_task is not None:
            self._eviction_task.cancel()
            self._eviction_task = None
        self._loop = None

        # Close all connections first (fastest operation)
        conns_snapshot = list(self._conns.values())
        self._conns.clear()
        self._fast_conn = None
        self._ringbuffer.clear()
        self._hashes.clear()
        for conn in conns_snapshot:
            conn.close()

    async def __aenter__(self) -> Self:
        """Open every requested socket; any failure rolls back the whole start."""
        if self._started and not self._should_stop:
            return self
        self._loop = asyncio.get_running_loop()
        self._should_stop = False
        self._generation += 1
        self._pool_state = ConnectionState.CONNECTING
        tasks = [
            self._loop.create_task(self._open_new_conn())
            for _ in range(self._pool_config.num_connections)
        ]
        self._opening_tasks.update(tasks)
        try:
            results = await asyncio.gather(*tasks)
            if (
                self._should_stop
                or len(results) != self._pool_config.num_connections
                or any(not conn.is_connected() for conn in results)
            ):
                raise RuntimeError("Failed to establish every WebSocket connection")
        except BaseException as exc:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self.close()
            if isinstance(exc, asyncio.CancelledError):
                raise
            raise RuntimeError(
                "Failed to establish every WebSocket connection"
            ) from exc
        finally:
            self._opening_tasks.difference_update(tasks)
        self._started = True
        self._pool_state = ConnectionState.CONNECTED
        self._update_fast_connection()
        self._eviction_task = asyncio.create_task(self._timed_operations())
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit. Closes all connections in the pool."""
        tasks = set(self._opening_tasks) | set(self._replacement_tasks)
        if self._eviction_task is not None:
            tasks.add(self._eviction_task)
        self.close()
        current = asyncio.current_task()
        await asyncio.gather(
            *(task for task in tasks if task is not current), return_exceptions=True
        )

    def __aiter__(self) -> Self:
        """Returns an async iterator over the pool's ringbuffer."""
        return self

    async def __anext__(self) -> bytes:
        """Returns the next hash-filtered message from the pool ringbuffer."""
        while True:
            try:
                msg = await asyncio.wait_for(self._ringbuffer.aconsume(), timeout=1.0)
            except asyncio.TimeoutError:
                if self._should_stop:
                    raise StopAsyncIteration
                continue
            if self._is_seen_hash(msg):
                continue
            return msg


if __name__ == "__main__":
    import asyncio

    async def _pool_websocket_example() -> None:
        """Example usage of the WsPool class."""
        config = WsConnectionConfig.default(
            wss_url="wss://fstream.binance.com/ws/btcusdt@bookTicker"
        )
        ws = await WsPool.new(config, on_message=lambda msg: print(json_decode(msg)))
        async with ws:
            msg_count = 0
            async for msg in ws:
                print(json_decode(msg))
                msg_count += 1
                if msg_count > 100:
                    print(f"Final state: {ws.get_state()}")
                    break

    asyncio.run(_pool_websocket_example())
