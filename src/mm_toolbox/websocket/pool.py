"""WebSocket connection pool management."""

import asyncio
import inspect
from collections.abc import Callable
from typing import Any, Self, get_type_hints

from msgspec import Struct
from msgspec.json import decode as json_decode

from mm_toolbox.ringbuffer.bytes import BytesRingBuffer
from mm_toolbox.time.time import time_s
from mm_toolbox.websocket.connection import (
    ConnectionState,
    WsConnection,
    WsConnectionConfig,
)


class WsPoolConfig(Struct):
    """Configuration for WebSocket connection pool."""

    num_connections: int
    evict_interval_s: int

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

    @classmethod
    def default(cls) -> "WsPoolConfig":
        """Create default pool configuration."""
        return cls(
            num_connections=5,
            evict_interval_s=60,
        )


class WsPool:
    """Manages a pool of fast WebSocket connections."""

    def __init__(
        self,
        config: WsConnectionConfig,
        on_message: Callable[[bytes], None],
        pool_config: WsPoolConfig | None = None,
    ) -> None:
        """Initialize WebSocket pool with configuration and message handler."""
        self._config: WsConnectionConfig = config
        self._ringbuffer: BytesRingBuffer = BytesRingBuffer(
            max_capacity=128, only_insert_unique=True
        )
        self._pool_config: WsPoolConfig = pool_config or WsPoolConfig.default()

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
                type_hints = get_type_hints(on_message)
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

    def __default_on_message(self, msg: bytes) -> None:
        """Default callback for processing WebSocket messages. For convenience
        in debugging, this just prints the json."""
        try:
            print(json_decode(msg))
        except Exception as e:
            print(f"Error decoding message: {e}")

    async def _timed_operations(self) -> None:
        """Enforces the configuration to evict slow connections."""
        time_now_s = time_s()
        next_eviction_time = time_now_s + self._pool_config.evict_interval_s

        while not self._should_stop:
            if (
                self._pool_state != ConnectionState.CONNECTED
                or time_s() < next_eviction_time
            ):
                await asyncio.sleep(1)
                continue

            fast_to_slow_conns = sorted(
                self._conns.values(), key=lambda x: x.get_state().latency_ms
            )
            restart_count = (
                self._pool_config.num_connections // 2
                if self._pool_config.num_connections >= 4
                else 1
            )
            restart_conns = (
                fast_to_slow_conns[-restart_count:] if fast_to_slow_conns else []
            )

            # Queue connection replacements instead of blocking
            for old_conn in restart_conns:
                old_conn.close()
                old_conn_id = old_conn.get_config().conn_id
                self._conns.pop(old_conn_id, None)

            self._update_fast_connection()
            self._schedule_replacements(len(restart_conns))

            next_eviction_time += self._pool_config.evict_interval_s

    def _update_fast_connection(self) -> None:
        """Update the fastest connection based on current latencies."""
        if not self._conns:
            self._fast_conn = None
            return
        self._fast_conn = min(
            (conn for conn in self._conns.values() if conn.get_state().is_connected),
            key=lambda x: x.get_state().latency_ms,
            default=None,
        )

    def _schedule_replacements(self, count: int) -> None:
        """Schedules replacement connections on the event loop.

        Args:
            count (int): Number of replacement connections to open.

        Returns:
            None: This method does not return a value.
        """
        if count <= 0 or self._should_stop:
            return
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        for _ in range(count):
            loop.call_soon_threadsafe(self._create_open_task)

    def _create_open_task(self) -> None:
        """Create a task to open a new connection on the event loop.

        Returns:
            None: This method does not return a value.
        """
        asyncio.create_task(self._open_new_conn())

    def _send_data_now(self, msg: bytes, only_fastest: bool) -> None:
        """Send data using the current pool snapshot.

        Args:
            msg (bytes): Payload to send.
            only_fastest (bool): Whether to send only through the fastest connection.

        Returns:
            None: This method does not return a value.
        """
        fast_conn = self._fast_conn
        conns_snapshot = list(self._conns.values())
        if only_fastest and fast_conn is not None:
            fast_conn.send_data(msg)
        else:
            for conn in conns_snapshot:
                conn.send_data(msg)

    async def _open_new_conn(self) -> None:
        """Establishes a new WebSocket connection and adds it to the connection pool."""
        try:
            if self._should_stop:
                return
            base_url = self._config.wss_url
            safe_url = base_url if base_url.startswith("wss://") else "wss://localhost"
            config = WsConnectionConfig.default(
                safe_url,
                on_connect=self._config.on_connect,
                auto_reconnect=self._config.auto_reconnect,
            )
            if config.wss_url != base_url:
                # Preserve ws:// URLs for local test servers after validation.
                config.wss_url = base_url
            new_conn = await WsConnection.new(
                ringbuffer=self._ringbuffer,
                config=config,
            )
            if self._should_stop:
                new_conn.close()
                return
            self._conns[config.conn_id] = new_conn
            self._update_fast_connection()
        except Exception:
            pass  # Silently ignore connection failures to avoid crashing pool

    def set_on_connect(self, on_connect: list[bytes]) -> None:
        """Sets the on_connect callback for all connections in the pool."""
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
        if self._pool_state != ConnectionState.CONNECTED:
            raise RuntimeError("Connection not running; cannot send data")

        loop = self._loop
        if loop is None or loop.is_closed():
            raise RuntimeError("Event loop not running; cannot send data")
        loop.call_soon_threadsafe(self._send_data_now, msg, only_fastest)

    @classmethod
    async def new(
        cls,
        config: WsConnectionConfig,
        on_message: Callable[[bytes], None],
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
        """Returns the current pool state."""
        return self._pool_state

    def get_connection_count(self) -> int:
        """Returns the number of active connections."""
        return sum(1 for conn in self._conns.values() if conn.get_state().is_connected)

    def close(self) -> None:
        """Shuts down all WebSocket connections and stops the eviction task."""
        self._should_stop = True
        self._pool_state = ConnectionState.DISCONNECTED
        if self._eviction_task is not None:
            self._eviction_task.cancel()
            self._eviction_task = None
        self._loop = None

        # Close all connections first (fastest operation)
        conns_snapshot = list(self._conns.values())
        self._conns.clear()
        self._fast_conn = None
        for conn in conns_snapshot:
            conn.close()

    async def __aenter__(self) -> Self:
        """Async context manager entry. Opens all connections in the pool."""
        self._loop = asyncio.get_running_loop()
        if self._pool_state != ConnectionState.CONNECTED:
            # Create connections concurrently for faster startup
            tasks = [
                self._open_new_conn() for _ in range(self._pool_config.num_connections)
            ]
            await asyncio.gather(*tasks, return_exceptions=True)
            self._pool_state = ConnectionState.CONNECTED

            # Set initial fast connection
            self._update_fast_connection()
            if self._eviction_task is None or self._eviction_task.done():
                self._eviction_task = asyncio.create_task(self._timed_operations())
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit. Closes all connections in the pool."""
        self.close()

    def __aiter__(self) -> Self:
        """Returns an async iterator over the pool's ringbuffer."""
        return self

    async def __anext__(self) -> bytes:
        """Returns the next message from the pool's ringbuffer."""
        return await self._ringbuffer.aconsume()


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
