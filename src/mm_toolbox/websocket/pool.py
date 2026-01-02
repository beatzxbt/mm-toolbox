"""WebSocket connection pool management."""

import asyncio
import inspect
import threading
import time
from collections.abc import Callable
from typing import Any, Self

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
            if (
                len(sig.parameters) != 1
                or list(sig.parameters.values())[0].annotation is not bytes
            ):
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
        self._pending_connections = []  # Queue for async connection creation

        self._timed_operations_thread = threading.Thread(
            target=self._timed_operations,
            daemon=True,
        )
        self._timed_operations_thread.start()

    def __default_on_message(self, msg: bytes) -> None:
        """Default callback for processing WebSocket messages. For convenience
        in debugging, this just prints the json."""
        try:
            print(json_decode(msg))
        except Exception as e:
            print(f"Error decoding message: {e}")

    def _timed_operations(self) -> None:
        """Enforces the configuration to evict slow connections."""
        time_now_s = time_s()
        next_eviction_time = time_now_s + self._pool_config.evict_interval_s

        while not self._should_stop:
            if (
                self._pool_state != ConnectionState.CONNECTED
                or time_s() < next_eviction_time
            ):
                time.sleep(1)
                continue

            # Update fast connection based on latency (cheapest operation first)
            self._update_fast_connection()

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
                old_conn_id = old_conn.get_config().conn_id
                old_conn.close()
                del self._conns[old_conn_id]
                self._pending_connections.append(None)  # Signal need for new connection

            next_eviction_time += self._pool_config.evict_interval_s

    def _update_fast_connection(self) -> None:
        """Update the fastest connection based on current latencies."""
        if self._conns:
            self._fast_conn = min(
                (
                    conn
                    for conn in self._conns.values()
                    if conn.get_state().is_connected
                ),
                key=lambda x: x.get_state().latency_ms,
                default=None,
            )

    async def _open_new_conn(self) -> None:
        """Establishes a new WebSocket connection and adds it to the connection pool."""
        try:
            config = WsConnectionConfig.default(self._config.wss_url)
            new_conn = await WsConnection.new(
                ringbuffer=self._ringbuffer,
                config=config,
            )
            self._conns[config.conn_id] = new_conn
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

        if only_fastest and self._fast_conn is not None:
            self._fast_conn.send_data(msg)
        else:
            for conn in self._conns.values():
                conn.send_data(msg)

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
        return len(
            [conn for conn in self._conns.values() if conn.get_state().is_connected]
        )

    def close(self) -> None:
        """Shuts down all WebSocket connections and stops the eviction task."""
        self._should_stop = True
        self._pool_state = ConnectionState.DISCONNECTED

        # Close all connections first (fastest operation)
        for conn in self._conns.values():
            conn.close()

        # Don't join thread - let it stop naturally to avoid blocking

    async def __aenter__(self) -> Self:
        """Async context manager entry. Opens all connections in the pool."""
        if self._pool_state != ConnectionState.CONNECTED:
            # Create connections concurrently for faster startup
            tasks = [
                self._open_new_conn() for _ in range(self._pool_config.num_connections)
            ]
            await asyncio.gather(*tasks, return_exceptions=True)
            self._pool_state = ConnectionState.CONNECTED

            # Set initial fast connection
            self._update_fast_connection()
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
