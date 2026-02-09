"""Single WebSocket connection management.

Provides:
- WsSingle wrapper around WsConnection
- async context manager usage for connect/close
- async iteration and optional on_message callback handling
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Self, get_type_hints

import msgspec

from mm_toolbox.ringbuffer.bytes import BytesRingBuffer
from mm_toolbox.websocket.connection import (
    ConnectionState,
    WsConnection,
    WsConnectionConfig,
)


class WsSingle:
    """A convenience wrapper around WsConnection for single-connection usage.

    There are two available patterns for consuming messages:
    1. Provide an on_message callback and call 'await ws.start()'
    2. Iterate over the connection by doing 'async for msg in ws' and
        handle the bytes directly as they arrive.
    """

    def __init__(
        self,
        config: WsConnectionConfig,
        on_message: Callable[[bytes], None] | None = None,
    ) -> None:
        """Initializes the single connection wrapper.

        Args:
            config (WsConnectionConfig): The configuration for the connection.
            on_message (Callable[[bytes], None], optional): Callback to handle
                received messages.

        """
        self._config = config
        self._ringbuffer = BytesRingBuffer(max_capacity=128, only_insert_unique=False)
        self._ws_conn: WsConnection | None = None

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

        self._on_message = on_message or self.__default_on_message

    def __default_on_message(self, msg: bytes) -> None:
        """Default callback for processing WebSocket messages. For convenience
        in debugging, this just prints the json."""
        try:
            print(msgspec.json.decode(msg))
        except Exception:
            print(msg)

    def set_on_connect(self, on_connect: list[bytes]) -> None:
        """Sets the messages to send upon connection establishment."""
        self._config.on_connect = on_connect
        if self._ws_conn is not None:
            self._ws_conn.set_on_connect(on_connect)

    def send_data(self, msg: bytes) -> None:
        """Sends data over the WebSocket connection."""
        if self._ws_conn is not None:
            self._ws_conn.send_data(msg)

    def _dispatch_message(self, msg: bytes) -> None:
        """Dispatches one message to the configured callback.

        Args:
            msg (bytes): Message payload to pass to the callback.

        Returns:
            None: This method does not return a value.
        """
        try:
            self._on_message(msg)
        except Exception as exc:
            print(f"Error in on_message callback: {exc}")

    async def _consume_callbacks(self) -> None:
        """Consumes ringbuffer messages and dispatches in arrival order.

        Returns:
            None: This coroutine runs until cancelled or an error occurs.
        """
        while True:
            self._dispatch_message(await self._ringbuffer.aconsume())

    async def start(self) -> None:
        """Opens the WebSocket connection and sends any on_connect messages."""
        if self._config.auto_reconnect:
            conn_iter = WsConnection.new_with_reconnect(self._ringbuffer, self._config)
            async for conn in conn_iter:
                self._ws_conn = conn
                try:
                    await self._consume_callbacks()
                except Exception as exc:
                    print(f"Error consuming WebSocket messages: {exc}")
        else:
            self._ws_conn = await WsConnection.new(self._ringbuffer, self._config)
            try:
                await self._consume_callbacks()
            except Exception as exc:
                print(f"Error consuming WebSocket messages: {exc}")

    def close(self) -> None:
        """Closes the WebSocket connection gracefully."""
        if self._ws_conn is not None:
            try:
                self._ws_conn.close()
            except Exception as exc:
                print(f"Error closing WebSocket connection: {exc}")

    def get_config(self) -> WsConnectionConfig:
        """Retrieves the connection's configuration."""
        return self._config

    def get_state(self) -> ConnectionState:
        """Retrieves the connection state."""
        if self._ws_conn is not None:
            return self._ws_conn.get_state().state
        return ConnectionState.DISCONNECTED

    async def __aenter__(self) -> Self:
        """Async context manager entry. Opens the connection."""
        if self._ws_conn is None:
            if self._config.auto_reconnect:
                conn_iter = WsConnection.new_with_reconnect(
                    self._ringbuffer, self._config
                )
                try:
                    self._ws_conn = await conn_iter.__aiter__().__anext__()
                except Exception:
                    # Fall back to single connection if reconnect fails
                    self._ws_conn = await WsConnection.new(
                        self._ringbuffer, self._config
                    )
            else:
                self._ws_conn = await WsConnection.new(self._ringbuffer, self._config)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit. Closes the connection."""
        self.close()

    def __aiter__(self) -> Self:
        """Returns an async iterator over the connection."""
        return self

    async def __anext__(self) -> bytes:
        """Returns the next message from the connection in arrival order."""
        return await self._ringbuffer.aconsume()


if __name__ == "__main__":
    import asyncio

    import msgspec

    async def _single_websocket_example() -> None:
        """Example usage of the WsSingle class."""
        config = WsConnectionConfig.default(
            wss_url="wss://fstream.binance.com/ws/btcusdt@bookTicker"
        )
        async with WsSingle(config) as ws:
            msg_count = 0
            async for msg in ws:
                print(f"Received message {msg_count}: {msgspec.json.decode(msg)}")
                msg_count += 1
                if msg_count > 100:
                    print(f"Final state: {ws.get_state()}")
                    break

    asyncio.run(_single_websocket_example())
