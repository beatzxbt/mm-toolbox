"""Single WebSocket connection management.

Provides:
- WsSingle wrapper around WsConnection
- async context manager usage for connect/close
- async iteration and optional on_message callback handling
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Callable
from inspect import get_annotations
from typing import Self

import msgspec

from mm_toolbox.websocket.connection import (
    ConnectionState,
    WsConnection,
    WsConnectionConfig,
    _ReceiveBuffer,
)


_logger = logging.getLogger(__name__)


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
        self._ringbuffer = _ReceiveBuffer(config.receive_buffer_capacity)
        self._ringbuffer.configure(config.receive_buffer_max_bytes)
        self._ws_conn: WsConnection | None = None
        self._closed = False
        self._tasks: set[asyncio.Task] = set()
        self._start_task: asyncio.Task | None = None
        self._reconnect_task: asyncio.Task | None = None
        self._connect_task: asyncio.Task[WsConnection] | None = None

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

    def send_data(self, msg: bytes, only_fastest: bool = True) -> None:
        """Send data through the connection.

        Args:
            msg: Message to send
            only_fastest: Ignored for single connections (compatibility with WsPool)
        """
        if self._closed or self._ws_conn is None:
            raise RuntimeError("WebSocket connection is disconnected")
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

    def _track_task(self, task: asyncio.Task) -> asyncio.Task:
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def _connect(self) -> None:
        if self._closed:
            raise RuntimeError("WebSocket client is closed")
        if self._connect_task is not None and not self._connect_task.done():
            task = self._connect_task
        else:
            task = self._track_task(
                asyncio.create_task(WsConnection.new(self._ringbuffer, self._config))
            )
            self._connect_task = task
        try:
            conn = await task
        except BaseException:
            # The connector may have completed just before its awaiting owner
            # was cancelled. Its successful transport still needs an owner.
            if task.done() and not task.cancelled() and task.exception() is None:
                task.result().close()
            raise
        if self._closed:
            conn.close()
            raise RuntimeError("WebSocket client closed during connection")
        self._ws_conn = conn

    async def _maintain_connection(self) -> None:
        """Monitor disconnects and retry failures without racing shutdown."""
        delay = 1.0
        while not self._closed:
            if self._ws_conn is not None and self._ws_conn.is_connected():
                await asyncio.sleep(0.1)
                continue
            if not self._config.auto_reconnect:
                self._ringbuffer.stop()
                return
            await asyncio.sleep(delay)
            if self._closed:
                return
            try:
                await self._connect()
            except Exception as exc:
                _logger.warning("WebSocket reconnect failed; retrying: %s", exc)
                delay = min(delay * 2.0, 30.0)
            else:
                delay = 1.0

    def _start_monitor(self) -> None:
        if self._reconnect_task is None or self._reconnect_task.done():
            self._reconnect_task = self._track_task(
                asyncio.create_task(self._maintain_connection())
            )

    async def _consume_callbacks(self) -> None:
        """Dispatch queued messages until closed or reconnection is disabled."""
        while not self._closed:
            try:
                msg = await self.__anext__()
            except StopAsyncIteration:
                return
            if not self._closed:
                self._dispatch_message(msg)

    async def start(self) -> None:
        """Connect and consume callbacks until close; retry if configured."""
        if self._closed:
            raise RuntimeError("WebSocket client is closed")
        if self._start_task is not None and not self._start_task.done():
            raise RuntimeError("WebSocket callback consumer is already running")
        task = asyncio.current_task()
        assert task is not None
        self._start_task = task
        self._track_task(task)
        try:
            if self._ws_conn is None or not self._ws_conn.is_connected():
                try:
                    await self._connect()
                except Exception:
                    if not self._config.auto_reconnect:
                        raise
                    _logger.warning(
                        "Initial WebSocket connection failed; retrying", exc_info=True
                    )
            self._start_monitor()
            await self._consume_callbacks()
        except asyncio.CancelledError:
            if not self._closed:
                raise
        finally:
            self.close()
            self._tasks.discard(task)

    def close(self) -> None:
        """Permanently stop owned work and wake readers; queued bytes may drain.

        Repeated close calls are safe. Use a new client to start another session.
        """
        if self._closed:
            return
        self._closed = True
        self._ringbuffer.stop()
        try:
            current = asyncio.current_task()
        except RuntimeError:
            current = None
        for task in tuple(self._tasks):
            if task is not current and not task.done():
                task.cancel()
        if self._ws_conn is not None:
            try:
                self._ws_conn.close()
            except Exception as exc:
                _logger.warning("Error closing WebSocket connection: %s", exc)

    def get_config(self) -> WsConnectionConfig:
        """Return the connection configuration.

        Returns:
            WsConnectionConfig: Current configuration instance.

        """
        return self._config

    def get_connection_count(self) -> int:
        """Return the number of active connections.

        Returns:
            int: ``1`` if connected, otherwise ``0``.

        """
        if (
            not self._closed
            and self._ws_conn is not None
            and self._ws_conn.is_connected()
        ):
            return 1
        return 0

    def get_latency_ms(self) -> float:
        """Return the connection latency in milliseconds.

        Returns:
            float: Latency from the underlying connection, or ``0.0`` when
            disconnected.

        """
        if self._ws_conn is not None:
            return self._ws_conn.get_latency_ms()
        return 0.0

    def get_seq_id(self) -> int:
        """Return the connection sequence ID.

        Returns:
            int: Sequence ID from the underlying connection, or ``0`` when
            disconnected.

        """
        if self._ws_conn is not None:
            return self._ws_conn.get_seq_id()
        return 0

    def get_dropped_messages(self) -> int:
        """Return receive overwrites over this client's entire lifetime.

        Includes startup and replaced connections; closing never resets it.
        """
        return self._ringbuffer.get_dropped_messages()

    def get_state(self) -> ConnectionState:
        """Return the current connection state.

        Returns:
            ConnectionState: State of the underlying connection, or
            ``DISCONNECTED`` when none exists.

        """
        if not self._closed and self._ws_conn is not None:
            return self._ws_conn.get_state()
        return ConnectionState.DISCONNECTED

    async def __aenter__(self) -> Self:
        """Open once, failing promptly, then monitor for configured reconnects."""
        if self._closed:
            raise RuntimeError("WebSocket client is closed")
        try:
            if self._ws_conn is None or not self._ws_conn.is_connected():
                await self._connect()
            self._start_monitor()
        except BaseException:
            self.close()
            raise
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close and wait for owned connection and callback work to terminate."""
        self.close()
        current = asyncio.current_task()
        await asyncio.gather(
            *(task for task in tuple(self._tasks) if task is not current),
            return_exceptions=True,
        )

    def __aiter__(self) -> Self:
        """Return the single-consumer message iterator."""
        return self

    async def __anext__(self) -> bytes:
        """Drain queued bytes, then raise StopAsyncIteration after close."""
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
