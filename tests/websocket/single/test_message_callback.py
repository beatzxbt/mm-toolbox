"""Message callback tests for WsSingle."""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from mm_toolbox.websocket.connection import ConnectionState
from mm_toolbox.websocket.single import WsSingle


async def shutdown_ws_task(ws: WsSingle, task: asyncio.Task[None]) -> None:
    """Stop a WsSingle start() task gracefully.

    Args:
        ws (WsSingle): WsSingle instance to close.
        task (asyncio.Task[None]): Task running ws.start().

    Returns:
        None: This helper does not return a value.
    """
    ws.close()
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


class TestWsSingleCallbackValidation:
    """Validate WsSingle callback signature validation."""

    def test_callback_validation(self) -> None:
        """Ensure invalid callback signatures raise errors.

        Returns:
            None: This test does not return a value.
        """
        from mm_toolbox.websocket.connection import WsConnectionConfig

        config = WsConnectionConfig.default("wss://test.com")

        def valid_callback(msg: bytes) -> None:
            """Accept a single bytes payload.

            Args:
                msg (bytes): Incoming message.

            Returns:
                None: This callback does not return a value.
            """
            return None

        ws = WsSingle(config, on_message=valid_callback)
        assert ws._on_message is valid_callback

        def invalid_no_args() -> None:
            """Callback missing required bytes parameter.

            Returns:
                None: This callback does not return a value.
            """
            return None

        def invalid_two_args(_one: bytes, _two: bytes) -> None:
            """Callback with too many parameters.

            Args:
                _one (bytes): First payload.
                _two (bytes): Second payload.

            Returns:
                None: This callback does not return a value.
            """
            return None

        with pytest.raises(ValueError):
            WsSingle(config, on_message=invalid_no_args)

        with pytest.raises(ValueError):
            WsSingle(config, on_message=invalid_two_args)


@pytest.mark.asyncio
class TestWsSingleMessageCallbacks:
    """Validate on_message callback behavior."""

    async def test_on_message_callback_invoked(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Ensure on_message callback fires for each message.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_config_factory: Fixture providing config factory.

        Returns:
            None: This test does not return a value.
        """
        async with basic_server:
            config = connection_config_factory(basic_server)
            received: list[bytes] = []

            def on_message(msg: bytes) -> None:
                """Append incoming messages to the collection.

                Args:
                    msg (bytes): Incoming message.

                Returns:
                    None: This callback does not return a value.
                """
                received.append(msg)

            ws = WsSingle(config, on_message=on_message)
            task = asyncio.create_task(ws.start())
            await asyncio.sleep(0.2)
            await basic_server.send_to_all_clients(b"m1")
            await basic_server.send_to_all_clients(b"m2")
            await asyncio.sleep(0.2)
            assert received[:2] == [b"m1", b"m2"]
            await shutdown_ws_task(ws, task)

    async def test_on_message_callback_exception_handling(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Ensure callback exceptions do not crash the connection.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_config_factory: Fixture providing config factory.

        Returns:
            None: This test does not return a value.
        """
        async with basic_server:
            config = connection_config_factory(basic_server)

            def on_message(_msg: bytes) -> None:
                """Raise to simulate callback errors.

                Args:
                    _msg (bytes): Incoming message.

                Returns:
                    None: This callback does not return a value.
                """
                raise ValueError("boom")

            ws = WsSingle(config, on_message=on_message)
            task = asyncio.create_task(ws.start())
            await asyncio.sleep(0.2)
            await basic_server.send_to_all_clients(b"oops")
            await asyncio.sleep(0.2)
            assert ws.get_state() == ConnectionState.CONNECTED
            await shutdown_ws_task(ws, task)

    async def test_on_message_with_fragmented_message(
        self,
        server_with_fragmentation,
        connection_config_factory,
    ) -> None:
        """Ensure fragmented messages are reassembled before callbacks.

        Args:
            server_with_fragmentation: Fixture providing fragmented server.
            connection_config_factory: Fixture providing config factory.

        Returns:
            None: This test does not return a value.
        """
        async with server_with_fragmentation:
            config = connection_config_factory(server_with_fragmentation)
            received: list[bytes] = []

            def on_message(msg: bytes) -> None:
                """Append fragmented message payloads.

                Args:
                    msg (bytes): Incoming message.

                Returns:
                    None: This callback does not return a value.
                """
                received.append(msg)

            ws = WsSingle(config, on_message=on_message)
            task = asyncio.create_task(ws.start())
            await asyncio.sleep(0.2)
            await server_with_fragmentation.send_to_all_clients(b"fragmented")
            await asyncio.sleep(0.2)
            assert received == [b"fragmented"]
            await shutdown_ws_task(ws, task)

    async def test_on_message_concurrent_with_async_iteration(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Ensure callback and async iteration can run in parallel instances.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_config_factory: Fixture providing config factory.

        Returns:
            None: This test does not return a value.
        """
        async with basic_server:
            config1 = connection_config_factory(basic_server)
            config2 = connection_config_factory(basic_server)
            received_cb: list[bytes] = []
            received_iter: list[bytes] = []

            def on_message(msg: bytes) -> None:
                """Collect callback messages for concurrency validation.

                Args:
                    msg (bytes): Incoming message.

                Returns:
                    None: This callback does not return a value.
                """
                received_cb.append(msg)

            ws_cb = WsSingle(config1, on_message=on_message)
            ws_iter = WsSingle(config2)
            task_cb = asyncio.create_task(ws_cb.start())

            async with ws_iter:

                async def _collector() -> None:
                    """Collect a single message via async iteration.

                    Returns:
                        None: This helper does not return a value.
                    """
                    async for msg in ws_iter:
                        received_iter.append(msg)
                        if len(received_iter) >= 1:
                            break

                collect_task = asyncio.create_task(_collector())
                await asyncio.sleep(0.2)
                await basic_server.send_to_all_clients(b"shared")
                await asyncio.wait_for(collect_task, timeout=2.0)

            await asyncio.sleep(0.2)
            assert received_iter == [b"shared"]
            assert received_cb
            await shutdown_ws_task(ws_cb, task_cb)
