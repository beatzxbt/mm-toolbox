"""Message callback tests for WsSingle.

Layer-2 tests validating on_message callback invocation, exception handling,
fragmented message delivery, concurrent callback + iteration, and
set_on_connect propagation.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from mm_toolbox.websocket.connection import ConnectionState, WsConnectionConfig
from mm_toolbox.websocket.single import WsSingle


async def shutdown_ws_task(ws: WsSingle, task: asyncio.Task[None]) -> None:
    """Stop a WsSingle start() task gracefully.

    Args:
        ws: WsSingle instance to close.
        task: Task running ws.start().
    """
    ws.close()
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


class TestWsSingleCallbackValidation:
    """Layer-2 tests for WsSingle callback signature validation."""

    def test_callback_validation(self) -> None:
        """Given valid and invalid callbacks, Then only the valid one is accepted.

        Early validation prevents frame-dispatch runtime errors."""
        config = WsConnectionConfig.default("wss://test.com")

        def valid_callback(msg: bytes) -> None:
            return None

        ws = WsSingle(config, on_message=valid_callback)
        assert ws._on_message is valid_callback

        def invalid_no_args() -> None:
            return None

        def invalid_two_args(_one: bytes, _two: bytes) -> None:
            return None

        with pytest.raises(ValueError):
            WsSingle(config, on_message=invalid_no_args)

        with pytest.raises(ValueError):
            WsSingle(config, on_message=invalid_two_args)


@pytest.mark.asyncio
class TestWsSingleMessageCallbacks:
    """Layer-2 tests for on_message callback behavior."""

    async def test_on_message_callback_invoked(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Given an on_message callback, When messages arrive, Then the callback fires for each one."""
        async with basic_server:
            config = connection_config_factory(basic_server)
            received: list[bytes] = []

            def on_message(msg: bytes) -> None:
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
        """Given a callback that always raises, When a message arrives, Then the connection stays CONNECTED.

                Callback exceptions must not tear down the transport; otherwise a
        buggy handler would kill the entire websocket session."""
        async with basic_server:
            config = connection_config_factory(basic_server)

            def on_message(_msg: bytes) -> None:
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
        """Given a fragmenting server, When a message arrives, Then the callback receives the reassembled payload.

                Fragmented frames must be buffered internally; delivering partial
        fragments to the callback would break message parsers."""
        async with server_with_fragmentation:
            config = connection_config_factory(server_with_fragmentation)
            received: list[bytes] = []

            def on_message(msg: bytes) -> None:
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
        """Given one WsSingle with a callback and another with async iteration, When a message is broadcast, Then both receive it.

                This validates that callback and iterator paths are independent
        and do not contend for the same ringbuffer slot."""
        async with basic_server:
            config1 = connection_config_factory(basic_server)
            config2 = connection_config_factory(basic_server)
            received_cb: list[bytes] = []
            received_iter: list[bytes] = []

            def on_message(msg: bytes) -> None:
                received_cb.append(msg)

            ws_cb = WsSingle(config1, on_message=on_message)
            ws_iter = WsSingle(config2)
            task_cb = asyncio.create_task(ws_cb.start())

            async with ws_iter:

                async def _collector() -> None:
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

    async def test_consume_callbacks_state_check(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Given a running WsSingle, When it is shut down, Then the internal callback consumer exits cleanly."""
        async with basic_server:
            config = connection_config_factory(basic_server)
            ws = WsSingle(config)
            task = asyncio.create_task(ws.start())
            await asyncio.sleep(0.2)
            await shutdown_ws_task(ws, task)

    async def test_set_on_connect_propagates(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Given a connected WsSingle, When set_on_connect is called, Then the update propagates to both the wrapper and the underlying connection.

                Propagation is required so that reconnections use the new payload
        rather than the stale one."""
        async with basic_server:
            config = connection_config_factory(basic_server)
            async with WsSingle(config) as ws:
                await asyncio.sleep(0.2)
                assert ws.get_state() == ConnectionState.CONNECTED
                new_payload = [b'{"sub": "test"}']
                ws.set_on_connect(new_payload)
                assert ws.get_config().on_connect == new_payload
                assert ws._ws_conn is not None
                assert ws._ws_conn.get_config().on_connect == new_payload

    async def test_empty_message_callback(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Given an empty payload broadcast, When the callback fires, Then it receives an empty bytes object.

                Empty frames are valid websocket messages and must not be silently
        dropped by the callback dispatcher."""
        async with basic_server:
            config = connection_config_factory(basic_server)
            received: list[bytes] = []

            def on_message(msg: bytes) -> None:
                received.append(msg)

            ws = WsSingle(config, on_message=on_message)
            task = asyncio.create_task(ws.start())
            await asyncio.sleep(0.2)
            await basic_server.send_to_all_clients(b"")
            await asyncio.sleep(0.2)
            assert b"" in received
            await shutdown_ws_task(ws, task)
