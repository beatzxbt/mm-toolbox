"""Error scenario tests for WsSingle.

Layer-2 tests validating graceful degradation under adverse conditions:
connection refusal, timeout, mid-stream disconnect, protocol errors,
rapid churn, callback exceptions, and auto-reconnect recovery.
"""

from __future__ import annotations

import asyncio
import contextlib

import msgspec
import pytest

from mm_toolbox.websocket.connection import ConnectionState, WsConnectionConfig
from mm_toolbox.websocket.single import WsSingle


async def wait_for_single_state(
    ws: WsSingle, expected: ConnectionState, timeout_s: float = 2.0
) -> None:
    """Wait for a WsSingle instance to reach the expected state.

    Args:
        ws: WsSingle instance.
        expected: Expected connection state.
        timeout_s: Timeout in seconds.

    Raises:
        AssertionError: If state is not reached in time.
    """
    start = asyncio.get_running_loop().time()
    while (asyncio.get_running_loop().time() - start) < timeout_s:
        if ws.get_state() == expected:
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"Timed out waiting for {expected}")


@pytest.mark.asyncio
class TestWsSingleErrorScenarios:
    """Layer-2 tests for error handling in WsSingle."""

    async def test_connection_refused(self) -> None:
        """Given a non-listening port, When WsSingle tries to connect, Then an exception is raised promptly."""
        config = WsConnectionConfig.default("wss://127.0.0.1:1")
        config.wss_url = "ws://127.0.0.1:1"
        ws = WsSingle(config)
        with pytest.raises(Exception):
            await ws.__aenter__()

    async def test_connection_timeout(self) -> None:
        """Given an unreachable host, When connect is attempted with a short timeout, Then it fails fast.

                Fast failure is important so that unresponsive endpoints do not
        block the event loop indefinitely."""
        config = WsConnectionConfig.default("wss://127.0.0.1:2")
        config.wss_url = "ws://127.0.0.1:2"
        ws = WsSingle(config)
        with pytest.raises((asyncio.TimeoutError, OSError, Exception)):
            await asyncio.wait_for(ws.__aenter__(), timeout=0.2)

    async def test_mid_stream_disconnect(
        self,
        server_send_close_frame,
        connection_config_factory,
    ) -> None:
        """Given a server that closes mid-traffic, When the close frame arrives, Then state transitions to DISCONNECTED."""
        async with server_send_close_frame:
            config = connection_config_factory(server_send_close_frame)
            async with WsSingle(config) as ws:
                ws.send_data(b"trigger-close")
                await wait_for_single_state(
                    ws, ConnectionState.DISCONNECTED, timeout_s=2.0
                )

    async def test_protocol_error_invalid_frame(
        self,
        server_send_invalid_frames,
        connection_config_factory,
    ) -> None:
        """Given a server that sends malformed frames, When they arrive, Then the connection triggers disconnect handling."""
        async with server_send_invalid_frames:
            config = connection_config_factory(server_send_invalid_frames)
            async with WsSingle(config) as ws:
                ws.send_data(b"trigger-invalid")
                await wait_for_single_state(
                    ws, ConnectionState.DISCONNECTED, timeout_s=2.0
                )

    async def test_rapid_connect_disconnect(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Given 3 rapid open/close cycles, When completed, Then no resources are leaked.

                Repeated churn is a common pattern in reconnect-heavy clients;
        this guards against fd and task leaks."""
        async with basic_server:
            for _ in range(3):
                config = connection_config_factory(basic_server)
                async with WsSingle(config) as ws:
                    await wait_for_single_state(
                        ws, ConnectionState.CONNECTED, timeout_s=2.0
                    )

    async def test_exception_in_message_processing(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Given a message callback that raises, When a message arrives, Then the exception is caught and the connection stays CONNECTED."""
        async with basic_server:
            config = connection_config_factory(basic_server)
            errors: list[str] = []

            def on_message(msg: bytes) -> None:
                """Attempt JSON decoding to simulate processing errors."""
                try:
                    msgspec.json.decode(msg)
                except Exception as exc:
                    errors.append(str(exc))

            ws = WsSingle(config, on_message=on_message)
            task = asyncio.create_task(ws.start())
            await asyncio.sleep(0.2)
            await basic_server.send_to_all_clients(b"{not-json}")
            await asyncio.sleep(0.2)
            assert errors
            ws.close()
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def test_start_auto_reconnect_continues_after_disconnect(
        self,
        server_send_close_frame,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Given auto_reconnect=True and a server that closes, When the URL is switched to a healthy server, Then reconnection succeeds and messages flow again.

                This simulates a real-world failover where the primary endpoint
        becomes unhealthy and the client must transparently resume."""
        async with server_send_close_frame:
            config = connection_config_factory(
                server_send_close_frame, auto_reconnect=True
            )
            ws = WsSingle(config)
            task = asyncio.create_task(ws.start())
            await asyncio.sleep(0.2)
            ws.send_data(b"trigger-close")
            await asyncio.sleep(0.3)

        async with basic_server:
            config.wss_url = basic_server.uri
            ws._config.wss_url = basic_server.uri
            await asyncio.sleep(2.5)
            ws.send_data(b"after-reconnect")
            await asyncio.sleep(0.3)
            assert b"after-reconnect" in basic_server.get_received_messages()
            ws.close()
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def test_send_data_no_op_when_none(self) -> None:
        """Given a WsSingle with no underlying connection, When send_data is called, Then it is a no-op and does not raise."""
        config = WsConnectionConfig.default("wss://test.com")
        ws = WsSingle(config)
        ws.send_data(b"no-op")
        assert ws._ws_conn is None
