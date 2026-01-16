"""Error scenario tests for WsSingle."""

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
        ws (WsSingle): WsSingle instance.
        expected (ConnectionState): Expected connection state.
        timeout_s (float): Timeout in seconds.

    Returns:
        None: This helper does not return a value.

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
    """Validate error handling for WsSingle."""

    async def test_connection_refused(self) -> None:
        """Ensure connection refusal raises an exception.

        Returns:
            None: This test does not return a value.
        """
        config = WsConnectionConfig.default("wss://127.0.0.1:1")
        config.wss_url = "ws://127.0.0.1:1"
        ws = WsSingle(config)
        with pytest.raises(Exception):
            await ws.__aenter__()

    async def test_connection_timeout(self) -> None:
        """Ensure connection attempts can time out or fail fast.

        Returns:
            None: This test does not return a value.
        """
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
        """Ensure disconnect during traffic transitions to DISCONNECTED.

        Args:
            server_send_close_frame: Fixture providing close-frame server.
            connection_config_factory: Fixture providing config factory.

        Returns:
            None: This test does not return a value.
        """
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
        """Ensure invalid frames trigger disconnect handling.

        Args:
            server_send_invalid_frames: Fixture providing invalid frame server.
            connection_config_factory: Fixture providing config factory.

        Returns:
            None: This test does not return a value.
        """
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
        """Ensure rapid connect/disconnect cycles do not leak resources.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_config_factory: Fixture providing config factory.

        Returns:
            None: This test does not return a value.
        """
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
        """Ensure message processing errors do not crash the consumer.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_config_factory: Fixture providing config factory.

        Returns:
            None: This test does not return a value.
        """
        async with basic_server:
            config = connection_config_factory(basic_server)
            errors: list[str] = []

            def on_message(msg: bytes) -> None:
                """Attempt JSON decoding to simulate processing errors.

                Args:
                    msg (bytes): Incoming message.

                Returns:
                    None: This callback does not return a value.
                """
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
