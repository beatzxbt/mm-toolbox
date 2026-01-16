"""Error scenario tests for WsPool."""

from __future__ import annotations

import asyncio

import pytest

from mm_toolbox.websocket.connection import ConnectionState
from mm_toolbox.websocket.pool import WsPool, WsPoolConfig


def noop_message_handler(msg: bytes) -> None:
    """No-op message handler for pool tests.

    Args:
        msg (bytes): Incoming message payload.

    Returns:
        None: This handler does not return a value.
    """
    return None


@pytest.mark.asyncio
class TestWsPoolErrorScenarios:
    """Validate error handling paths in WsPool."""

    async def test_pool_connection_failures(
        self,
        server_reject_connections,
        connection_config_factory,
    ) -> None:
        """Ensure pool tolerates connection failures without crashing.

        Args:
            server_reject_connections: Fixture providing rejecting server.
            connection_config_factory: Fixture providing config factory.

        Returns:
            None: This test does not return a value.
        """
        async with server_reject_connections:
            config = connection_config_factory(server_reject_connections)
            pool_config = WsPoolConfig(num_connections=2, evict_interval_s=60)
            pool = await WsPool.new(
                config, on_message=noop_message_handler, pool_config=pool_config
            )
            async with pool:
                await asyncio.sleep(0.3)
                assert pool.get_connection_count() == 0
            pool.close()

    async def test_pool_mid_stream_disconnect(
        self,
        server_send_close_frame,
        connection_config_factory,
    ) -> None:
        """Ensure pool handles mid-stream disconnects gracefully.

        Args:
            server_send_close_frame: Fixture providing close-frame server.
            connection_config_factory: Fixture providing config factory.

        Returns:
            None: This test does not return a value.
        """
        async with server_send_close_frame:
            config = connection_config_factory(server_send_close_frame)
            pool_config = WsPoolConfig(num_connections=2, evict_interval_s=60)
            pool = await WsPool.new(
                config, on_message=noop_message_handler, pool_config=pool_config
            )
            async with pool:
                await asyncio.sleep(0.3)
                pool.send_data(b"trigger-close", only_fastest=False)
                await asyncio.sleep(0.3)
                assert pool.get_state() == ConnectionState.CONNECTED
            pool.close()

    async def test_pool_protocol_error_invalid_frame(
        self,
        server_send_invalid_frames,
        connection_config_factory,
    ) -> None:
        """Ensure pool survives protocol errors from the server.

        Args:
            server_send_invalid_frames: Fixture providing invalid frame server.
            connection_config_factory: Fixture providing config factory.

        Returns:
            None: This test does not return a value.
        """
        async with server_send_invalid_frames:
            config = connection_config_factory(server_send_invalid_frames)
            pool_config = WsPoolConfig(num_connections=2, evict_interval_s=60)
            pool = await WsPool.new(
                config, on_message=noop_message_handler, pool_config=pool_config
            )
            async with pool:
                await asyncio.sleep(0.3)
                pool.send_data(b"trigger-invalid", only_fastest=False)
                await asyncio.sleep(0.3)
                assert pool.get_state() == ConnectionState.CONNECTED
            pool.close()

    async def test_pool_send_data_when_closed(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Ensure send_data raises once the pool is closed.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_config_factory: Fixture providing config factory.

        Returns:
            None: This test does not return a value.
        """
        async with basic_server:
            config = connection_config_factory(basic_server)
            pool_config = WsPoolConfig(num_connections=2, evict_interval_s=60)
            pool = await WsPool.new(
                config, on_message=noop_message_handler, pool_config=pool_config
            )
            async with pool:
                await asyncio.sleep(0.2)
            with pytest.raises(RuntimeError):
                pool.send_data(b"fail", only_fastest=False)
