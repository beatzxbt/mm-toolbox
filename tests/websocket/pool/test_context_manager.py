"""Async context manager tests for WsPool."""

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
class TestWsPoolContextManager:
    """Validate WsPool async context manager behavior."""

    async def test_context_manager_starts_pool(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Ensure async with starts and stops the pool cleanly.

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
                await asyncio.sleep(0.3)
                assert pool.get_state() == ConnectionState.CONNECTED
                assert pool.get_connection_count() == pool_config.num_connections
            assert pool.get_state() == ConnectionState.DISCONNECTED

    async def test_context_manager_cleanup_on_exception(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Ensure exceptions inside context still cleanup pool resources.

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
            with pytest.raises(ValueError):
                async with pool:
                    await asyncio.sleep(0.2)
                    raise ValueError("boom")
            assert pool.get_state() == ConnectionState.DISCONNECTED

    async def test_context_manager_reentrant_usage(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Ensure pool can be started and stopped multiple times.

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
            for _ in range(2):
                async with pool:
                    await asyncio.sleep(0.2)
                    assert pool.get_state() == ConnectionState.CONNECTED
                assert pool.get_state() == ConnectionState.DISCONNECTED
