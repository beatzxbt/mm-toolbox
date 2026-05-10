"""Async context manager tests for WsPool.

Layer-2 tests validating that WsPool can be used correctly inside
`async with` blocks, including normal startup, exception cleanup,
reentrant usage, and fallback when all connections are rejected.
"""

from __future__ import annotations

import asyncio

import pytest

from mm_toolbox.websocket.connection import ConnectionState
from mm_toolbox.websocket.pool import WsPool, WsPoolConfig


def noop_message_handler(msg: bytes) -> None:
    """No-op message handler for pool tests."""
    return None


@pytest.mark.asyncio
class TestWsPoolContextManager:
    """Layer-2 tests for WsPool async context manager behavior."""

    async def test_context_manager_starts_pool(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Given a pool, When entered via async with, Then it starts all connections and exits to DISCONNECTED."""
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
        """Given an exception inside the context body, When raised, Then the pool still cleans up and ends in DISCONNECTED.

        This prevents background tasks from leaking when user code fails."""
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
        """Given a pool, When started and stopped twice, Then each cycle is independent and ends in DISCONNECTED."""
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

    async def test_aenter_raises_when_all_connections_fail(
        self,
        server_reject_connections,
        connection_config_factory,
    ) -> None:
        """Given a rejecting server, When the pool is entered, Then it starts but reports zero healthy connections.

        This documents the behavior where handshake succeeds but the
server immediately closes, leaving the pool empty but not crashed."""
        async with server_reject_connections:
            config = connection_config_factory(server_reject_connections)
            pool_config = WsPoolConfig(num_connections=2, evict_interval_s=60)
            pool = await WsPool.new(
                config, on_message=noop_message_handler, pool_config=pool_config
            )
            async with pool:
                await asyncio.sleep(0.3)
                assert pool.get_connection_count() == 0
