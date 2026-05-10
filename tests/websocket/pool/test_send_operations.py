"""Send operations tests for WsPool.

Layer-2 tests validating pool-level send routing: fastest-only vs
multicast, and safe failure when the pool is not connected.
"""

from __future__ import annotations

import asyncio

import pytest

from mm_toolbox.websocket.pool import WsPool, WsPoolConfig


def noop_message_handler(msg: bytes) -> None:
    """No-op message handler for pool tests."""
    return None


async def wait_for_pool_connections(
    pool: WsPool, expected: int, timeout_s: float = 2.0
) -> None:
    """Wait for the pool to reach the expected connection count.

    Args:
        pool: Pool instance to monitor.
        expected: Expected connection count.
        timeout_s: Timeout in seconds.

    Raises:
        AssertionError: If the expected count is not reached.
    """
    start = asyncio.get_running_loop().time()
    while (asyncio.get_running_loop().time() - start) < timeout_s:
        if pool.get_connection_count() == expected:
            return
        await asyncio.sleep(0.05)
    raise AssertionError("Timed out waiting for pool connections")


@pytest.mark.asyncio
class TestWsPoolSendOperations:
    """Layer-2 tests for pool send_data behaviors."""

    async def test_send_data_with_only_fastest(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Given a pool with 3 connections, When only_fastest=True, Then exactly one server-side receive is recorded.

        Fastest-only routing minimises outbound bandwidth for idempotent
messages like heartbeats or subscriptions."""
        async with basic_server:
            config = connection_config_factory(basic_server)
            pool_config = WsPoolConfig(num_connections=3, evict_interval_s=60)
            pool = await WsPool.new(
                config, on_message=noop_message_handler, pool_config=pool_config
            )

            async with pool:
                await wait_for_pool_connections(pool, pool_config.num_connections)
                assert pool._fast_conn is not None
                before = len(basic_server.get_received_messages())
                pool.send_data(b"fastest-only", only_fastest=True)
                await asyncio.sleep(0.2)
                after = len(basic_server.get_received_messages())
                assert after - before == 1

    async def test_send_data_multicast(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Given a pool with 3 connections, When only_fastest=False, Then the server receives one copy per connection.

        Multicast is used for state-changing messages that must reach all
endpoints (e.g., subscription updates)."""
        async with basic_server:
            config = connection_config_factory(basic_server)
            pool_config = WsPoolConfig(num_connections=3, evict_interval_s=60)
            pool = await WsPool.new(
                config, on_message=noop_message_handler, pool_config=pool_config
            )

            async with pool:
                await wait_for_pool_connections(pool, pool_config.num_connections)
                before = len(basic_server.get_received_messages())
                pool.send_data(b"broadcast", only_fastest=False)
                await asyncio.sleep(0.3)
                after = len(basic_server.get_received_messages())
                assert after - before == pool_config.num_connections

    async def test_send_data_requires_connected_state(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Given a pool that has not been entered, When send_data is called, Then RuntimeError is raised.

        This prevents accidental no-ops when the caller forgets to start
the pool before sending."""
        async with basic_server:
            config = connection_config_factory(basic_server)
            pool_config = WsPoolConfig(num_connections=2, evict_interval_s=60)
            pool = await WsPool.new(
                config, on_message=noop_message_handler, pool_config=pool_config
            )
            with pytest.raises(RuntimeError, match="Connection not running"):
                pool.send_data(b"fail", only_fastest=False)
