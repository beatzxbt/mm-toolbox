"""Send operations tests for WsPool."""

from __future__ import annotations

import asyncio

import pytest

from mm_toolbox.websocket.pool import WsPool, WsPoolConfig


def noop_message_handler(msg: bytes) -> None:
    """No-op message handler for pool tests.

    Args:
        msg (bytes): Incoming message payload.

    Returns:
        None: This handler does not return a value.
    """
    return None


async def wait_for_pool_connections(
    pool: WsPool, expected: int, timeout_s: float = 2.0
) -> None:
    """Wait for the pool to reach the expected connection count.

    Args:
        pool (WsPool): Pool instance to monitor.
        expected (int): Expected connection count.
        timeout_s (float): Timeout in seconds.

    Returns:
        None: This helper does not return a value.

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
    """Validate pool send_data behaviors."""

    async def test_send_data_with_only_fastest(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Ensure only_fastest sends through a single connection.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_config_factory: Fixture providing config factory.

        Returns:
            None: This test does not return a value.
        """
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
        """Ensure multicast sends to all connections when only_fastest is False.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_config_factory: Fixture providing config factory.

        Returns:
            None: This test does not return a value.
        """
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

    async def test_send_data_requires_running_loop(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Ensure send_data raises when event loop is not running.

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
            with pytest.raises(RuntimeError):
                pool.send_data(b"fail", only_fastest=False)
