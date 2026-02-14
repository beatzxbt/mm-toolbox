"""Hash-history filtering tests for WsPool."""

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
    """Wait for pool to reach the expected active connection count.

    Args:
        pool (WsPool): Pool instance to monitor.
        expected (int): Expected active connection count.
        timeout_s (float): Timeout in seconds.

    Returns:
        None: This helper does not return a value.
    """
    start = asyncio.get_running_loop().time()
    while (asyncio.get_running_loop().time() - start) < timeout_s:
        if pool.get_connection_count() == expected:
            return
        await asyncio.sleep(0.05)
    raise AssertionError("Timed out waiting for pool connections")


@pytest.mark.asyncio
class TestWsPoolHashHistory:
    """Validate hash-history filtering behavior in WsPool."""

    async def test_filters_duplicate_cross_connection_broadcast(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Ensure one server broadcast yields a single downstream message.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_config_factory: Fixture providing config factory.

        Returns:
            None: This test does not return a value.
        """
        async with basic_server:
            config = connection_config_factory(basic_server)
            pool = await WsPool.new(
                config,
                on_message=noop_message_handler,
                pool_config=WsPoolConfig(
                    num_connections=3,
                    evict_interval_s=60,
                    hash_capacity=1024,
                ),
            )

            async with pool:
                await wait_for_pool_connections(pool, expected=3)
                await basic_server.send_to_all_clients(b"same-payload")

                first = await asyncio.wait_for(pool.__anext__(), timeout=1.0)
                assert first == b"same-payload"

                with pytest.raises(asyncio.TimeoutError):
                    await asyncio.wait_for(pool.__anext__(), timeout=0.3)

    async def test_hash_history_persists_after_message_is_consumed(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Ensure consuming one copy does not allow immediate duplicates through.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_config_factory: Fixture providing config factory.

        Returns:
            None: This test does not return a value.
        """
        async with basic_server:
            config = connection_config_factory(basic_server)
            pool = await WsPool.new(
                config,
                on_message=noop_message_handler,
                pool_config=WsPoolConfig(
                    num_connections=2,
                    evict_interval_s=60,
                    hash_capacity=1024,
                ),
            )

            async with pool:
                await wait_for_pool_connections(pool, expected=2)

                await basic_server.send_to_all_clients(b"persist-check")
                first = await asyncio.wait_for(pool.__anext__(), timeout=1.0)
                assert first == b"persist-check"

                await basic_server.send_to_all_clients(b"persist-check")
                with pytest.raises(asyncio.TimeoutError):
                    await asyncio.wait_for(pool.__anext__(), timeout=0.3)

    async def test_hash_history_eviction_allows_old_payload_again(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Ensure old hashes can be evicted from bounded hash history.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_config_factory: Fixture providing config factory.

        Returns:
            None: This test does not return a value.
        """
        async with basic_server:
            config = connection_config_factory(basic_server)
            pool = await WsPool.new(
                config,
                on_message=noop_message_handler,
                pool_config=WsPoolConfig(
                    num_connections=2,
                    evict_interval_s=60,
                    hash_capacity=2,
                ),
            )

            async with pool:
                await wait_for_pool_connections(pool, expected=2)

                expected = [b"a", b"b", b"c", b"a"]
                received: list[bytes] = []
                for payload in expected:
                    await basic_server.send_to_all_clients(payload)
                    received.append(await asyncio.wait_for(pool.__anext__(), timeout=1.0))

                assert received == expected

    async def test_hash_history_resets_on_pool_restart(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Ensure a fresh pool session starts with empty hash history.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_config_factory: Fixture providing config factory.

        Returns:
            None: This test does not return a value.
        """
        async with basic_server:
            config = connection_config_factory(basic_server)
            pool_a = await WsPool.new(
                config,
                on_message=noop_message_handler,
                pool_config=WsPoolConfig(
                    num_connections=2,
                    evict_interval_s=60,
                    hash_capacity=1024,
                ),
            )

            async with pool_a:
                await wait_for_pool_connections(pool_a, expected=2)
                await basic_server.send_to_all_clients(b"restart-check")
                assert (
                    await asyncio.wait_for(pool_a.__anext__(), timeout=1.0)
                    == b"restart-check"
                )

            pool_b = await WsPool.new(
                config,
                on_message=noop_message_handler,
                pool_config=WsPoolConfig(
                    num_connections=2,
                    evict_interval_s=60,
                    hash_capacity=1024,
                ),
            )
            async with pool_b:
                await wait_for_pool_connections(pool_b, expected=2)
                await basic_server.send_to_all_clients(b"restart-check")
                assert (
                    await asyncio.wait_for(pool_b.__anext__(), timeout=1.0)
                    == b"restart-check"
                )
