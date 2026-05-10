"""Hash-history filtering tests for WsPool.

Layer-2 tests validating that WsPool deduplicates messages across its
connections using a bounded hash-history ringbuffer.

Key coverage:
- One server broadcast to 3 connections yields a single downstream message.
- Consuming a message does not allow immediate duplicates through.
- Old hashes can be evicted when capacity is exceeded.
- A fresh pool session starts with empty hash history.
- Ringbuffer overflow evicts oldest hashes, allowing old payloads again.
- Empty payload deduplication works correctly.
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
    """Wait for pool to reach the expected active connection count.

    Args:
        pool: Pool instance to monitor.
        expected: Expected active connection count.
        timeout_s: Timeout in seconds.

    Raises:
        AssertionError: If the expected count is not reached in time.
    """
    start = asyncio.get_running_loop().time()
    while (asyncio.get_running_loop().time() - start) < timeout_s:
        if pool.get_connection_count() == expected:
            return
        await asyncio.sleep(0.05)
    raise AssertionError("Timed out waiting for pool connections")


@pytest.mark.asyncio
class TestWsPoolHashHistory:
    """Layer-2 tests for hash-history filtering behavior in WsPool."""

    async def test_filters_duplicate_cross_connection_broadcast(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Given 3 connections and one broadcast, When the pool iterates, Then only one copy is yielded.

                Without deduplication downstream consumers would see N copies of
        every message, breaking aggregation logic."""
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
        """Given a consumed message in hash history, When the same payload is broadcast again, Then it is still deduplicated.

                Consumption must not clear the hash entry; otherwise duplicate
        floods would pass through after the first read."""
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
        """Given a tiny hash capacity, When enough unique messages pass through, Then an old payload can reappear.

                Bounded history is required for memory safety; this test verifies
        that eviction works and does not permanently blacklist payloads."""
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
                    received.append(
                        await asyncio.wait_for(pool.__anext__(), timeout=1.0)
                    )

                assert received == expected

    async def test_hash_history_resets_on_pool_restart(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Given two separate pool sessions, When the same payload is broadcast to each, Then both yield it because history does not leak across sessions.

                Cross-session state leakage would cause the second pool to miss
        messages that were seen by the first."""
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

    async def test_hash_history_ringbuffer_overflow(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Given a hash capacity of 4, When 5 unique payloads arrive, Then the oldest is evicted and can reappear.

        This tests the LRU-like behavior of the bounded history buffer."""
        async with basic_server:
            config = connection_config_factory(basic_server)
            pool = await WsPool.new(
                config,
                on_message=noop_message_handler,
                pool_config=WsPoolConfig(
                    num_connections=2,
                    evict_interval_s=60,
                    hash_capacity=4,
                ),
            )
            async with pool:
                await wait_for_pool_connections(pool, expected=2)
                payloads = [b"a", b"b", b"c", b"d", b"e"]
                for payload in payloads:
                    await basic_server.send_to_all_clients(payload)
                    msg = await asyncio.wait_for(pool.__anext__(), timeout=1.0)
                    assert msg == payload

                await basic_server.send_to_all_clients(b"a")
                msg = await asyncio.wait_for(pool.__anext__(), timeout=1.0)
                assert msg == b"a"

    async def test_empty_payload_hash_deduplication(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Given an empty payload broadcast to 3 connections, When the pool iterates, Then only one empty bytes object is yielded.

                Empty payloads are valid websocket frames; they must participate
        in deduplication like any other message."""
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
                await basic_server.send_to_all_clients(b"")
                first = await asyncio.wait_for(pool.__anext__(), timeout=1.0)
                assert first == b""
                with pytest.raises(asyncio.TimeoutError):
                    await asyncio.wait_for(pool.__anext__(), timeout=0.3)
