"""Async iteration tests for WsPool.

Layer-2 tests validating that WsPool correctly exposes an async iterator
over deduplicated messages from its underlying connections.

Key coverage:
- Basic async iteration yields messages from the pool ringbuffer.
- Arrival validation (all expected payloads arrive, ordering is relaxed).
- Disconnect handling: iteration can be cancelled after pool connections drop.
- StopAsyncIteration raised after pool.close().
- Timeout behavior: __anext__ returns promptly once the pool is closed.
"""

from __future__ import annotations

import asyncio

import pytest

from mm_toolbox.websocket.pool import WsPool, WsPoolConfig


def noop_message_handler(msg: bytes) -> None:
    """No-op message handler for pool tests."""
    return None


@pytest.mark.asyncio
class TestWsPoolAsyncIteration:
    """Layer-2 tests for async iteration over the pool ringbuffer."""

    async def test_async_iteration_over_pool_ringbuffer(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Given a pool with 2 connections, When the server broadcasts 3 messages, Then async iteration yields all 3."""
        async with basic_server:
            config = connection_config_factory(basic_server)
            pool_config = WsPoolConfig(num_connections=2, evict_interval_s=60)
            pool = await WsPool.new(
                config, on_message=noop_message_handler, pool_config=pool_config
            )
            expected = {b"p1", b"p2", b"p3"}
            seen: set[bytes] = set()

            async with pool:

                async def _collector() -> None:
                    async for msg in pool:
                        if msg in expected:
                            seen.add(msg)
                        if expected.issubset(seen):
                            break

                collect_task = asyncio.create_task(_collector())
                await asyncio.sleep(0.2)
                await basic_server.send_to_all_clients(b"p1")
                await basic_server.send_to_all_clients(b"p2")
                await basic_server.send_to_all_clients(b"p3")
                await asyncio.wait_for(collect_task, timeout=3.0)

            assert seen == expected

    async def test_async_iteration_arrival_validation(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Given a pool with 2 connections, When multiple messages are broadcast, Then all arrive even if ordering varies.

                Cross-connection scheduling means order is not guaranteed; the
        important invariant is completeness, not sequence."""
        async with basic_server:
            config = connection_config_factory(basic_server)
            pool_config = WsPoolConfig(num_connections=2, evict_interval_s=60)
            pool = await WsPool.new(
                config, on_message=noop_message_handler, pool_config=pool_config
            )
            payloads = [b"a", b"b", b"c"]
            expected = set(payloads)
            seen: set[bytes] = set()

            async with pool:

                async def _collector() -> None:
                    async for msg in pool:
                        if msg in expected:
                            seen.add(msg)
                        if expected.issubset(seen):
                            break

                collect_task = asyncio.create_task(_collector())
                await asyncio.sleep(0.2)
                for payload in payloads:
                    await basic_server.send_to_all_clients(payload)
                await asyncio.wait_for(collect_task, timeout=3.0)

            assert seen == expected

    async def test_async_iteration_stops_on_disconnect(
        self,
        server_send_close_frame,
        connection_config_factory,
    ) -> None:
        """Given a pool connected to a close-frame server, When disconnects occur, Then the pending __anext__ either returns the last message or can be cancelled cleanly."""
        async with server_send_close_frame:
            config = connection_config_factory(server_send_close_frame)
            pool_config = WsPoolConfig(num_connections=2, evict_interval_s=60)
            pool = await WsPool.new(
                config, on_message=noop_message_handler, pool_config=pool_config
            )

            async with pool:
                collect_task = asyncio.create_task(pool.__anext__())
                await asyncio.sleep(0.2)
                pool.send_data(b"trigger-close", only_fastest=False)
                await asyncio.sleep(0.2)
                if collect_task.done():
                    assert collect_task.result() == b"trigger-close"
                else:
                    collect_task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await collect_task

    async def test_anext_stop_async_iteration(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Given a closed pool, When __anext__ is called, Then StopAsyncIteration is raised immediately."""
        async with basic_server:
            config = connection_config_factory(basic_server)
            pool_config = WsPoolConfig(num_connections=2, evict_interval_s=60)
            pool = await WsPool.new(
                config, on_message=noop_message_handler, pool_config=pool_config
            )
            async with pool:
                await asyncio.sleep(0.2)
                pool.close()
                with pytest.raises(StopAsyncIteration):
                    await pool.__anext__()

    async def test_anext_timeout_behavior(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Given a closed pool, When __anext__ is called, Then it returns within 1.5 seconds.

                This prevents the iterator from hanging indefinitely when the
        underlying connections have all been torn down."""
        async with basic_server:
            config = connection_config_factory(basic_server)
            pool_config = WsPoolConfig(num_connections=2, evict_interval_s=60)
            pool = await WsPool.new(
                config, on_message=noop_message_handler, pool_config=pool_config
            )
            async with pool:
                await asyncio.sleep(0.2)
                start = asyncio.get_running_loop().time()
                pool.close()
                with pytest.raises(StopAsyncIteration):
                    await pool.__anext__()
                elapsed = asyncio.get_running_loop().time() - start
                assert elapsed < 1.5
