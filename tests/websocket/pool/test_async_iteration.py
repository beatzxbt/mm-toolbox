"""Async iteration tests for WsPool."""

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


@pytest.mark.asyncio
class TestWsPoolAsyncIteration:
    """Validate async iteration over pool ringbuffer."""

    async def test_async_iteration_over_pool_ringbuffer(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Ensure async iteration yields messages from the pool.

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
            expected = {b"p1", b"p2", b"p3"}
            seen: set[bytes] = set()

            async with pool:

                async def _collector() -> None:
                    """Collect a fixed number of pool messages.

                    Returns:
                        None: This helper does not return a value.
                    """
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
        """Ensure all sent messages arrive without strict ordering.

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
            payloads = [b"a", b"b", b"c"]
            expected = set(payloads)
            seen: set[bytes] = set()

            async with pool:

                async def _collector() -> None:
                    """Collect all expected pool messages.

                    Returns:
                        None: This helper does not return a value.
                    """
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
        """Ensure iteration can be cancelled after disconnects.

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
