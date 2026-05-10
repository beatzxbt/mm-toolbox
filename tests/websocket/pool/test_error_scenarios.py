"""Error scenario tests for WsPool.

Layer-2 tests validating that WsPool degrades gracefully under failures:
connection rejection, mid-stream disconnects, protocol errors, closed-pool
sends, zero-connection sends, and pending replacement tasks during close.
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
class TestWsPoolErrorScenarios:
    """Layer-2 tests for error handling paths in WsPool."""

    async def test_pool_connection_failures(
        self,
        server_reject_connections,
        connection_config_factory,
    ) -> None:
        """Given a rejecting server, When a pool starts, Then it tolerates failures and reports zero connections without crashing."""
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
        """Given a pool connected to a close-frame server, When a message triggers disconnect, Then the pool remains CONNECTED because other connections stay alive.

        This validates that a single failed connection does not bring
down the entire pool."""
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
        """Given a server that sends malformed frames, When the pool receives them, Then it survives and stays CONNECTED.

        Protocol errors must be isolated to the offending connection
rather than tearing down the whole pool."""
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
        """Given a closed pool, When send_data is called, Then RuntimeError is raised."""
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

    async def test_send_data_raises_when_zero_connections(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Given a pool where all connections were manually closed, When send_data is called, Then RuntimeError with 'No live connections' is raised.

        This prevents silent no-ops when the caller expects a message to
actually be transmitted."""
        async with basic_server:
            config = connection_config_factory(basic_server)
            pool_config = WsPoolConfig(num_connections=2, evict_interval_s=60)
            pool = await WsPool.new(
                config, on_message=noop_message_handler, pool_config=pool_config
            )
            async with pool:
                await asyncio.sleep(0.3)
                for conn in list(pool._conns.values()):
                    conn.close()
                pool._conns.clear()
                pool._fast_conn = None
                with pytest.raises(RuntimeError, match="No live connections"):
                    pool.send_data(b"fail", only_fastest=False)

    async def test_close_while_open_tasks_in_flight(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Given pending replacement tasks, When close() is called, Then it is safe and the pool ends in DISCONNECTED.

        Closing while background coroutines are reconnecting is a common
teardown race; this test ensures no unhandled exception propagates."""
        async with basic_server:
            config = connection_config_factory(basic_server)
            pool_config = WsPoolConfig(num_connections=2, evict_interval_s=60)
            pool = await WsPool.new(
                config, on_message=noop_message_handler, pool_config=pool_config
            )
            async with pool:
                await asyncio.sleep(0.3)
                pool._schedule_replacements(1)
                await asyncio.sleep(0.05)
                pool.close()
                assert pool.get_state() == ConnectionState.DISCONNECTED
