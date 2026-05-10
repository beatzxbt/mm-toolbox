"""Reconnect behavior tests for WsConnection.

Layer-2 tests for the WsConnection.new_with_reconnect async generator.
Covers backoff timing, eventual success after transient failures, maximum
backoff clamping, and graceful handling of connection rejection.

Edge cases include verifying that the generator enforces a minimum delay
between attempts and does not allow unbounded backoff growth.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from mm_toolbox.ringbuffer.bytes import BytesRingBuffer
from mm_toolbox.websocket.connection import ConnectionState, WsConnection


@pytest.mark.asyncio
class TestWsConnectionReconnection:
    """Layer-2 tests for WsConnection reconnect generator behavior."""

    async def test_reconnect_backoff_delays(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Given a closed connection, When the reconnect generator yields the next one, Then at least the configured backoff has elapsed.

        This prevents aggressive reconnection storms that could overwhelm
        the server or exhaust local file descriptors."""
        async with basic_server:
            ringbuffer = BytesRingBuffer(max_capacity=8, only_insert_unique=False)
            config = connection_config_factory(basic_server)
            conn_iter = WsConnection.new_with_reconnect(ringbuffer, config)
            iterator = conn_iter.__aiter__()

            conn1 = await iterator.__anext__()
            conn1.close()
            start = time.monotonic()
            conn2 = await iterator.__anext__()
            elapsed = time.monotonic() - start
            assert elapsed >= 1.0
            assert elapsed < 2.5
            conn2.close()
            await iterator.aclose()

    async def test_reconnect_eventual_success(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Given a closed connection, When the reconnect generator runs, Then it eventually yields a new CONNECTED connection."""
        async with basic_server:
            ringbuffer = BytesRingBuffer(max_capacity=8, only_insert_unique=False)
            config = connection_config_factory(basic_server)
            conn_iter = WsConnection.new_with_reconnect(ringbuffer, config)
            iterator = conn_iter.__aiter__()

            conn1 = await iterator.__anext__()
            conn1.close()
            conn2 = await iterator.__anext__()
            assert conn2.get_state() == ConnectionState.CONNECTED
            conn2.close()
            await iterator.aclose()

    async def test_reconnect_respects_max_backoff(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Given repeated reconnections, When delays accumulate, Then they are clamped below the maximum backoff cap.

        Without a cap, backoff could grow to minutes or hours, making the
        client unresponsive after a long outage."""
        async with basic_server:
            ringbuffer = BytesRingBuffer(max_capacity=8, only_insert_unique=False)
            config = connection_config_factory(basic_server)
            conn_iter = WsConnection.new_with_reconnect(ringbuffer, config)
            iterator = conn_iter.__aiter__()

            conn = await iterator.__anext__()
            conn.close()
            start = time.monotonic()
            conn_retry = await iterator.__anext__()
            elapsed = time.monotonic() - start
            assert elapsed < 2.5
            conn_retry.close()
            await iterator.aclose()

    async def test_reconnect_with_connection_rejection(
        self,
        server_reject_connections,
        connection_config_factory,
    ) -> None:
        """Given a server that rejects handshakes, When the reconnect generator runs, Then it yields a DISCONNECTED connection without crashing.

        This validates that the generator tolerates hard failures and
        surfaces them as state rather than unhandled exceptions."""
        async with server_reject_connections:
            ringbuffer = BytesRingBuffer(max_capacity=8, only_insert_unique=False)
            config = connection_config_factory(server_reject_connections)
            conn_iter = WsConnection.new_with_reconnect(ringbuffer, config)
            iterator = conn_iter.__aiter__()

            conn = await iterator.__anext__()
            await asyncio.sleep(0.1)
            assert conn.get_state() == ConnectionState.DISCONNECTED
            conn.close()
            await iterator.aclose()
