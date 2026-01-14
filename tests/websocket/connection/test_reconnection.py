"""Reconnect behavior tests for WsConnection."""

from __future__ import annotations

import asyncio
import time

import pytest

from mm_toolbox.ringbuffer.bytes import BytesRingBuffer
from mm_toolbox.websocket.connection import ConnectionState, WsConnection


@pytest.mark.asyncio
class TestWsConnectionReconnection:
    """Validate reconnect generator behavior."""

    async def test_reconnect_backoff_delays(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Ensure reconnect generator enforces a delay between attempts.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_config_factory: Fixture providing config factory.

        Returns:
            None: This test does not return a value.
        """
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
        """Ensure reconnection yields a usable connection after a close.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_config_factory: Fixture providing config factory.

        Returns:
            None: This test does not return a value.
        """
        async with basic_server:
            ringbuffer = BytesRingBuffer(max_capacity=8, only_insert_unique=False)
            config = connection_config_factory(basic_server)
            conn_iter = WsConnection.new_with_reconnect(ringbuffer, config)
            iterator = conn_iter.__aiter__()

            conn1 = await iterator.__anext__()
            conn1.close()
            conn2 = await iterator.__anext__()
            assert conn2.get_state().state == ConnectionState.CONNECTED
            conn2.close()
            await iterator.aclose()

    async def test_reconnect_respects_max_backoff(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Ensure reconnect delay does not grow without bound.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_config_factory: Fixture providing config factory.

        Returns:
            None: This test does not return a value.
        """
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
        """Ensure generator handles immediate connection closure.

        Args:
            server_reject_connections: Fixture providing rejecting server.
            connection_config_factory: Fixture providing config factory.

        Returns:
            None: This test does not return a value.
        """
        async with server_reject_connections:
            ringbuffer = BytesRingBuffer(max_capacity=8, only_insert_unique=False)
            config = connection_config_factory(server_reject_connections)
            conn_iter = WsConnection.new_with_reconnect(ringbuffer, config)
            iterator = conn_iter.__aiter__()

            conn = await iterator.__anext__()
            await asyncio.sleep(0.1)
            assert conn.get_state().state == ConnectionState.DISCONNECTED
            conn.close()
            await iterator.aclose()
