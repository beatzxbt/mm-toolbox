"""Latency task and timed operation tests for WsConnection."""

from __future__ import annotations

import asyncio
import time

import pytest


@pytest.mark.asyncio
class TestWsConnectionThreading:
    """Validate background latency timing behavior for WsConnection."""

    async def test_ema_latency_tracking(
        self,
        basic_server,
        connection_factory,
        latency_waiter,
    ) -> None:
        """Ensure ping/pong updates EMA latency metrics.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_factory: Fixture providing connected WsConnection factory.
            latency_waiter: Fixture providing latency wait helper.

        Returns:
            None: This test does not return a value.
        """
        async with basic_server:
            conn = await connection_factory(basic_server)
            await latency_waiter(conn, timeout_s=3.0)
            assert conn.get_state().latency_ms < 1000.0
            conn.close()

    async def test_latency_task_lifecycle(
        self,
        basic_server,
        connection_factory,
    ) -> None:
        """Ensure latency task starts and stops cleanly.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_factory: Fixture providing connected WsConnection factory.

        Returns:
            None: This test does not return a value.
        """
        async with basic_server:
            conn = await connection_factory(basic_server)
            task = getattr(conn, "_latency_task", None)
            assert task is not None
            assert task.done() is False
            conn.close()
            await asyncio.sleep(0.1)
            assert task.done() is True

    async def test_ping_interval_timing(
        self,
        basic_server,
        connection_factory,
        latency_waiter,
    ) -> None:
        """Ensure ping timing updates latency within a reasonable window.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_factory: Fixture providing connected WsConnection factory.
            latency_waiter: Fixture providing latency wait helper.

        Returns:
            None: This test does not return a value.
        """
        async with basic_server:
            conn = await connection_factory(basic_server)
            start = time.monotonic()
            await latency_waiter(conn, timeout_s=2.0)
            elapsed = time.monotonic() - start
            assert elapsed < 2.0
            conn.close()

    async def test_concurrent_send_and_ping(
        self,
        basic_server,
        connection_factory,
        latency_waiter,
    ) -> None:
        """Ensure sending data does not interfere with ping tracking.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_factory: Fixture providing connected WsConnection factory.
            latency_waiter: Fixture providing latency wait helper.

        Returns:
            None: This test does not return a value.
        """
        async with basic_server:
            conn = await connection_factory(basic_server)
            conn.send_data(b"payload")
            await latency_waiter(conn, timeout_s=2.0)
            await asyncio.sleep(0.1)
            assert b"payload" in basic_server.get_received_messages()
            conn.close()
