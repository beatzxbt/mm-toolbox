"""Send operations tests for WsConnection.

Layer-2 tests validating send_data, send_data_bytearray, send_ping,
and send_pong against a real WebSocket transport.

Key coverage:
- Active-transport delivery for bytes and bytearray payloads.
- No-op safety when disconnected.
- Concurrent send_data from multiple coroutines does not corrupt frames.
- PING/PONG exchange keeps the connection alive.
- Payload size variation (0, 1, 128, 1024, 4096 bytes) for bytearray path.
"""

from __future__ import annotations

import asyncio

import pytest

from mm_toolbox.websocket.connection import ConnectionState


@pytest.mark.asyncio
class TestWsConnectionSendOperations:
    """Layer-2 tests for WsConnection send behaviors."""

    async def test_send_data_with_active_transport(
        self,
        basic_server,
        connection_factory,
    ) -> None:
        """Given a connected transport, When send_data is called, Then the payload is echoed by the server."""
        async with basic_server:
            conn = await connection_factory(basic_server)
            payload = b"test message"
            conn.send_data(payload)
            await asyncio.sleep(0.1)
            assert payload in basic_server.get_received_messages()
            conn.close()

    async def test_send_data_bytearray_with_active_transport(
        self,
        basic_server,
        connection_factory,
    ) -> None:
        """Given a connected transport, When send_data_bytearray is called, Then the payload is echoed by the server."""
        async with basic_server:
            conn = await connection_factory(basic_server)
            payload = bytearray(b"bytearray message")
            conn.send_data_bytearray(payload)
            await asyncio.sleep(0.1)
            assert bytes(payload) in basic_server.get_received_messages()
            conn.close()

    async def test_send_data_without_transport(
        self,
        basic_server,
        connection_factory,
    ) -> None:
        """Given a disconnected connection, When send_data is called, Then it is a no-op and the server receives nothing.

        This prevents crashes when application code sends during teardown
        or after an unexpected disconnect."""
        async with basic_server:
            conn = await connection_factory(basic_server)
            conn.close()
            before = len(basic_server.get_received_messages())
            conn.send_data(b"offline")
            await asyncio.sleep(0.05)
            after = len(basic_server.get_received_messages())
            assert after == before

    async def test_send_ping_with_active_transport(
        self,
        basic_server,
        connection_factory,
    ) -> None:
        """Given a connected transport, When send_ping is called, Then the connection remains CONNECTED."""
        async with basic_server:
            conn = await connection_factory(basic_server)
            conn.send_ping()
            await asyncio.sleep(0.2)
            assert conn.get_state() == ConnectionState.CONNECTED
            conn.close()

    async def test_send_pong_response(
        self,
        basic_server,
        connection_factory,
    ) -> None:
        """Given a server PING, When the client auto-responds with PONG, Then the server receives it before timeout."""
        async with basic_server:
            conn = await connection_factory(basic_server)
            await asyncio.sleep(0.1)
            client = next(iter(basic_server._clients))
            pong_waiter = await client.ping()
            await asyncio.wait_for(pong_waiter, timeout=1.0)
            conn.close()

    async def test_send_data_concurrent_multiple_frames(
        self,
        basic_server,
        connection_factory,
        chaos_runner,
    ) -> None:
        """Given multiple coroutines calling send_data concurrently, When they complete, Then all payloads are received by the server.

        Concurrent sends are common in high-throughput paths; this test
        verifies that internal frame queuing does not drop or corrupt data."""
        async with basic_server:
            conn = await connection_factory(basic_server)
            payloads = [f"msg-{idx}".encode("utf-8") for idx in range(5)]

            async def _send(payload: bytes) -> None:
                """Send a payload through the connection."""
                conn.send_data(payload)
                await asyncio.sleep(0)

            def _make_sender(payload: bytes):
                """Create a sender coroutine for a payload."""

                async def _sender() -> None:
                    """Send the bound payload."""
                    await _send(payload)

                return _sender

            await chaos_runner([_make_sender(p) for p in payloads], seed=42)
            await asyncio.sleep(0.1)
            received = basic_server.get_received_messages()
            assert set(payloads).issubset(set(received))
            conn.close()

    async def test_send_pong_with_payload(
        self,
        basic_server,
        connection_factory,
    ) -> None:
        """Given a custom PONG payload, When send_pong is called, Then the connection stays CONNECTED without crashing."""
        async with basic_server:
            conn = await connection_factory(basic_server)
            conn.send_pong(b"custom")
            await asyncio.sleep(0.1)
            assert conn.get_state() == ConnectionState.CONNECTED
            conn.close()

    async def test_send_data_bytearray_safe_allocation(
        self,
        basic_server,
        connection_factory,
    ) -> None:
        """Given bytearray payloads of varying sizes (0 to 4096 bytes), When sent, Then all are received by the server.

        This exercises internal buffer allocation paths that may behave
differently for empty, small, and medium-large payloads."""
        async with basic_server:
            conn = await connection_factory(basic_server)
            for size in [0, 1, 128, 1024, 4096]:
                payload = bytearray(b"x" * size)
                conn.send_data_bytearray(payload)
            await asyncio.sleep(0.2)
            received = basic_server.get_received_messages()
            assert len(received) >= 5
            conn.close()
