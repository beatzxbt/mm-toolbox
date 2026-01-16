"""Callback-driven tests for WsConnection using real WebSocket frames.

Covers on_connect payloads, frame handling with fragmentation/compression,
buffer accumulation, and disconnect handling.
"""

from __future__ import annotations

import asyncio

import pytest

from mm_toolbox.websocket.connection import ConnectionState


@pytest.mark.asyncio
class TestWsConnectionCallbacks:
    """Validate WsConnection callback-driven behaviors."""

    async def test_on_connected_with_real_frame(
        self,
        basic_server,
        connection_factory,
    ) -> None:
        """Verify on_connect payloads are sent after connection.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_factory: Fixture providing connected WsConnection factory.

        Returns:
            None: This test does not return a value.
        """
        async with basic_server:
            payload = b'{"type":"hello"}'
            conn = await connection_factory(basic_server, on_connect=[payload])
            await asyncio.sleep(0.1)
            assert payload in basic_server.get_received_messages()
            conn.close()

    async def test_on_frame_with_single_message(
        self,
        basic_server,
        connection_factory,
    ) -> None:
        """Ensure a single server message reaches the ringbuffer.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_factory: Fixture providing connected WsConnection factory.

        Returns:
            None: This test does not return a value.
        """
        async with basic_server:
            conn = await connection_factory(basic_server)
            await basic_server.send_to_all_clients(b"hello")
            ringbuffer = conn.get_state().ringbuffer
            msg = await asyncio.wait_for(ringbuffer.aconsume(), timeout=1.0)
            assert msg == b"hello"
            conn.close()

    async def test_on_frame_with_fragmented_message(
        self,
        server_with_fragmentation,
        connection_factory,
    ) -> None:
        """Verify fragmented frames are reassembled into one message.

        Args:
            server_with_fragmentation: Fixture providing fragmented server.
            connection_factory: Fixture providing connected WsConnection factory.

        Returns:
            None: This test does not return a value.
        """
        async with server_with_fragmentation:
            conn = await connection_factory(server_with_fragmentation)
            payload = b"fragmented-message"
            await server_with_fragmentation.send_to_all_clients(payload)
            ringbuffer = conn.get_state().ringbuffer
            msg = await asyncio.wait_for(ringbuffer.aconsume(), timeout=1.0)
            assert msg == payload
            conn.close()

    async def test_on_frame_with_compressed_message(
        self,
        server_with_compression,
        connection_factory,
    ) -> None:
        """Verify compressed frames decode into the original message.

        Args:
            server_with_compression: Fixture providing compressed server.
            connection_factory: Fixture providing connected WsConnection factory.

        Returns:
            None: This test does not return a value.
        """
        async with server_with_compression:
            conn = await connection_factory(server_with_compression)
            payload = b"compressed-message"
            await server_with_compression.send_to_all_clients(payload)
            ringbuffer = conn.get_state().ringbuffer
            msg = await asyncio.wait_for(ringbuffer.aconsume(), timeout=1.0)
            assert msg == payload
            conn.close()

    async def test_on_frame_buffer_accumulation(
        self,
        basic_server,
        connection_factory,
    ) -> None:
        """Ensure multiple frames accumulate in the ringbuffer.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_factory: Fixture providing connected WsConnection factory.

        Returns:
            None: This test does not return a value.
        """
        async with basic_server:
            conn = await connection_factory(basic_server)
            payloads = [b"one", b"two", b"three"]
            for payload in payloads:
                await basic_server.send_to_all_clients(payload)
            ringbuffer = conn.get_state().ringbuffer
            received = [
                await asyncio.wait_for(ringbuffer.aconsume(), timeout=1.0)
                for _ in payloads
            ]
            assert set(received) == set(payloads)
            conn.close()

    async def test_on_frame_oversized_message(
        self,
        basic_server,
        connection_factory,
        oversized_payload_factory,
    ) -> None:
        """Verify oversized messages are rejected safely.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_factory: Fixture providing connected WsConnection factory.
            oversized_payload_factory: Fixture providing oversized payloads.

        Returns:
            None: This test does not return a value.
        """
        async with basic_server:
            conn = await connection_factory(basic_server)
            oversized = oversized_payload_factory()
            await basic_server.send_to_all_clients(oversized)
            await asyncio.sleep(0.1)
            ringbuffer = conn.get_state().ringbuffer
            assert ringbuffer.is_empty()
            conn.close()

    async def test_on_disconnected_callback(
        self,
        server_send_close_frame,
        connection_factory,
        state_waiter,
    ) -> None:
        """Verify disconnection updates connection state.

        Args:
            server_send_close_frame: Fixture providing close-frame server.
            connection_factory: Fixture providing connected WsConnection factory.
            state_waiter: Fixture providing state wait helper.

        Returns:
            None: This test does not return a value.
        """
        async with server_send_close_frame:
            conn = await connection_factory(server_send_close_frame)
            conn.send_data(b"close-me")
            await state_waiter(conn, ConnectionState.DISCONNECTED, timeout_s=2.0)
            conn.close()

    async def test_on_disconnected_mid_message(
        self,
        server_send_close_frame,
        connection_factory,
        state_waiter,
    ) -> None:
        """Verify disconnection during traffic leaves clean state.

        Args:
            server_send_close_frame: Fixture providing close-frame server.
            connection_factory: Fixture providing connected WsConnection factory.
            state_waiter: Fixture providing state wait helper.

        Returns:
            None: This test does not return a value.
        """
        async with server_send_close_frame:
            conn = await connection_factory(server_send_close_frame)
            conn.send_data(b"partial")
            await state_waiter(conn, ConnectionState.DISCONNECTED, timeout_s=2.0)
            conn.close()
