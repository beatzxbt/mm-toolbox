"""Callback-driven tests for WsConnection using real WebSocket frames.

Covers on_connect payloads, frame handling with fragmentation/compression,
buffer accumulation, and disconnect handling.
"""

from __future__ import annotations

import asyncio

import pytest

from mm_toolbox.ringbuffer.bytes import BytesRingBuffer
from mm_toolbox.websocket.connection import ConnectionState, WsConnection
from tests.websocket.conftest import wait_for_connection_state


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
            ringbuffer = conn.get_ringbuffer()
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
            ringbuffer = conn.get_ringbuffer()
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
            ringbuffer = conn.get_ringbuffer()
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
            ringbuffer = conn.get_ringbuffer()
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
            ringbuffer = conn.get_ringbuffer()
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

    async def test_close_frame_triggers_disconnect(
        self,
        server_send_close_frame,
        connection_factory,
        state_waiter,
    ) -> None:
        """Verify server CLOSE frame transitions state and cleans up.

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
            assert conn._latency_task is None or conn._latency_task.done()
            conn.close()

    async def test_should_stop_blocks_frame_processing(
        self,
        basic_server,
        connection_factory,
    ) -> None:
        """Verify closing prevents further frame processing.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_factory: Fixture providing connected WsConnection factory.

        Returns:
            None: This test does not return a value.
        """
        async with basic_server:
            conn = await connection_factory(basic_server)
            ringbuffer = conn.get_ringbuffer()
            conn.close()
            await asyncio.sleep(0.1)
            before = len(ringbuffer)
            await basic_server.send_to_all_clients(b"after-close")
            await asyncio.sleep(0.2)
            after = len(ringbuffer)
            assert after == before

    async def test_mid_fragment_disconnect_clears_buffer(
        self,
        basic_server,
        connection_factory,
    ) -> None:
        """Verify incomplete fragments are cleared on disconnect.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_factory: Fixture providing connected WsConnection factory.

        Returns:
            None: This test does not return a value.
        """
        pytest.skip("_unfin_msg_buffer is a cdef field inaccessible from Python")

    async def test_empty_payload_fast_path(
        self,
        basic_server,
        connection_factory,
    ) -> None:
        """Verify empty TEXT frame reaches the ringbuffer.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_factory: Fixture providing connected WsConnection factory.

        Returns:
            None: This test does not return a value.
        """
        async with basic_server:
            conn = await connection_factory(basic_server)
            await basic_server.send_to_all_clients(b"")
            ringbuffer = conn.get_ringbuffer()
            msg = await asyncio.wait_for(ringbuffer.aconsume(), timeout=1.0)
            assert msg == b""
            conn.close()

    async def test_exact_max_frame_size_boundary(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Verify frame size boundary at max_frame_size.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_config_factory: Fixture providing config factory.

        Returns:
            None: This test does not return a value.
        """
        async with basic_server:
            config = connection_config_factory(basic_server)
            config.max_frame_size = 1024
            ringbuffer = BytesRingBuffer(max_capacity=8, only_insert_unique=False)
            conn = await WsConnection.new(ringbuffer, config)
            await wait_for_connection_state(
                conn, ConnectionState.CONNECTED, timeout_s=2.0
            )

            await basic_server.send_to_all_clients(b"x" * 1023)
            msg = await asyncio.wait_for(ringbuffer.aconsume(), timeout=1.0)
            assert msg == b"x" * 1023

            await basic_server.send_to_all_clients(b"x" * 1024)
            msg = await asyncio.wait_for(ringbuffer.aconsume(), timeout=1.0)
            assert msg == b"x" * 1024

            await basic_server.send_to_all_clients(b"x" * 1025)
            await asyncio.sleep(0.2)
            assert ringbuffer.is_empty()
            conn.close()

    async def test_ping_with_payload(
        self,
        basic_server,
        connection_factory,
    ) -> None:
        """Verify PING with payload triggers PONG with matching payload.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_factory: Fixture providing connected WsConnection factory.

        Returns:
            None: This test does not return a value.
        """
        async with basic_server:
            conn = await connection_factory(basic_server)
            client = next(iter(basic_server._clients))
            pong_waiter = await client.ping(b"ping-payload")
            await asyncio.wait_for(pong_waiter, timeout=1.0)
            conn.close()
