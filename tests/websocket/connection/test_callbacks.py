"""Callback-driven frame-handling tests for WsConnection.

Layer-2 tests exercising the on_connect, on_frame, and on_disconnect
callback paths against a local WebSocket server.

Key coverage:
- on_connect payloads sent immediately after handshake.
- Single, fragmented, and compressed TEXT frames reassembled correctly.
- Ringbuffer accumulation and exact/max-frame-size boundary rejection.
- Oversized payload safety (must not crash or leak into buffer).
- Empty payload fast path.
- Graceful disconnect on server CLOSE frame and mid-message termination.
- should_stop guard preventing further frame processing after close.
"""

from __future__ import annotations

import asyncio

import pytest

from mm_toolbox.ringbuffer.bytes import BytesRingBuffer
from mm_toolbox.websocket.connection import ConnectionState, WsConnection
from tests.websocket.conftest import wait_for_connection_state


@pytest.mark.asyncio
class TestWsConnectionCallbacks:
    """Layer-2 tests for WsConnection callback-driven frame handling."""

    async def test_on_connected_with_real_frame(
        self,
        basic_server,
        connection_factory,
    ) -> None:
        """Given on_connect payloads configured, When connection succeeds, Then payloads are transmitted to the server."""
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
        """Given one TEXT frame from the server, When it arrives, Then it is placed intact into the ringbuffer."""
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
        """Given a server that fragments outbound frames, When a message arrives, Then it is reassembled into the original payload.

        Fragmentation is common with large messages or certain proxies; the
driver must buffer and reassemble before surfacing to the ringbuffer."""
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
        """Given a server with permessage-deflate enabled, When a message arrives, Then it is decompressed to the original payload."""
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
        """Given multiple sequential frames, When they arrive, Then all are stored in the ringbuffer without loss."""
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
        """Given a payload larger than max_frame_size, When it arrives, Then it is discarded and the ringbuffer stays empty.

        This protects downstream consumers from unbounded memory growth
when a peer sends unexpectedly large frames."""
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
        """Given a server that sends a CLOSE frame, When the client processes it, Then state transitions to DISCONNECTED."""
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
        """Given a server that closes mid-traffic, When the close arrives, Then the connection ends in a clean DISCONNECTED state."""
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
        """Given a server CLOSE frame, When handled, Then state transitions and background tasks are cleaned up."""
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
        """Given the connection is closed, When the server sends more frames, Then they are ignored by the client."""
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
        """Given an incomplete fragmented message, When disconnect occurs, Then the partial buffer is cleared.

        Skipped because the underlying buffer is a cdef field inaccessible
from Python."""
        pytest.skip("_unfin_msg_buffer is a cdef field inaccessible from Python")

    async def test_empty_payload_fast_path(
        self,
        basic_server,
        connection_factory,
    ) -> None:
        """Given an empty TEXT frame, When it arrives, Then it is stored as an empty bytes object in the ringbuffer."""
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
        """Given max_frame_size=1024, When payloads of 1023, 1024, and 1025 bytes arrive, Then only the 1025-byte payload is rejected.

        Boundary testing is critical because off-by-one errors in frame-size
logic can silently drop valid messages or admit oversized ones."""
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
        """Given a server PING with a custom payload, When the client responds, Then a matching PONG is received by the server."""
        async with basic_server:
            conn = await connection_factory(basic_server)
            client = next(iter(basic_server._clients))
            pong_waiter = await client.ping(b"ping-payload")
            await asyncio.wait_for(pong_waiter, timeout=1.0)
            conn.close()
