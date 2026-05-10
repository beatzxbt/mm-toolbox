"""End-to-end workflow tests for websocket components.

Layer-3 tests exercising realistic connect/send/receive/error workflows
through the WsSingle high-level wrapper and the WsConnection reconnect
iterator.

Key coverage:
- Full connect->send->receive->close workflow via WsSingle.
- Reconnection after explicit close using the async generator.
- Recovery after protocol errors (invalid frames) by switching to a
  healthy server mid-iteration.
"""

from __future__ import annotations

import asyncio

import pytest

from mm_toolbox.ringbuffer.bytes import BytesRingBuffer
from mm_toolbox.websocket.connection import ConnectionState, WsConnection
from mm_toolbox.websocket.single import WsSingle


@pytest.mark.asyncio
class TestWebSocketWorkflows:
    """Layer-3 tests for realistic connect/send/receive/error workflows."""

    async def test_connect_send_receive_disconnect(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Given a local echo server, When WsSingle is used to connect, send, and iterate, Then the expected message is received and the connection closes cleanly."""
        async with basic_server:
            config = connection_config_factory(basic_server)
            collected: list[bytes] = []
            target = b"welcome"

            async with WsSingle(config) as ws:

                async def _collector() -> None:
                    async for msg in ws:
                        if msg == target:
                            collected.append(msg)
                            break

                task = asyncio.create_task(_collector())
                await asyncio.sleep(0.2)
                ws.send_data(b"hello")
                await basic_server.send_to_all_clients(b"welcome")
                await asyncio.wait_for(task, timeout=2.0)
            assert collected == [target]
            assert b"hello" in basic_server.get_received_messages()

    async def test_reconnect_after_disconnect(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Given a closed WsConnection, When the reconnect iterator is resumed, Then a new CONNECTED instance is yielded and messages flow again."""
        async with basic_server:
            ringbuffer = BytesRingBuffer(max_capacity=16, only_insert_unique=False)
            config = connection_config_factory(basic_server)
            conn_iter = WsConnection.new_with_reconnect(ringbuffer, config)
            iterator = conn_iter.__aiter__()

            conn1 = await iterator.__anext__()
            assert conn1.get_state() == ConnectionState.CONNECTED
            conn1.close()

            conn2 = await iterator.__anext__()
            assert conn2.get_state() == ConnectionState.CONNECTED
            conn2.send_data(b"reconnected")
            await asyncio.sleep(0.2)
            assert b"reconnected" in basic_server.get_received_messages()
            conn2.close()
            await iterator.aclose()

    async def test_error_then_recover_with_reconnect(
        self,
        server_send_invalid_frames,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Given a server that sends invalid frames, When the reconnect iterator switches to a healthy server, Then recovery succeeds.

        This simulates a real-world scenario where one endpoint is
misbehaving and the client must transparently resume on a good one."""
        async with server_send_invalid_frames:
            ringbuffer = BytesRingBuffer(max_capacity=16, only_insert_unique=False)
            config = connection_config_factory(server_send_invalid_frames)
            conn_iter = WsConnection.new_with_reconnect(ringbuffer, config)
            iterator = conn_iter.__aiter__()

            conn1 = await iterator.__anext__()
            conn1.send_data(b"trigger-invalid")
            await asyncio.sleep(0.2)
            conn1.close()

            async with basic_server:
                config.wss_url = basic_server.uri
                conn2 = await iterator.__anext__()
                assert conn2.get_state() == ConnectionState.CONNECTED
                conn2.close()
            await iterator.aclose()
