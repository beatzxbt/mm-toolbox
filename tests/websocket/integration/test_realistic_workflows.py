"""End-to-end workflow tests for websocket components."""

from __future__ import annotations

import asyncio

import pytest

from mm_toolbox.ringbuffer.bytes import BytesRingBuffer
from mm_toolbox.websocket.connection import ConnectionState, WsConnection
from mm_toolbox.websocket.single import WsSingle


@pytest.mark.asyncio
class TestWebSocketWorkflows:
    """Exercise realistic connect/send/receive/error workflows."""

    async def test_connect_send_receive_disconnect(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Validate a full connect->send->receive->close workflow.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_config_factory: Fixture providing config factory.

        Returns:
            None: This test does not return a value.
        """
        async with basic_server:
            config = connection_config_factory(basic_server)
            collected: list[bytes] = []
            target = b"welcome"

            async with WsSingle(config) as ws:

                async def _collector() -> None:
                    """Collect a single message from the workflow.

                    Returns:
                        None: This helper does not return a value.
                    """
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
        """Validate reconnection workflow after explicit close.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_config_factory: Fixture providing config factory.

        Returns:
            None: This test does not return a value.
        """
        async with basic_server:
            ringbuffer = BytesRingBuffer(max_capacity=16, only_insert_unique=False)
            config = connection_config_factory(basic_server)
            conn_iter = WsConnection.new_with_reconnect(ringbuffer, config)
            iterator = conn_iter.__aiter__()

            conn1 = await iterator.__anext__()
            assert conn1.get_state().state == ConnectionState.CONNECTED
            conn1.close()

            conn2 = await iterator.__anext__()
            assert conn2.get_state().state == ConnectionState.CONNECTED
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
        """Validate recovery after protocol errors using reconnect iterator.

        Args:
            server_send_invalid_frames: Fixture providing invalid frame server.
            basic_server: Fixture providing a basic echo server.
            connection_config_factory: Fixture providing config factory.

        Returns:
            None: This test does not return a value.
        """
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
                assert conn2.get_state().state == ConnectionState.CONNECTED
                conn2.close()
            await iterator.aclose()
