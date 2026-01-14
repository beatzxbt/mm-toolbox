"""Send operations tests for WsConnection.

Validates send_data, send_ping, and ping/pong response handling with
real websocket transport.
"""

from __future__ import annotations

import asyncio

import pytest

from mm_toolbox.websocket.connection import ConnectionState


@pytest.mark.asyncio
class TestWsConnectionSendOperations:
    """Validate WsConnection send behaviors."""

    async def test_send_data_with_active_transport(
        self,
        basic_server,
        connection_factory,
    ) -> None:
        """Ensure send_data delivers payloads to the server.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_factory: Fixture providing connected WsConnection factory.

        Returns:
            None: This test does not return a value.
        """
        async with basic_server:
            conn = await connection_factory(basic_server)
            payload = b"test message"
            conn.send_data(payload)
            await asyncio.sleep(0.1)
            assert payload in basic_server.get_received_messages()
            conn.close()

    async def test_send_data_without_transport(
        self,
        basic_server,
        connection_factory,
    ) -> None:
        """Ensure send_data is a no-op when disconnected.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_factory: Fixture providing connected WsConnection factory.

        Returns:
            None: This test does not return a value.
        """
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
        """Ensure send_ping does not disrupt the connection.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_factory: Fixture providing connected WsConnection factory.
        Returns:
            None: This test does not return a value.
        """
        async with basic_server:
            conn = await connection_factory(basic_server)
            conn.send_ping()
            await asyncio.sleep(0.2)
            assert conn.get_state().state == ConnectionState.CONNECTED
            conn.close()

    async def test_send_pong_response(
        self,
        basic_server,
        connection_factory,
    ) -> None:
        """Ensure WsConnection responds to server pings with pongs.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_factory: Fixture providing connected WsConnection factory.

        Returns:
            None: This test does not return a value.
        """
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
        """Ensure concurrent send_data calls do not interfere.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_factory: Fixture providing connected WsConnection factory.
            chaos_runner: Fixture providing chaotic concurrency helper.

        Returns:
            None: This test does not return a value.
        """
        async with basic_server:
            conn = await connection_factory(basic_server)
            payloads = [f"msg-{idx}".encode("utf-8") for idx in range(5)]

            async def _send(payload: bytes) -> None:
                """Send a payload through the connection.

                Args:
                    payload (bytes): Payload to send.

                Returns:
                    None: This helper does not return a value.
                """
                conn.send_data(payload)
                await asyncio.sleep(0)

            def _make_sender(payload: bytes):
                """Create a sender coroutine for a payload.

                Args:
                    payload (bytes): Payload to send.

                Returns:
                    Callable[[], Awaitable[None]]: Sender coroutine factory.
                """

                async def _sender() -> None:
                    """Send the bound payload.

                    Returns:
                        None: This helper does not return a value.
                    """
                    await _send(payload)

                return _sender

            await chaos_runner([_make_sender(p) for p in payloads], seed=42)
            await asyncio.sleep(0.1)
            received = basic_server.get_received_messages()
            assert set(payloads).issubset(set(received))
            conn.close()
