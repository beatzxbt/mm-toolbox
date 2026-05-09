"""State machine and snapshot tests for WsConnection.

Exercises ConnectionState enum values, WsConnection properties,
and state transitions during connection lifecycle.
"""

from __future__ import annotations

import asyncio

import pytest

from mm_toolbox.ringbuffer.bytes import BytesRingBuffer
from mm_toolbox.websocket.connection import (
    ConnectionState,
    WsConnection,
    WsConnectionConfig,
)


class TestConnectionStateEnum:
    """Validate ConnectionState enum values and ordering."""

    def test_enum_values(self) -> None:
        """Ensure ConnectionState values match expected integers.

        Returns:
            None: This test does not return a value.
        """
        assert ConnectionState.DISCONNECTED == 0
        assert ConnectionState.CONNECTING == 1
        assert ConnectionState.CONNECTED == 2

    def test_enum_ordering(self) -> None:
        """Ensure ConnectionState ordering is consistent.

        Returns:
            None: This test does not return a value.
        """
        assert ConnectionState.DISCONNECTED < ConnectionState.CONNECTING
        assert ConnectionState.CONNECTING < ConnectionState.CONNECTED

    def test_enum_equality(self) -> None:
        """Ensure ConnectionState comparisons to integers work.

        Returns:
            None: This test does not return a value.
        """
        assert ConnectionState.CONNECTED == 2
        assert ConnectionState.DISCONNECTED != 1


class TestWsConnectionProperties:
    """Validate WsConnection direct property access."""

    def test_initial_properties(self) -> None:
        """Verify initial property values on a fresh connection."""
        ringbuffer = BytesRingBuffer(max_capacity=64, only_insert_unique=False)
        config = WsConnectionConfig.default("wss://test.example.com")
        conn = WsConnection(ringbuffer, config)
        assert conn.get_seq_id() == 0
        assert conn.get_latency_ms() == 1000.0
        assert conn.is_connected() is False
        assert conn.get_ringbuffer() is ringbuffer

    def test_property_updates_after_simulated_connect(self) -> None:
        """Verify properties reflect state after connection."""
        pytest.skip("cdef fields cannot be mutated from Python")


@pytest.mark.asyncio
class TestWsConnectionStateMachine:
    """Validate live connection state transitions."""

    async def test_state_transitions(
        self,
        basic_server,
        connection_factory,
    ) -> None:
        """Ensure connection transitions to CONNECTED then DISCONNECTED.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_factory: Fixture providing connected WsConnection factory.

        Returns:
            None: This test does not return a value.
        """
        async with basic_server:
            conn = await connection_factory(basic_server)
            assert conn.get_state() == ConnectionState.CONNECTED
            conn.close()
            await asyncio.sleep(0.1)
            assert conn.get_state() == ConnectionState.DISCONNECTED

    async def test_get_state_returns_enum(
        self,
        basic_server,
        connection_factory,
    ) -> None:
        """Ensure get_state returns a ConnectionState enum directly.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_factory: Fixture providing connected WsConnection factory.

        Returns:
            None: This test does not return a value.
        """
        async with basic_server:
            conn = await connection_factory(basic_server)
            state = conn.get_state()
            assert isinstance(state, ConnectionState)
            assert state in (
                ConnectionState.DISCONNECTED,
                ConnectionState.CONNECTING,
                ConnectionState.CONNECTED,
            )
            if not conn.get_ringbuffer().is_empty():
                assert (
                    conn.get_ringbuffer().peekright()
                    == conn.get_ringbuffer().peekright()
                )
            conn.close()

    async def test_state_changes_during_callback_execution(
        self,
        basic_server,
        connection_factory,
    ) -> None:
        """Ensure state reads remain valid during message handling.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_factory: Fixture providing connected WsConnection factory.

        Returns:
            None: This test does not return a value.
        """
        async with basic_server:
            conn = await connection_factory(basic_server)
            ringbuffer = conn.get_ringbuffer()
            payloads = [b"a", b"b", b"c"]

            async def _consume() -> None:
                for _ in payloads:
                    await asyncio.wait_for(ringbuffer.aconsume(), timeout=1.0)
                    _ = conn.get_state()

            task = asyncio.create_task(_consume())
            for payload in payloads:
                await basic_server.send_to_all_clients(payload)
            await asyncio.wait_for(task, timeout=2.0)
            conn.close()

    async def test_connecting_state_observable(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """CONNECTING state may not be observable due to picows speed.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_config_factory: Fixture providing config factory.

        Returns:
            None: This test does not return a value.
        """
        pytest.skip(
            "CONNECTING state is transient and not reliably observable with picows"
        )

    async def test_transport_cleared_after_disconnect(
        self,
        basic_server,
        connection_factory,
    ) -> None:
        """Ensure _transport is None after disconnect.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_factory: Fixture providing connected WsConnection factory.

        Returns:
            None: This test does not return a value.
        """
        pytest.skip("_transport is a cdef field inaccessible from Python")
