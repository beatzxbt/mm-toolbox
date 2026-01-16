"""State machine and snapshot tests for WsConnection.

Exercises ConnectionState enum values, WsConnectionState properties,
and state transitions during connection lifecycle.
"""

from __future__ import annotations

import asyncio

import pytest

from mm_toolbox.ringbuffer.bytes import BytesRingBuffer
from mm_toolbox.websocket.connection import (
    ConnectionState,
    LatencyTrackerState,
    WsConnectionState,
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


class TestWsConnectionState:
    """Validate WsConnectionState properties and transitions."""

    @pytest.fixture
    def sample_state(self) -> WsConnectionState:
        """Create a sample WsConnectionState for unit testing.

        Returns:
            WsConnectionState: Initialized state instance.
        """
        ringbuffer = BytesRingBuffer(max_capacity=64, only_insert_unique=False)
        latency = LatencyTrackerState.default()
        latency.latency_ms = 25.5

        return WsConnectionState(
            seq_id=42,
            state=ConnectionState.CONNECTED,
            ringbuffer=ringbuffer,
            latency=latency,
        )

    def test_state_properties(self, sample_state: WsConnectionState) -> None:
        """Verify state properties expose expected values.

        Args:
            sample_state (WsConnectionState): Sample state fixture.

        Returns:
            None: This test does not return a value.
        """
        assert sample_state.seq_id == 42
        assert sample_state.state == ConnectionState.CONNECTED
        assert sample_state.is_connected is True
        assert sample_state.latency_ms == 25.5

    def test_connection_state_transitions(
        self, sample_state: WsConnectionState
    ) -> None:
        """Verify connection state flag updates.

        Args:
            sample_state (WsConnectionState): Sample state fixture.

        Returns:
            None: This test does not return a value.
        """
        assert sample_state.is_connected is True

        sample_state.state = ConnectionState.DISCONNECTED
        assert sample_state.is_connected is False

        sample_state.state = ConnectionState.CONNECTING
        assert sample_state.is_connected is False

    def test_recent_message_access(self, sample_state: WsConnectionState) -> None:
        """Verify most recent message property reads the ringbuffer tail.

        Args:
            sample_state (WsConnectionState): Sample state fixture.

        Returns:
            None: This test does not return a value.
        """
        messages = [b'{"msg": 1}', b'{"msg": 2}']
        for msg in messages:
            sample_state.ringbuffer.insert(msg)

        assert sample_state.recent_message == messages[-1]


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
            assert conn.get_state().state == ConnectionState.CONNECTED
            conn.close()
            await asyncio.sleep(0.1)
            assert conn.get_state().state == ConnectionState.DISCONNECTED

    async def test_get_state_snapshot(
        self,
        basic_server,
        connection_factory,
    ) -> None:
        """Ensure get_state returns a valid WsConnectionState snapshot.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_factory: Fixture providing connected WsConnection factory.

        Returns:
            None: This test does not return a value.
        """
        async with basic_server:
            conn = await connection_factory(basic_server)
            state = conn.get_state()
            assert isinstance(state, WsConnectionState)
            assert state.state in (
                ConnectionState.DISCONNECTED,
                ConnectionState.CONNECTING,
                ConnectionState.CONNECTED,
            )
            if not state.ringbuffer.is_empty():
                assert state.recent_message == state.ringbuffer.peekright()
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
            ringbuffer = conn.get_state().ringbuffer
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
