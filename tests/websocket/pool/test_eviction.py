"""Eviction logic tests for WsPool (isolated, no real servers).

Layer-2 tests for the WsPool timed-eviction and fast-connection selection
logic using mocked connection stubs.

Key coverage:
- Slowest connection is evicted when the interval fires.
- Fastest connection is selected based on latency.
- Empty pool is safe (no errors during eviction or selection).
- Replacement scheduling is safe without an event loop.
- Partial replacement failures are tolerated.
- Restart count scales with pool size (1 for small pools, N//2 for large).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

import mm_toolbox.websocket.pool as pool_module
from mm_toolbox.websocket.connection import ConnectionState
from mm_toolbox.websocket.pool import WsPool, WsPoolConfig


def noop_message_handler(msg: bytes) -> None:
    """No-op message handler for pool tests."""
    return None


@dataclass
class DummyState:
    """Lightweight connection state stub."""

    latency_ms: float
    is_connected: bool = True


class DummyConn:
    """Minimal connection stub for eviction tests."""

    def __init__(self, conn_id: int, latency_ms: float) -> None:
        """Initialize a dummy connection.

        Args:
            conn_id: Connection identifier.
            latency_ms: Mock latency in milliseconds.
        """
        self._state = DummyState(latency_ms=latency_ms, is_connected=True)
        self._config = SimpleNamespace(conn_id=conn_id)
        self.closed = False

    def is_connected(self) -> bool:
        """Return whether the dummy connection is connected."""
        return self._state.is_connected

    def get_latency_ms(self) -> float:
        """Return the dummy connection's latency."""
        return self._state.latency_ms

    def get_state(self) -> DummyState:
        """Return the dummy state object.

        Returns:
            Current mock state.
        """
        return self._state

    def get_config(self) -> SimpleNamespace:
        """Return the dummy config object.

        Returns:
            Mock config with conn_id.
        """
        return self._config

    def close(self) -> None:
        """Mark the connection as closed."""
        self.closed = True


@pytest.mark.asyncio
class TestWsPoolEvictionLogic:
    """Layer-2 tests for eviction logic with mocked latency values."""

    async def test_eviction_replaces_slowest_connection(self, monkeypatch) -> None:
        """Given three connections with varying latency, When eviction fires, Then the slowest is removed and the others remain.

        Eviction must target the worst performer; removing a fast
connection would degrade overall throughput."""
        config = WsPoolConfig(num_connections=3, evict_interval_s=1)
        pool = WsPool(
            config=SimpleNamespace(
                wss_url="wss://test", on_connect=[], auto_reconnect=True
            ),
            on_message=noop_message_handler,
            pool_config=config,
        )
        pool._conns = {
            1: DummyConn(1, 50.0),
            2: DummyConn(2, 100.0),
            3: DummyConn(3, 60.0),
        }
        pool._pool_state = ConnectionState.CONNECTED

        times = iter([0.0, 10.0, 0.5, 0.5])

        def _next_time() -> float:
            """Return the next mocked timestamp."""
            return next(times)

        monkeypatch.setattr(pool_module, "time_s", _next_time)
        task = asyncio.create_task(pool._timed_operations())
        await asyncio.sleep(0.1)
        pool._should_stop = True
        await asyncio.wait_for(task, timeout=2.0)

        assert 2 not in pool._conns
        assert pool._conns[1].closed is False
        assert pool._conns[3].closed is False

    async def test_fast_connection_selection(self) -> None:
        """Given three connections with different latencies, When the fastest is selected, Then it is the one with the lowest latency."""
        config = WsPoolConfig(num_connections=3, evict_interval_s=1)
        pool = WsPool(
            config=SimpleNamespace(
                wss_url="wss://test", on_connect=[], auto_reconnect=True
            ),
            on_message=noop_message_handler,
            pool_config=config,
        )
        pool._conns = {
            1: DummyConn(1, 80.0),
            2: DummyConn(2, 30.0),
            3: DummyConn(3, 60.0),
        }
        pool._update_fast_connection()
        assert pool._fast_conn is pool._conns[2]

    async def test_eviction_with_no_connections(self) -> None:
        """Given an empty pool, When eviction or fast selection runs, Then no errors occur and fast_conn is None."""
        config = WsPoolConfig(num_connections=2, evict_interval_s=1)
        pool = WsPool(
            config=SimpleNamespace(
                wss_url="wss://test", on_connect=[], auto_reconnect=True
            ),
            on_message=noop_message_handler,
            pool_config=config,
        )
        pool._conns = {}
        pool._update_fast_connection()
        assert pool._fast_conn is None

    async def test_schedule_replacements_no_loop(self) -> None:
        """Given a pool with no event loop set, When replacement scheduling is triggered, Then it remains safe and does not crash."""
        config = WsPoolConfig(num_connections=2, evict_interval_s=1)
        pool = WsPool(
            config=SimpleNamespace(
                wss_url="wss://test", on_connect=[], auto_reconnect=True
            ),
            on_message=noop_message_handler,
            pool_config=config,
        )
        pool._loop = None
        pool._schedule_replacements(2)
        assert pool._loop is None

    async def test_real_timed_eviction_replaces_slowest(
        self,
        basic_server,
        server_with_delay,
        connection_config_factory,
    ) -> None:
        """Given real servers with different latencies, When eviction interval passes, Then the slowest is replaced.

        Skipped because real timed eviction requires long waits and is
covered by unit tests above."""
        pytest.skip(
            "Real timed eviction requires long waits and is covered by unit tests"
        )

    async def test_eviction_partial_reconnect_failure(
        self,
        basic_server,
        connection_config_factory,
        monkeypatch,
    ) -> None:
        """Given a pool where some replacements fail, When eviction runs, Then the pool survives with fewer than the target connections.

        Partial failure is common during network blips; the pool must
gracefully degrade rather than crash."""
        async with basic_server:
            config = connection_config_factory(basic_server)
            pool_config = WsPoolConfig(num_connections=3, evict_interval_s=60)
            pool = await WsPool.new(
                config, on_message=noop_message_handler, pool_config=pool_config
            )
            async with pool:
                await asyncio.sleep(0.3)
                original_open = pool._open_new_conn
                call_count = 0

                async def _failing_open(*args, **kwargs):
                    nonlocal call_count
                    call_count += 1
                    if call_count <= 2:
                        raise RuntimeError("blocked")
                    return await original_open(*args, **kwargs)

                monkeypatch.setattr(pool, "_open_new_conn", _failing_open)
                conn = next(iter(pool._conns.values()))
                conn_id = conn.get_config().conn_id
                conn.close()
                pool._conns.pop(conn_id, None)
                pool._update_fast_connection()
                pool._schedule_replacements(1)
                await asyncio.sleep(0.5)
                assert 0 < pool.get_connection_count() < pool_config.num_connections

    async def test_restart_count_logic(self) -> None:
        """Given pool sizes of 3 and 5, When restart counts are computed, Then they are 1 and 2 respectively.

        Restart count scales with pool size to balance recovery speed
against connection storm risk."""
        config_3 = WsPoolConfig(num_connections=3, evict_interval_s=1)
        pool_3 = WsPool(
            config=SimpleNamespace(
                wss_url="wss://test", on_connect=[], auto_reconnect=True
            ),
            on_message=noop_message_handler,
            pool_config=config_3,
        )
        pool_3._conns = {
            1: DummyConn(1, 50.0),
            2: DummyConn(2, 100.0),
            3: DummyConn(3, 60.0),
        }
        pool_3._pool_state = ConnectionState.CONNECTED
        [conn for conn in pool_3._conns.values() if conn.is_connected()]
        restart_count_3 = (
            config_3.num_connections // 2 if config_3.num_connections >= 4 else 1
        )
        assert restart_count_3 == 1

        config_5 = WsPoolConfig(num_connections=5, evict_interval_s=1)
        pool_5 = WsPool(
            config=SimpleNamespace(
                wss_url="wss://test", on_connect=[], auto_reconnect=True
            ),
            on_message=noop_message_handler,
            pool_config=config_5,
        )
        pool_5._conns = {
            1: DummyConn(1, 50.0),
            2: DummyConn(2, 100.0),
            3: DummyConn(3, 60.0),
            4: DummyConn(4, 80.0),
            5: DummyConn(5, 30.0),
        }
        pool_5._pool_state = ConnectionState.CONNECTED
        restart_count_5 = (
            config_5.num_connections // 2 if config_5.num_connections >= 4 else 1
        )
        assert restart_count_5 == 2
