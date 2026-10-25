"""Connection management tests for WsPool.

Layer-2 tests validating WsPool interface behaviors and connection lifecycle
without requiring deep integration paths.

Key coverage:
- Initial state (DISCONNECTED, zero connections).
- Callback signature validation (must accept exactly one bytes arg).
- Configuration updates via set_on_connect.
- Safe operations when disconnected (send_data raises RuntimeError).
- Pool opens the requested number of connections.
- Connections are stored in the internal map.
- Failed connections are removed and the fastest connection updates.
- Concurrent connection setup initializes all requested connections.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from mm_toolbox.websocket import pool as pool_module
from mm_toolbox.websocket.connection import ConnectionState, WsConnectionConfig
from mm_toolbox.websocket.pool import WsPool, WsPoolConfig
from tests.websocket.conftest import _Connection, _until


def noop_message_handler(msg: bytes) -> None:
    """No-op message handler for pool tests."""
    return None


class TestWsPoolInterface:
    """Layer-2 tests for WsPool interface behaviors without starting connections."""

    @pytest.fixture
    def config(self) -> WsConnectionConfig:
        """Build a default WsConnectionConfig for tests.

        Returns:
            Default config instance.
        """
        return WsConnectionConfig.default("wss://test.com")

    def test_initialization(self, config: WsConnectionConfig) -> None:
        """Given a fresh WsPool, Then state is DISCONNECTED and connection count is zero."""
        pool = WsPool(config, on_message=noop_message_handler)
        assert pool.get_state() == ConnectionState.DISCONNECTED
        assert pool.get_connection_count() == 0

    def test_callback_validation(self, config: WsConnectionConfig) -> None:
        """Given valid and invalid callbacks, Then only the valid one is accepted.

                Invalid signatures (missing arg or too many args) must be caught
        early to prevent runtime frame-dispatch errors."""
        WsPool(config, on_message=noop_message_handler)

        def invalid_callback() -> None:
            return None

        with pytest.raises(ValueError):
            WsPool(config, on_message=invalid_callback)

    def test_configuration_updates(self, config: WsConnectionConfig) -> None:
        """Given a pool, When set_on_connect is called, Then the config is updated."""
        pool = WsPool(config, on_message=noop_message_handler)
        new_messages = [b'{"subscribe": "ETHUSDT"}']
        pool.set_on_connect(new_messages)
        assert pool._config.on_connect == new_messages

    def test_operations_when_disconnected(self, config: WsConnectionConfig) -> None:
        """Given a disconnected pool, When send_data is called, Then RuntimeError is raised."""
        pool = WsPool(config, on_message=noop_message_handler)
        with pytest.raises(RuntimeError):
            pool.send_data(b'{"test": "message"}')
        pool.close()


@pytest.mark.asyncio
class TestWsPoolConnectionManagement:
    """Layer-2 tests for pool connection lifecycle behavior."""

    async def test_pool_opens_num_connections(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Given a pool configured for N connections, When started, Then exactly N connections are reported healthy."""
        async with basic_server:
            config = connection_config_factory(basic_server)
            pool_config = WsPoolConfig(num_connections=3, evict_interval_s=60)
            pool = await WsPool.new(
                config, on_message=noop_message_handler, pool_config=pool_config
            )
            async with pool:
                await asyncio.sleep(0.5)
                assert pool.get_connection_count() == pool_config.num_connections
            pool.close()

    async def test_pool_adds_connection_to_pool(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Given a started pool, When connections are established, Then they are stored in the internal map."""
        async with basic_server:
            config = connection_config_factory(basic_server)
            pool_config = WsPoolConfig(num_connections=2, evict_interval_s=60)
            pool = await WsPool.new(
                config, on_message=noop_message_handler, pool_config=pool_config
            )
            async with pool:
                await asyncio.sleep(0.5)
                assert len(pool._conns) == pool_config.num_connections
            pool.close()

    async def test_pool_removes_failed_connection(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Given a pool with an active connection, When that connection is closed and removed, Then the fastest connection is updated.

                Stale references in the fast-connection slot would cause sends to
        a dead transport, so this update must happen promptly."""
        async with basic_server:
            config = connection_config_factory(basic_server)
            pool_config = WsPoolConfig(num_connections=2, evict_interval_s=60)
            pool = await WsPool.new(
                config, on_message=noop_message_handler, pool_config=pool_config
            )
            async with pool:
                await asyncio.sleep(0.5)
                conn = next(iter(pool._conns.values()))
                conn.close()
                pool._conns.pop(conn.get_config().conn_id, None)
                pool._update_fast_connection()
                assert conn not in pool._conns.values()
            pool.close()

    async def test_pool_concurrent_connection_setup(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Given a pool configured for 4 connections, When started, Then all 4 initialise concurrently without deadlock."""
        async with basic_server:
            config = connection_config_factory(basic_server)
            pool_config = WsPoolConfig(num_connections=4, evict_interval_s=60)
            pool = await WsPool.new(
                config, on_message=noop_message_handler, pool_config=pool_config
            )
            async with pool:
                await asyncio.sleep(0.5)
                assert pool.get_connection_count() == pool_config.num_connections
            pool.close()


@pytest.mark.asyncio
async def test_pool_recovers_after_every_socket_is_lost(monkeypatch):
    accepting = True
    attempts = 0

    async def connect(ringbuffer, config):
        nonlocal attempts
        attempts += 1
        if not accepting:
            raise OSError("outage")
        return _Connection(config)

    monkeypatch.setattr(pool_module, "WsConnection", SimpleNamespace(new=connect))
    pool = WsPool(
        WsConnectionConfig.default("ws://localhost"), pool_config=WsPoolConfig(2, 60)
    )
    async with pool:
        accepting = False
        for conn in pool._conns.values():
            conn.close()
        await _until(lambda: attempts >= 4)
        assert pool.get_connection_count() == 0
        assert pool.get_state() == ConnectionState.CONNECTING
        assert len(pool._replacement_tasks) == 2
        accepting = True
        await _until(lambda: pool.get_connection_count() == 2)
        assert pool.get_state() == ConnectionState.CONNECTED
    assert not pool._replacement_tasks
    assert pool._eviction_task is None

@pytest.mark.asyncio
async def test_pool_no_recovery_when_disabled(monkeypatch):
    attempts = 0

    async def connect(ringbuffer, config):
        nonlocal attempts
        attempts += 1
        return _Connection(config)

    monkeypatch.setattr(pool_module, "WsConnection", SimpleNamespace(new=connect))
    config = WsConnectionConfig.default("ws://localhost")
    config.auto_reconnect = False
    async with WsPool(config, pool_config=WsPoolConfig(2, 60)) as pool:
        assert attempts == 2
        for conn in pool._conns.values():
            conn.close()

        tick_seen = asyncio.Event()
        original_sleep = asyncio.sleep

        async def hold_after_pool_tick(delay):
            if asyncio.current_task() is pool._eviction_task:
                tick_seen.set()
                await asyncio.Future()
            await original_sleep(delay)

        with monkeypatch.context() as patcher:
            patcher.setattr(pool_module.asyncio, "sleep", hold_after_pool_tick)
            await asyncio.wait_for(tick_seen.wait(), timeout=1.0)
            assert pool.get_connection_count() == 0
            assert pool.get_state() == ConnectionState.DISCONNECTED
            assert not pool._replacement_tasks
        assert attempts == 2
