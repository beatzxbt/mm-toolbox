"""Release regressions for explicit failures, recovery, and task ownership."""

import asyncio
from types import SimpleNamespace

import pytest

from mm_toolbox.ringbuffer.bytes import BytesRingBuffer
from mm_toolbox.websocket import pool as pool_module
from mm_toolbox.websocket import single as single_module
from mm_toolbox.websocket.connection import (
    ConnectionState,
    WsConnection,
    WsConnectionConfig,
)
from mm_toolbox.websocket.pool import WsPool, WsPoolConfig
from mm_toolbox.websocket.single import WsSingle


async def _until(predicate, timeout=3.0):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


class _Connection:
    def __init__(self, config):
        self.config = config
        self.live = True

    def is_connected(self):
        return self.live

    def close(self):
        self.live = False

    def get_config(self):
        return self.config

    def get_latency_ms(self):
        return 1.0


@pytest.mark.asyncio
async def test_pool_partial_start_rolls_back_and_cancels_pending_open(monkeypatch):
    created = []
    cancelled = asyncio.Event()
    calls = 0

    async def connect(ringbuffer, config):
        nonlocal calls
        calls += 1
        if calls == 2:
            await asyncio.sleep(0)
            raise OSError("refused")
        if calls == 3:
            try:
                await asyncio.Future()
            finally:
                cancelled.set()
        conn = _Connection(config)
        created.append(conn)
        return conn

    monkeypatch.setattr(pool_module, "WsConnection", SimpleNamespace(new=connect))
    pool = WsPool(
        WsConnectionConfig.default("ws://localhost"), pool_config=WsPoolConfig(3, 60)
    )
    with pytest.raises(RuntimeError, match="every WebSocket connection"):
        await pool.__aenter__()
    assert cancelled.is_set()
    assert created and all(not conn.live for conn in created)
    assert pool.get_connection_count() == 0
    assert pool.get_state() == ConnectionState.DISCONNECTED
    assert not pool._opening_tasks
    assert not pool._replacement_tasks


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
        for conn in pool._conns.values():
            conn.close()
        await asyncio.sleep(0.25)
        assert attempts == 2
        assert pool.get_state() == ConnectionState.DISCONNECTED


@pytest.mark.asyncio
async def test_pool_reselects_dead_fastest(basic_server, connection_config_factory):
    async with basic_server:
        pool = WsPool(
            connection_config_factory(basic_server), pool_config=WsPoolConfig(2, 60)
        )
        async with pool:
            dead = pool._fast_conn
            dead.close()
            pool.send_data(b"exactly-once", only_fastest=True)
            await _until(
                lambda: b"exactly-once" in basic_server.get_received_messages()
            )
            assert basic_server.get_received_messages().count(b"exactly-once") == 1
            assert pool._fast_conn is not dead


@pytest.mark.asyncio
async def test_single_close_wakes_reader_and_is_terminal():
    ws = WsSingle(WsConnectionConfig.default("ws://localhost"))
    reader = asyncio.create_task(anext(ws))
    await asyncio.sleep(0)
    ws.close()
    ws.close()
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(reader, 0.2)
    with pytest.raises(RuntimeError, match="closed"):
        await ws.start()
    with pytest.raises(RuntimeError, match="closed"):
        await ws.__aenter__()


@pytest.mark.asyncio
async def test_single_close_during_connect_cancels_connector(monkeypatch):
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def connect(ringbuffer, config):
        entered.set()
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    monkeypatch.setattr(single_module, "WsConnection", SimpleNamespace(new=connect))
    ws = WsSingle(WsConnectionConfig.default("ws://localhost"))
    starter = asyncio.create_task(ws.start())
    await entered.wait()
    ws.close()
    await asyncio.wait_for(starter, 0.2)
    assert cancelled.is_set()
    assert ws.get_state() == ConnectionState.DISCONNECTED
    assert all(task.done() for task in ws._tasks)


@pytest.mark.asyncio
async def test_single_close_during_retry_prevents_further_attempts(monkeypatch):
    attempts = 0

    async def connect(ringbuffer, config):
        nonlocal attempts
        attempts += 1
        raise OSError("offline")

    monkeypatch.setattr(single_module, "WsConnection", SimpleNamespace(new=connect))
    ws = WsSingle(WsConnectionConfig.default("ws://localhost"))
    starter = asyncio.create_task(ws.start())
    await _until(lambda: ws._reconnect_task is not None)
    ws.close()
    await asyncio.wait_for(starter, 0.2)
    await asyncio.sleep(0.05)
    assert attempts == 1
    assert all(task.done() for task in ws._tasks)


@pytest.mark.asyncio
async def test_connection_open_cancellation_closes_created_listener(monkeypatch):
    listeners = []
    entered = asyncio.Event()

    class Listener:
        def __init__(self, ringbuffer, config):
            self.closed = False
            listeners.append(self)

        def close(self):
            self.closed = True

    async def connect(**kwargs):
        kwargs["ws_listener_factory"]()
        entered.set()
        await asyncio.Future()

    monkeypatch.setattr("mm_toolbox.websocket.connection.ws_connect", connect)
    task = asyncio.create_task(
        WsConnection.new.__func__(
            Listener, BytesRingBuffer(8), WsConnectionConfig.default("ws://localhost")
        )
    )
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert listeners and all(listener.closed for listener in listeners)
