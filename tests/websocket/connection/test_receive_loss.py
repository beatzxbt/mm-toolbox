"""Receive overwrite observability survives native and wrapper lifecycles."""

import asyncio
import logging

import pytest
import pytest_asyncio
import websockets

import mm_toolbox.websocket.connection as connection_module
from mm_toolbox.ringbuffer.bytes import BytesRingBuffer
from mm_toolbox.websocket import (
    WsConnection,
    WsConnectionConfig,
    WsPool,
    WsPoolConfig,
    WsSingle,
)
from mm_toolbox.websocket.connection import _ReceiveBuffer


async def _received(conn, count):
    async with asyncio.timeout(2):
        while conn.get_seq_id() < count:
            await asyncio.sleep(0)


@pytest_asyncio.fixture
async def burst_server():
    async def send_burst(socket):
        for index in range(132):
            await socket.send(str(index).encode())
        await socket.wait_closed()

    async with websockets.serve(send_burst, "127.0.0.1", 0) as server:
        yield f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"


@pytest.mark.asyncio
async def test_native_overwrite_keeps_exact_count_with_sampled_warnings(
    burst_server, caplog
):
    ring = BytesRingBuffer(4)
    config = WsConnectionConfig.default(burst_server, auto_reconnect=False)
    with caplog.at_level(logging.WARNING, logger="mm_toolbox.websocket.connection"):
        conn = await WsConnection.new(ring, config)
        try:
            await _received(conn, 132)
            assert conn.get_dropped_messages() == 128
            assert ring.consume_all() == [b"128", b"129", b"130", b"131"]
        finally:
            conn.close()
    warnings = [
        record
        for record in caplog.records
        if "receive buffer overflow" in record.message
    ]
    assert [record.args[1] for record in warnings] == [1, 2, 4, 8, 16, 32, 64, 128]
    assert conn.get_dropped_messages() == 128


@pytest.mark.asyncio
async def test_single_lifetime_count_survives_reconnect_and_close(burst_server):
    ws = WsSingle(WsConnectionConfig.default(burst_server, auto_reconnect=False))
    ws._ringbuffer = _ReceiveBuffer(4)
    try:
        for expected in (128, 256):
            await ws._connect()
            await _received(ws._ws_conn, 132)
            assert ws.get_dropped_messages() == expected
            ws._ws_conn.close()
            ws._ringbuffer.consume_all()
    finally:
        ws.close()
    assert ws.get_dropped_messages() == 256


@pytest.mark.asyncio
@pytest.mark.parametrize("client_type", [WsSingle, WsPool])
async def test_loss_during_failed_open_is_counted_before_listener_is_returned(
    client_type, burst_server, monkeypatch
):
    original_connect = connection_module.ws_connect

    async def fail_after_receiving(**kwargs):
        transport, listener = await original_connect(**kwargs)
        await _received(listener, 132)
        raise RuntimeError("injected open failure after frames arrived")

    monkeypatch.setattr(connection_module, "ws_connect", fail_after_receiving)
    config = WsConnectionConfig.default(burst_server, auto_reconnect=False)
    client = client_type(config)
    client._ringbuffer = _ReceiveBuffer(4)
    try:
        with pytest.raises(RuntimeError):
            if isinstance(client, WsSingle):
                await client._connect()
            else:
                await client._open_new_conn()
        assert client.get_dropped_messages() == 128
        assert client.get_connection_count() == 0
    finally:
        client.close()
    assert client.get_dropped_messages() == 128


@pytest.mark.asyncio
async def test_pool_lifetime_count_survives_retirement_close_and_restart(burst_server):
    pool = WsPool(
        WsConnectionConfig.default(burst_server, auto_reconnect=False),
        pool_config=WsPoolConfig(num_connections=2, evict_interval_s=60),
    )
    pool._ringbuffer = _ReceiveBuffer(4)
    try:
        first = await pool._open_new_conn()
        await _received(first, 132)
        pool._ringbuffer.consume_all()
        first.close()
        pool._conns.pop(first.get_config().conn_id)
        second = await pool._open_new_conn()
        await _received(second, 132)
        assert pool.get_dropped_messages() == 256
        pool.close()
        assert pool.get_dropped_messages() == 256
        pool._should_stop = False
        third = await pool._open_new_conn()
        await _received(third, 132)
        assert pool.get_dropped_messages() == 384
    finally:
        pool.close()
    assert pool.get_dropped_messages() == 384
