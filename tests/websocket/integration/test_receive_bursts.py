"""Bounded local bursts exercise the receive path independently of processing."""

import asyncio

import pytest
import websockets

from mm_toolbox.websocket import WsConnectionConfig, WsPool, WsPoolConfig, WsSingle


@pytest.mark.asyncio
@pytest.mark.parametrize("connections", [1, 3])
@pytest.mark.parametrize("during_startup", [False, True])
async def test_receive_bursts_are_delivered_without_overwrite(
    connections, during_startup
):
    total = 8192 if during_startup else 2048
    sent = 0
    all_sent = asyncio.Event()

    async def source(socket):
        nonlocal sent
        if not during_startup:
            await socket.recv()
        for index in range(total):
            await socket.send(index.to_bytes(8, "big") + b"x" * 248)
            if not during_startup and (index + 1) % 256 == 0:
                await asyncio.sleep(0.05)
        sent += 1
        if sent == connections:
            all_sent.set()
        await socket.wait_closed()

    async with websockets.serve(source, "127.0.0.1", 0, compression=None) as server:
        uri = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
        config = WsConnectionConfig.default(uri, auto_reconnect=False)
        client = (
            WsSingle(config)
            if connections == 1
            else WsPool(config, pool_config=WsPoolConfig(connections, 60))
        )
        received = []
        async with client:
            if during_startup:
                # No consumer exists yet. More than the queue capacity is
                # offered, and reading must stop rather than overwrite it.
                async with asyncio.timeout(3):
                    while not client._ringbuffer.is_reading_paused():
                        await asyncio.sleep(0)
                assert len(client._ringbuffer) <= config.receive_buffer_capacity
                assert (
                    client._ringbuffer.get_buffered_bytes()
                    <= config.receive_buffer_max_bytes
                )
                assert client.get_dropped_messages() == 0

            async def consume():
                async for payload in client:
                    received.append(int.from_bytes(payload[:8], "big"))

            reader = asyncio.create_task(consume())
            try:
                if not during_startup:
                    client.send_data(b"go", only_fastest=False)
                await asyncio.wait_for(all_sent.wait(), 5)
                async with asyncio.timeout(5):
                    while len(received) < total or len(client._ringbuffer):
                        await asyncio.sleep(0)
                conns = (
                    [client._ws_conn]
                    if connections == 1
                    else list(client._conns.values())
                )
                async with asyncio.timeout(5):
                    while any(conn.get_seq_id() != total for conn in conns) or len(
                        client._ringbuffer
                    ):
                        await asyncio.sleep(0)
                assert received == list(range(total))
                assert client.get_dropped_messages() == 0
                assert client._ringbuffer.get_buffered_bytes() == 0
            finally:
                reader.cancel()
                await asyncio.gather(reader, return_exceptions=True)
