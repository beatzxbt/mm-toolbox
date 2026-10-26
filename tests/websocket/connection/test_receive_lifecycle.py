"""Direct receive wakeups retain close, cancellation and batch semantics."""

import asyncio

import pytest

from mm_toolbox.websocket import WsConnectionConfig, WsPool, WsPoolConfig, WsSingle
from mm_toolbox.websocket.connection import _ReceiveBuffer


@pytest.mark.asyncio
async def test_single_pending_reader_drains_payload_before_same_turn_close():
    client = WsSingle(WsConnectionConfig.default("ws://localhost"))
    reader = asyncio.create_task(anext(client))
    await asyncio.sleep(0)
    client._ringbuffer.insert(b"last payload")
    client.close()
    assert await reader == b"last payload"
    with pytest.raises(StopAsyncIteration):
        await anext(client)


@pytest.mark.asyncio
async def test_old_pool_reader_cannot_consume_after_immediate_restart():
    client = WsPool(
        WsConnectionConfig.default("ws://localhost"), pool_config=WsPoolConfig(2, 60)
    )
    old_reader = asyncio.create_task(anext(client))
    await asyncio.sleep(0)
    client.close()
    client._ringbuffer.reopen()
    client._ringbuffer.insert(b"new session")
    with pytest.raises(StopAsyncIteration):
        await old_reader
    assert await anext(client) == b"new session"
    client.close()
    with pytest.raises(StopAsyncIteration):
        await anext(client)


@pytest.mark.asyncio
@pytest.mark.parametrize("client_type", [WsSingle, WsPool])
async def test_cancelled_empty_read_does_not_spawn_tasks_or_steal_next_payload(
    client_type,
):
    client = client_type(WsConnectionConfig.default("ws://localhost"))
    before = asyncio.all_tasks()
    reader = asyncio.create_task(anext(client))
    await asyncio.sleep(0)
    assert asyncio.all_tasks() - before == {reader}
    reader.cancel()
    with pytest.raises(asyncio.CancelledError):
        await reader
    client._ringbuffer.insert(b"after cancellation")
    assert await anext(client) == b"after cancellation"
    client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "drain", ["consume_all", "consume_into", "consume_all_into", "clear"]
)
async def test_batch_operations_release_bytes_and_reset_readiness(drain):
    ring = _ReceiveBuffer(8)
    ring.configure(4096)
    ring.insert_batch([b"one", b"two"])
    assert ring.get_buffered_bytes() == 6
    if drain == "consume_all":
        assert ring.consume_all() == [b"one", b"two"]
    elif drain == "consume_into":
        for expected in (b"one", b"two"):
            destination = bytearray(3)
            assert ring.consume_into(destination) == 3
            assert destination == expected
    elif drain == "consume_all_into":
        buffers = [bytearray(3), bytearray(3)]
        assert ring.consume_all_into(buffers) == 2
        assert buffers == [b"one", b"two"]
    else:
        ring.clear()
    assert ring.get_buffered_bytes() == 0
    reader = asyncio.create_task(ring.aconsume())
    await asyncio.sleep(0)
    assert not reader.done()
    ring.insert_batch([b"next"])
    assert await reader == b"next"
    assert ring.get_buffered_bytes() == 0
