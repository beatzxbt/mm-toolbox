"""Shared SPSC/MPSC behavior contracts for shared-memory ring buffers."""

from __future__ import annotations

import asyncio
import gc
import os

import pytest

from mm_toolbox.ringbuffer.shm import (
    ShmMpscConsumer,
    ShmMpscProducer,
    ShmSpscConsumer,
    ShmSpscProducer,
)

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX shared memory")


@pytest.fixture(params=["spsc", "mpsc"])
def ring_types(request):
    if request.param == "spsc":
        return ShmSpscProducer, ShmSpscConsumer, {}
    return ShmMpscProducer, ShmMpscConsumer, {"num_rings": 1}


def test_exact_overwrite_accounting_and_insert_status(tmp_path, ring_types):
    producer_type, consumer_type, kwargs = ring_types
    path = str(tmp_path / "drops.bin")
    with producer_type(path, 64, **kwargs) as producer:
        with consumer_type(path) as consumer:
            assert producer.insert_batch([b"a" * 8, b"b" * 8, b"c" * 8, b"d" * 8])
            assert len(consumer) == 4
            assert producer.insert(b"replacement" * 3)  # padded span 48
            assert producer.last_insert_dropped == 3
            assert producer.dropped_count == consumer.dropped_count == 3
            assert len(producer) == len(consumer) == 2
            assert consumer.consume_all() == [b"d" * 8, b"replacement" * 3]
            assert consumer.is_empty()
            assert len(consumer) == 0
            assert producer.insert(b"kept")
            assert producer.last_insert_dropped == 0
            assert not producer.insert(b"x" * 57)
            assert producer.last_insert_dropped == 0
            assert producer.failed_inserts == 1
            assert consumer.consume() == b"kept"
            assert consumer.dropped_count == 3

def test_oversized_padded_batch_is_rejected_before_mutation(tmp_path, ring_types):
    producer_type, consumer_type, kwargs = ring_types
    path = str(tmp_path / "batch.bin")
    with producer_type(path, 32, **kwargs) as producer:
        with consumer_type(path) as consumer:
            assert producer.insert(b"keep")
            assert not producer.insert_batch([b"a", b"b", b"c"])
            assert producer.dropped_count == 0
            assert producer.failed_inserts == 1
            assert consumer.consume_all() == [b"keep"]

def test_into_preserves_destination_when_too_small(tmp_path, ring_types):
    producer_type, consumer_type, kwargs = ring_types
    path = str(tmp_path / "into.bin")
    with producer_type(path, 64, **kwargs) as producer:
        with consumer_type(path) as consumer:
            assert producer.insert(b"eight123")
            destination = bytearray(b"safe")
            with pytest.raises(ValueError, match="exceeds buffer size"):
                consumer.consume_into(destination)
            assert destination == b"safe"
            assert len(consumer) == 1
            larger = bytearray(b"z" * 16)
            assert consumer.consume_into(larger) == 8
            assert larger == b"eight123" + b"z" * 8
            assert len(consumer) == 0

@pytest.mark.asyncio
async def test_async_idle_consumer_yields_to_producer(tmp_path, ring_types):
    producer_type, consumer_type, kwargs = ring_types
    path = str(tmp_path / "async.bin")
    with producer_type(path, 64, **kwargs) as producer:
        with consumer_type(path) as consumer:
            pending = asyncio.create_task(consumer.aconsume())
            await asyncio.sleep(0)
            assert not pending.done()
            assert producer.insert(b"wake")
            assert await asyncio.wait_for(pending, timeout=2) == b"wake"
            assert consumer.is_empty()

def test_producer_finalizer_releases_claim(tmp_path, ring_types):
    producer_type, consumer_type, kwargs = ring_types
    path = str(tmp_path / "finalizer.bin")
    producer = producer_type(path, 64, **kwargs)
    assert producer.insert(b"before")
    del producer
    gc.collect()
    with producer_type(path, 64, create=False, **kwargs) as replacement:
        assert replacement.insert(b"after")
        with consumer_type(path) as consumer:
            assert consumer.consume_all() == [b"before", b"after"]

def test_into_reuses_capacity_across_payload_sizes_and_empty_records(
    tmp_path, ring_types
):
    producer_type, consumer_type, kwargs = ring_types
    path = str(tmp_path / "scratch.bin")
    with producer_type(path, 256, **kwargs) as producer:
        with consumer_type(path) as consumer:
            for size in [0, 1, 33, 9, 0, 100, 33]:
                payload = bytes([size]) * size
                assert producer.insert(payload)
                destination = bytearray(b"z" * 128)
                assert consumer.consume_into(destination) == size
                assert destination[:size] == payload
                assert destination[size:] == b"z" * (128 - size)
            assert producer.insert_batch([b"", b"x", b"large" * 12])
            buffers = [bytearray(b"z" * 64) for _ in range(3)]
            assert consumer.consume_all_into(buffers) == 3
            assert buffers[0] == b"z" * 64
            assert buffers[1] == b"x" + b"z" * 63
            assert buffers[2] == b"large" * 12 + b"z" * 4
