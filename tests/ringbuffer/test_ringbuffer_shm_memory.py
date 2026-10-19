"""Format 2 aligned atomic payload and boundary regression tests."""

from __future__ import annotations

import os

import pytest

from mm_toolbox.ringbuffer.shm import ShmSpscConsumer, ShmSpscProducer

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX shared memory")


@pytest.mark.parametrize("length", [0, 1, 6, 7, 8, 9, 15, 16, 23, 24])
def test_atomic_padded_payload_roundtrip(tmp_path, length):
    path = str(tmp_path / "atomic.bin")
    with ShmSpscProducer(path, 32, unlink_on_close=True) as producer:
        with ShmSpscConsumer(path) as consumer:
            payload = bytes(range(length))
            assert producer.insert(payload)
            assert consumer.peekleft() == payload
            assert consumer.peekright() == payload
            assert consumer.unwrapped() == [payload]
            assert len(consumer) == 1
            assert consumer.consume() == payload
            assert len(consumer) == 0


def test_atomic_payload_crosses_ring_boundary(tmp_path):
    path = str(tmp_path / "wrap.bin")
    with ShmSpscProducer(path, 32, unlink_on_close=True) as producer:
        with ShmSpscConsumer(path) as consumer:
            assert producer.insert(b"position")  # aligned span 16
            assert consumer.consume() == b"position"
            payload = b"0123456789abcdef"  # payload starts at offset 24
            assert producer.insert(payload)
            assert consumer.unwrapped() == [payload]
            assert consumer.consume() == payload


def test_padding_affects_overwrite_capacity(tmp_path):
    path = str(tmp_path / "padding.bin")
    with ShmSpscProducer(path, 32, unlink_on_close=True) as producer:
        with ShmSpscConsumer(path) as consumer:
            assert producer.insert_batch([b"a", b"b"])
            assert producer.insert(b"c")
            assert producer.last_insert_dropped == 1
            assert producer.dropped_count == consumer.dropped_count == 1
            assert consumer.consume_all() == [b"b", b"c"]


def test_repeated_atomic_wraps(tmp_path):
    path = str(tmp_path / "wrap.bin")
    with ShmSpscProducer(path, 32, unlink_on_close=True) as producer:
        with ShmSpscConsumer(path) as consumer:
            for sequence in range(100):
                payload = str(sequence).encode() * (1 + sequence % 5)
                assert producer.insert(payload)
                assert consumer.consume() == payload
