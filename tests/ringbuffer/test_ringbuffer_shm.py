"""Shared-memory ring buffer tests.

Exercises single- and multi-process behavior, batch semantics, and header validation.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import random
import struct
from pathlib import Path

import pytest

from mm_toolbox.ringbuffer.shm import (
    SharedBytesRingBufferConsumer,
    SharedBytesRingBufferProducer,
)

pytestmark = pytest.mark.skipif(
    os.name != "posix",
    reason="Shared memory ringbuffer requires POSIX support",
)


@pytest.fixture()
def shm_path(tmp_path: Path) -> str:
    """Build a temporary backing file path for SHM ringbuffer tests.

    Args:
        tmp_path: pytest-provided temporary directory path.

    Returns:
        String path to the backing file.
    """
    return str(tmp_path / "shm_ring.bin")


def _consumer_proc(path: str, n: int, q: mp.Queue) -> None:
    """Consume a fixed number of messages and return a checksum.

    Args:
        path: Filesystem path to the shared memory ringbuffer file.
        n: Number of messages to consume.
        q: Multiprocessing queue used to return results.
    """
    cons = SharedBytesRingBufferConsumer(path, spin_wait=4096)
    try:
        got: list[bytes] = []
        for _ in range(n):
            got.append(cons.consume())
        checksum = sum(len(x) + (x[0] if x else 0) for x in got)
        q.put((len(got), checksum))
    finally:
        cons.close()


class TestSharedBytesRingBuffer:
    """Tests for SharedBytesRingBuffer (shm) implementation."""

    def test_basic_send_receive(self, shm_path: str) -> None:
        """Send one payload and confirm it round-trips correctly.

        Args:
            shm_path: Temporary file path for the shared memory ringbuffer.
        """
        prod = SharedBytesRingBufferProducer(
            shm_path, 1 << 16, create=True, unlink_on_close=True
        )
        cons = SharedBytesRingBufferConsumer(shm_path)
        try:
            payload = b"hello-world"
            assert prod.insert(payload)
            got = cons.consume()
            assert got == payload
        finally:
            cons.close()
            prod.close()
            assert not os.path.exists(shm_path)

    def test_batch_and_drain(self, shm_path: str) -> None:
        """Insert a batch and drain in order.

        Args:
            shm_path: Temporary file path for the shared memory ringbuffer.
        """
        prod = SharedBytesRingBufferProducer(shm_path, 1 << 15, create=True)
        cons = SharedBytesRingBufferConsumer(shm_path)
        try:
            msgs = [f"m{i}".encode() for i in range(1000)]
            assert prod.insert_batch(msgs)
            got = cons.consume_all()
            assert got == msgs
        finally:
            cons.close()
            prod.close()

    def test_insert_overwrites_oldest(self, shm_path: str) -> None:
        """Ensure overwrites drop oldest items when capacity is exceeded.

        Args:
            shm_path: Temporary file path for the shared memory ringbuffer.
        """
        capacity = 1 << 12
        prod = SharedBytesRingBufferProducer(shm_path, capacity, create=True)
        cons = SharedBytesRingBufferConsumer(shm_path)
        try:
            msg = b"x" * (capacity // 8 - 8)
            total = 500
            for _ in range(total):
                assert prod.insert(msg)
            drained = cons.consume_all()
            assert len(drained) > 0
            assert all(m == msg for m in drained)
        finally:
            cons.close()
            prod.close()

    def test_packed_roundtrip(self, shm_path: str) -> None:
        """Verify packed messages unpack correctly.

        Args:
            shm_path: Temporary file path for the shared memory ringbuffer.
        """
        prod = SharedBytesRingBufferProducer(shm_path, 1 << 14, create=True)
        cons = SharedBytesRingBufferConsumer(shm_path)
        try:
            items = [b"a", b"bb", b"ccc", b"dddd"]
            assert prod.insert_packed(items)
            got = cons.consume_packed()
            assert got == items
        finally:
            cons.close()
            prod.close()

    def test_oversize_rejected(self, shm_path: str) -> None:
        """Reject inserts that exceed capacity.

        Args:
            shm_path: Temporary file path for the shared memory ringbuffer.
        """
        capacity = 1 << 12
        prod = SharedBytesRingBufferProducer(shm_path, capacity, create=True)
        cons = SharedBytesRingBufferConsumer(shm_path)
        try:
            oversize = b"z" * (capacity - 7)  # +8 header exceeds
            assert not prod.insert(oversize)
        finally:
            cons.close()
            prod.close()

    def test_batch_then_single_insert_keeps_order(self, shm_path: str) -> None:
        """Keep consistent ordering when a batch is followed by a single insert.

        Args:
            shm_path: Temporary file path for the shared memory ringbuffer.
        """
        prod = SharedBytesRingBufferProducer(shm_path, 1 << 14, create=True)
        cons = SharedBytesRingBufferConsumer(shm_path)
        try:
            msgs = [f"m{i}".encode() for i in range(32)]
            assert prod.insert_batch(msgs)
            assert prod.insert(b"tail")
            got = cons.consume_all()
            assert got == msgs + [b"tail"]
        finally:
            cons.close()
            prod.close()

    def test_attach_rejects_invalid_header(self, shm_path: str) -> None:
        """Reject attaching to a ringbuffer with a corrupted header.

        Args:
            shm_path: Temporary file path for the shared memory ringbuffer.
        """
        prod = SharedBytesRingBufferProducer(
            shm_path, 1 << 12, create=True, unlink_on_close=False
        )
        prod.close()
        try:
            with open(shm_path, "r+b") as handle:
                handle.seek(16)  # mask offset
                handle.write(struct.pack("Q", 123))
            with pytest.raises(RuntimeError):
                SharedBytesRingBufferConsumer(shm_path)
        finally:
            if os.path.exists(shm_path):
                os.unlink(shm_path)

    def test_multiprocess_roundtrip(self, shm_path: str) -> None:
        """Send messages from producer and consume in a separate process.

        Args:
            shm_path: Temporary file path for the shared memory ringbuffer.
        """
        n = 2000
        prod = SharedBytesRingBufferProducer(shm_path, 1 << 18, create=True)
        try:
            q: mp.Queue = mp.Queue()
            p = mp.Process(target=_consumer_proc, args=(shm_path, n, q))
            p.start()

            rng = random.Random(1337)
            for _ in range(n):
                mlen = rng.randint(1, 128)
                msg = bytes(rng.randrange(0, 256) for _ in range(mlen))
                assert prod.insert(msg)

            p.join(timeout=10)
            assert p.exitcode == 0
            got_n, checksum = q.get(timeout=2)
            assert got_n == n
            assert isinstance(checksum, int) and checksum > 0
        finally:
            prod.close()
