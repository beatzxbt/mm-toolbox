import os
import random
import multiprocessing as mp
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
    return str(tmp_path / "shm_ring.bin")


def _consumer_proc(path: str, n: int, q: mp.Queue) -> None:
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
        capacity = 1 << 12
        prod = SharedBytesRingBufferProducer(shm_path, capacity, create=True)
        cons = SharedBytesRingBufferConsumer(shm_path)
        try:
            oversize = b"z" * (capacity - 7)  # +8 header exceeds
            assert not prod.insert(oversize)
        finally:
            cons.close()
            prod.close()

    def test_multiprocess_roundtrip(self, shm_path: str) -> None:
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
