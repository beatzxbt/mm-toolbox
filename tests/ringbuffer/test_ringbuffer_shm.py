"""Shared-memory ring buffer tests.

Exercises single- and multi-process behavior, batch semantics, and header validation.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import random
import struct
import threading
import time
from pathlib import Path

import pytest

from mm_toolbox.ringbuffer.shm import (
    ShmSpscConsumer,
    ShmSpscProducer,
    ShmSpscConfig,
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
    cons = ShmSpscConsumer(path, spin_wait=4096)
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
        prod = ShmSpscProducer(shm_path, 1 << 16, create=True, unlink_on_close=True)
        cons = ShmSpscConsumer(shm_path)
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
        prod = ShmSpscProducer(shm_path, 1 << 15, create=True)
        cons = ShmSpscConsumer(shm_path)
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
        prod = ShmSpscProducer(shm_path, capacity, create=True)
        cons = ShmSpscConsumer(shm_path)
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
        prod = ShmSpscProducer(shm_path, 1 << 14, create=True)
        cons = ShmSpscConsumer(shm_path)
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
        prod = ShmSpscProducer(shm_path, capacity, create=True)
        cons = ShmSpscConsumer(shm_path)
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
        prod = ShmSpscProducer(shm_path, 1 << 14, create=True)
        cons = ShmSpscConsumer(shm_path)
        try:
            msgs = [f"m{i}".encode() for i in range(32)]
            assert prod.insert_batch(msgs)
            assert prod.insert(b"tail")
            got = cons.consume_all()
            assert got == msgs + [b"tail"]
        finally:
            cons.close()
            prod.close()

    def test_empty_payload_insert(self, shm_path: str) -> None:
        """Empty payload should insert and consume correctly.

        Args:
            shm_path: Temporary file path for the shared memory ringbuffer.
        """
        prod = ShmSpscProducer(shm_path, 1 << 12, create=True)
        cons = ShmSpscConsumer(shm_path)
        try:
            assert prod.insert(b"")
            got = cons.consume()
            assert got == b""
        finally:
            cons.close()
            prod.close()

    def test_exact_capacity_message(self, shm_path: str) -> None:
        """Insert a message of exactly capacity - 8 bytes (the max).

        Args:
            shm_path: Temporary file path for the shared memory ringbuffer.
        """
        capacity = 1 << 12
        prod = ShmSpscProducer(shm_path, capacity, create=True)
        cons = ShmSpscConsumer(shm_path)
        try:
            msg = b"x" * (capacity - 8)
            assert prod.insert(msg)
            got = cons.consume()
            assert got == msg
        finally:
            cons.close()
            prod.close()

    def test_peekleft_empty_returns_none(self, shm_path: str) -> None:
        """peekleft on empty buffer returns None.

        Args:
            shm_path: Temporary file path for the shared memory ringbuffer.
        """
        prod = ShmSpscProducer(shm_path, 1 << 12, create=True)
        cons = ShmSpscConsumer(shm_path)
        try:
            assert cons.peekleft() is None
        finally:
            cons.close()
            prod.close()

    def test_peekright_empty_returns_none(self, shm_path: str) -> None:
        """peekright on empty buffer returns None.

        Args:
            shm_path: Temporary file path for the shared memory ringbuffer.
        """
        prod = ShmSpscProducer(shm_path, 1 << 12, create=True)
        cons = ShmSpscConsumer(shm_path)
        try:
            assert cons.peekright() is None
        finally:
            cons.close()
            prod.close()

    def test_insert_batch_empty_list(self, shm_path: str) -> None:
        """insert_batch with empty list returns True.

        Args:
            shm_path: Temporary file path for the shared memory ringbuffer.
        """
        prod = ShmSpscProducer(shm_path, 1 << 12, create=True)
        try:
            assert prod.insert_batch([])
        finally:
            prod.close()

    def test_insert_packed_empty_list(self, shm_path: str) -> None:
        """insert_packed with empty list returns True.

        Args:
            shm_path: Temporary file path for the shared memory ringbuffer.
        """
        prod = ShmSpscProducer(shm_path, 1 << 12, create=True)
        try:
            assert prod.insert_packed([])
        finally:
            prod.close()

    def test_idempotent_close(self, shm_path: str) -> None:
        """Calling close twice should not crash.

        Args:
            shm_path: Temporary file path for the shared memory ringbuffer.
        """
        prod = ShmSpscProducer(shm_path, 1 << 12, create=True, unlink_on_close=True)
        prod.close()
        prod.close()
        assert not os.path.exists(shm_path)

    def test_consume_with_yield(self, shm_path: str) -> None:
        """Consume still works after replacing sleep with sched_yield.

        Args:
            shm_path: Temporary file path for the shared memory ringbuffer.
        """
        prod = ShmSpscProducer(shm_path, 1 << 12, create=True)
        cons = ShmSpscConsumer(shm_path)
        try:
            payload = b"hello"
            assert prod.insert(payload)
            got = cons.consume()
            assert got == payload
        finally:
            cons.close()
            prod.close()

    def test_attach_rejects_invalid_header(self, shm_path: str) -> None:
        """Reject attaching to a ringbuffer with a corrupted header.

        Args:
            shm_path: Temporary file path for the shared memory ringbuffer.
        """
        prod = ShmSpscProducer(shm_path, 1 << 12, create=True, unlink_on_close=False)
        prod.close()
        try:
            with open(shm_path, "r+b") as handle:
                handle.seek(16)  # mask offset
                handle.write(struct.pack("Q", 123))
            with pytest.raises(RuntimeError):
                ShmSpscConsumer(shm_path)
        finally:
            if os.path.exists(shm_path):
                os.unlink(shm_path)

    def test_multiprocess_roundtrip(self, shm_path: str) -> None:
        """Send messages from producer and consume in a separate process.

        Args:
            shm_path: Temporary file path for the shared memory ringbuffer.
        """
        n = 2000
        prod = ShmSpscProducer(shm_path, 1 << 18, create=True)
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
            if os.path.exists(shm_path):
                os.unlink(shm_path)

    # --- P0 Critical ---

    def test_insert_char_roundtrip(self, shm_path: str) -> None:
        """insert_char(b'hello', 5) should round-trip through consume().

        Args:
            shm_path: Temporary file path for the shared memory ringbuffer.
        """
        prod = ShmSpscProducer(shm_path, 1 << 12, create=True)
        cons = ShmSpscConsumer(shm_path)
        try:
            assert prod.insert_char(b"hello", 5)
            got = cons.consume()
            assert got == b"hello"
        finally:
            cons.close()
            prod.close()

    def test_create_false_attach(self, shm_path: str) -> None:
        """Create with create=True, then attach a second producer with create=False.

        Args:
            shm_path: Temporary file path for the shared memory ringbuffer.
        """
        prod1 = ShmSpscProducer(shm_path, 1 << 12, create=True)
        try:
            prod2 = ShmSpscProducer(shm_path, 1 << 12, create=False)
            try:
                assert prod2.insert(b"attached")
                cons = ShmSpscConsumer(shm_path)
                try:
                    assert cons.consume() == b"attached"
                finally:
                    cons.close()
            finally:
                prod2.close()
        finally:
            prod1.close()

    def test_context_manager_producer(self, shm_path: str) -> None:
        """Use producer as a context manager and verify auto-close.

        Args:
            shm_path: Temporary file path for the shared memory ringbuffer.
        """
        with ShmSpscProducer(
            shm_path, 1 << 12, create=True, unlink_on_close=True
        ) as prod:
            assert prod.insert(b"ctx")
        assert not os.path.exists(shm_path)

    def test_context_manager_consumer(self, shm_path: str) -> None:
        """Use consumer as a context manager and verify auto-close.

        Args:
            shm_path: Temporary file path for the shared memory ringbuffer.
        """
        prod = ShmSpscProducer(shm_path, 1 << 12, create=True)
        try:
            prod.insert(b"ctx")
            with ShmSpscConsumer(shm_path) as cons:
                assert cons.consume() == b"ctx"
        finally:
            prod.close()

    def test_len_property(self, shm_path: str) -> None:
        """Verify len(producer) behavior across inserts, consumes, and overwrites.

        Args:
            shm_path: Temporary file path for the shared memory ringbuffer.
        """
        capacity = 1 << 12
        prod = ShmSpscProducer(shm_path, capacity, create=True)
        cons = ShmSpscConsumer(shm_path)
        try:
            assert len(prod) == 0
            assert prod.insert(b"a")
            assert len(prod) == 1
            assert prod.insert(b"b")
            assert len(prod) == 2
            cons.consume()
            assert len(prod) == 1
            cons.consume()
            assert len(prod) == 0
            # Overwrite case: fill until oldest dropped
            msg = b"x" * (capacity // 8 - 8)
            for _ in range(20):
                assert prod.insert(msg)
            assert len(prod) > 0
        finally:
            cons.close()
            prod.close()

    def test_peekleft_nonempty(self, shm_path: str) -> None:
        """peekleft returns first item without consuming it.

        Args:
            shm_path: Temporary file path for the shared memory ringbuffer.
        """
        prod = ShmSpscProducer(shm_path, 1 << 12, create=True)
        cons = ShmSpscConsumer(shm_path)
        try:
            assert prod.insert(b"first")
            assert prod.insert(b"second")
            assert cons.peekleft() == b"first"
            assert cons.consume() == b"first"
            assert cons.consume() == b"second"
        finally:
            cons.close()
            prod.close()

    def test_peekright_nonempty(self, shm_path: str) -> None:
        """peekright returns last item without consuming it.

        Args:
            shm_path: Temporary file path for the shared memory ringbuffer.
        """
        prod = ShmSpscProducer(shm_path, 1 << 12, create=True)
        cons = ShmSpscConsumer(shm_path)
        try:
            assert prod.insert(b"first")
            assert prod.insert(b"second")
            assert cons.peekright() == b"second"
            assert cons.consume() == b"first"
            assert cons.consume() == b"second"
        finally:
            cons.close()
            prod.close()

    def test_config_validation_empty_path(self) -> None:
        """ShmSpscConfig with empty path raises ValueError."""
        with pytest.raises(ValueError):
            ShmSpscConfig(path="", capacity_bytes=1024)

    def test_config_validation_zero_capacity(self) -> None:
        """ShmSpscConfig with zero capacity raises ValueError."""
        with pytest.raises(ValueError):
            ShmSpscConfig(path="/tmp/x", capacity_bytes=0)

    def test_config_validation_zero_spin_wait(self) -> None:
        """ShmSpscConfig with spin_wait=0 raises ValueError."""
        with pytest.raises(ValueError):
            ShmSpscConfig(path="/tmp/x", capacity_bytes=1024, spin_wait=0)

    def test_config_validation_unlink_without_create(self) -> None:
        """ShmSpscConfig with unlink_on_close=True and create=False raises ValueError."""
        with pytest.raises(ValueError):
            ShmSpscConfig(
                path="/tmp/x",
                capacity_bytes=1024,
                create=False,
                unlink_on_close=True,
            )

    def test_config_default(self) -> None:
        """ShmSpscConfig.default() returns valid config with expected defaults."""
        cfg = ShmSpscConfig.default()
        assert cfg.path == "/tmp/shm_ring.bin"
        assert cfg.capacity_bytes == 1 << 16
        assert cfg.create is True
        assert cfg.unlink_on_close is False
        assert cfg.spin_wait == 1024

    def test_config_kwargs(self) -> None:
        """producer_kwargs and consumer_kwargs return correct dicts."""
        cfg = ShmSpscConfig(path="/tmp/x", capacity_bytes=2048, spin_wait=512)
        pkw = cfg.producer_kwargs()
        assert pkw["path"] == "/tmp/x"
        assert pkw["capacity_bytes"] == 2048
        assert pkw["spin_wait"] == 512
        ckw = cfg.consumer_kwargs()
        assert ckw["path"] == "/tmp/x"
        assert ckw["spin_wait"] == 512
        assert "capacity_bytes" not in ckw

    def test_truncated_file_rejected(self, shm_path: str) -> None:
        """File smaller than 64 bytes is rejected on consumer attach.

        Args:
            shm_path: Temporary file path for the shared memory ringbuffer.
        """
        with open(shm_path, "wb") as f:
            f.write(b"\x00" * 32)
        with pytest.raises(RuntimeError):
            ShmSpscConsumer(shm_path)

    # --- P1 Important ---

    def test_spin_wait_small(self, shm_path: str) -> None:
        """Producer and consumer with spin_wait=1 work correctly.

        Args:
            shm_path: Temporary file path for the shared memory ringbuffer.
        """
        prod = ShmSpscProducer(shm_path, 1 << 12, create=True, spin_wait=1)
        cons = ShmSpscConsumer(shm_path, spin_wait=1)
        try:
            assert prod.insert(b"small")
            assert cons.consume() == b"small"
        finally:
            cons.close()
            prod.close()

    def test_spin_wait_large(self, shm_path: str) -> None:
        """Producer and consumer with spin_wait=65536 work correctly.

        Args:
            shm_path: Temporary file path for the shared memory ringbuffer.
        """
        prod = ShmSpscProducer(shm_path, 1 << 12, create=True, spin_wait=65536)
        cons = ShmSpscConsumer(shm_path, spin_wait=65536)
        try:
            assert prod.insert(b"large")
            assert cons.consume() == b"large"
        finally:
            cons.close()
            prod.close()

    def test_mismatched_spin_wait(self, shm_path: str) -> None:
        """Producer and consumer with different spin_wait values interoperate.

        Args:
            shm_path: Temporary file path for the shared memory ringbuffer.
        """
        prod = ShmSpscProducer(shm_path, 1 << 12, create=True, spin_wait=100)
        cons = ShmSpscConsumer(shm_path, spin_wait=10000)
        try:
            assert prod.insert(b"mismatch")
            assert cons.consume() == b"mismatch"
        finally:
            cons.close()
            prod.close()

    def test_insert_batch_exceeds_capacity(self, shm_path: str) -> None:
        """Batch whose total size exceeds capacity returns False.

        Args:
            shm_path: Temporary file path for the shared memory ringbuffer.
        """
        capacity = 1 << 8
        prod = ShmSpscProducer(shm_path, capacity, create=True)
        try:
            msgs = [b"x" * (capacity - 7)]
            assert not prod.insert_batch(msgs)
        finally:
            prod.close()

    def test_insert_packed_oversized_item(self, shm_path: str) -> None:
        """Item with len > 0xFFFFFFFF returns False.

        Args:
            shm_path: Temporary file path for the shared memory ringbuffer.
        """
        prod = ShmSpscProducer(shm_path, 1 << 20, create=True)
        try:
            oversized = b"x" * (0xFFFFFFFF + 1)
            assert not prod.insert_packed([oversized])
        finally:
            prod.close()

    def test_consume_packed_corrupted(self, shm_path: str) -> None:
        """Corrupted length prefix inside packed message raises ValueError.

        Args:
            shm_path: Temporary file path for the shared memory ringbuffer.
        """
        prod = ShmSpscProducer(shm_path, 1 << 14, create=True, unlink_on_close=False)
        prod.insert_packed([b"a", b"bb"])
        prod.close()
        try:
            with open(shm_path, "r+b") as f:
                # Header is 64 bytes; packed msg starts at offset 64.
                # 8-byte total size, then 4-byte length prefix for first item.
                f.seek(64 + 8)
                # Corrupt length to a huge value
                f.write(struct.pack("<I", 0x7FFFFFFF))
            cons = ShmSpscConsumer(shm_path)
            try:
                with pytest.raises(ValueError):
                    cons.consume_packed()
            finally:
                cons.close()
        finally:
            if os.path.exists(shm_path):
                os.unlink(shm_path)

    def test_unlink_on_close_false(self, shm_path: str) -> None:
        """Close producer with unlink_on_close=False; file still exists.

        Args:
            shm_path: Temporary file path for the shared memory ringbuffer.
        """
        prod = ShmSpscProducer(shm_path, 1 << 12, create=True, unlink_on_close=False)
        prod.close()
        assert os.path.exists(shm_path)
        os.unlink(shm_path)

    def test_timestamp_properties(self, shm_path: str) -> None:
        """Timestamps are non-zero and monotonically increase.

        Args:
            shm_path: Temporary file path for the shared memory ringbuffer.
        """
        prod = ShmSpscProducer(shm_path, 1 << 12, create=True)
        cons = ShmSpscConsumer(shm_path)
        try:
            assert prod.latest_insert_time_ns == 0
            assert prod.latest_consume_time_ns == 0
            prod.insert(b"first")
            t1 = prod.latest_insert_time_ns
            assert t1 > 0
            time.sleep(0.001)
            prod.insert(b"second")
            t2 = prod.latest_insert_time_ns
            assert t2 > t1
            cons.consume()
            c1 = cons.latest_consume_time_ns
            assert c1 > 0
            time.sleep(0.001)
            cons.consume()
            c2 = cons.latest_consume_time_ns
            assert c2 > c1
        finally:
            cons.close()
            prod.close()

    def test_thread_safety(self, shm_path: str) -> None:
        """One thread producing, one thread consuming 1000 messages each.

        Args:
            shm_path: Temporary file path for the shared memory ringbuffer.
        """
        n = 1000
        prod = ShmSpscProducer(shm_path, 1 << 18, create=True)
        cons = ShmSpscConsumer(shm_path)
        received: list[bytes] = []

        def consumer_thread() -> None:
            for _ in range(n):
                received.append(cons.consume())

        t = threading.Thread(target=consumer_thread)
        t.start()
        for i in range(n):
            assert prod.insert(str(i).encode())
        t.join(timeout=30)
        assert len(received) == n
        assert received[0] == b"0"
        assert received[-1] == str(n - 1).encode()
        cons.close()
        prod.close()

    def test_consume_all_empty(self, shm_path: str) -> None:
        """consume_all on empty buffer returns empty list.

        Args:
            shm_path: Temporary file path for the shared memory ringbuffer.
        """
        prod = ShmSpscProducer(shm_path, 1 << 12, create=True)
        cons = ShmSpscConsumer(shm_path)
        try:
            assert cons.consume_all() == []
        finally:
            cons.close()
            prod.close()

    # --- P2 Nice-to-have ---

    def test_invalid_path_oserror(self) -> None:
        """Invalid path raises OSError."""
        with pytest.raises(OSError):
            ShmSpscProducer("/nonexistent/dir/file", 1 << 12, create=True)

    def test_capacity_bytes_zero_handled(self, shm_path: str) -> None:
        """capacity_bytes=0 uses pow2_at_least(1) and does not crash.

        Args:
            shm_path: Temporary file path for the shared memory ringbuffer.
        """
        prod = ShmSpscProducer(shm_path, 0, create=True, unlink_on_close=False)
        try:
            prod.close()
            # Header (64 bytes) + 1 byte capacity
            assert os.path.getsize(shm_path) == 65
        finally:
            if os.path.exists(shm_path):
                os.unlink(shm_path)

    def test_consume_iterable(self, shm_path: str) -> None:
        """consume_iterable yields items in FIFO order.

        Args:
            shm_path: Temporary file path for the shared memory ringbuffer.
        """
        prod = ShmSpscProducer(shm_path, 1 << 12, create=True)
        cons = ShmSpscConsumer(shm_path)
        try:
            prod.insert(b"first")
            prod.insert(b"second")
            prod.insert(b"third")
            gen = cons.consume_iterable()
            assert next(gen) == b"first"
            assert next(gen) == b"second"
            assert next(gen) == b"third"
        finally:
            cons.close()
            prod.close()

    @pytest.mark.asyncio
    async def test_aconsume(self, shm_path: str) -> None:
        """aconsume returns items asynchronously.

        Args:
            shm_path: Temporary file path for the shared memory ringbuffer.
        """
        prod = ShmSpscProducer(shm_path, 1 << 12, create=True)
        cons = ShmSpscConsumer(shm_path)
        try:
            prod.insert(b"async")
            result = await cons.aconsume()
            assert result == b"async"
        finally:
            cons.close()
            prod.close()

    @pytest.mark.asyncio
    async def test_aconsume_iterable(self, shm_path: str) -> None:
        """aconsume_iterable yields items asynchronously.

        Args:
            shm_path: Temporary file path for the shared memory ringbuffer.
        """
        prod = ShmSpscProducer(shm_path, 1 << 12, create=True)
        cons = ShmSpscConsumer(shm_path)
        try:
            prod.insert(b"one")
            prod.insert(b"two")
            collected = []
            async for item in cons.aconsume_iterable():
                collected.append(item)
                if len(collected) == 2:
                    break
            assert collected == [b"one", b"two"]
        finally:
            cons.close()
            prod.close()
