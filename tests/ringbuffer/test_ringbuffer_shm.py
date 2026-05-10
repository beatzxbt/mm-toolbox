"""Shared-memory ring buffer tests.

Layer 3 tests: exercises single- and multi-process behavior, batch semantics,
and header validation for ShmSpscProducer/ShmSpscConsumer.
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
    """Layer 3: Tests for SharedBytesRingBuffer (shm) implementation."""

    def test_basic_send_receive(self, shm_path: str) -> None:
        """Given one payload, When sent and received, Then round-trips correctly."""
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
        """Given batch insert, When drained, Then order preserved."""
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
        """Given capacity exceeded, When inserted, Then overwrites oldest."""
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

    def test_oversize_rejected(self, shm_path: str) -> None:
        """Given oversized message, When inserted, Then rejected."""
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
        """Given batch then single insert, When consumed, Then order preserved."""
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
        """Given empty payload, When inserted, Then round-trips correctly."""
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
        """Given message of exactly capacity - 8 bytes, When inserted, Then accepted."""
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
        """Given empty buffer, When peekleft called, Then returns None."""
        prod = ShmSpscProducer(shm_path, 1 << 12, create=True)
        cons = ShmSpscConsumer(shm_path)
        try:
            assert cons.peekleft() is None
        finally:
            cons.close()
            prod.close()

    def test_peekright_empty_returns_none(self, shm_path: str) -> None:
        """Given empty buffer, When peekright called, Then returns None."""
        prod = ShmSpscProducer(shm_path, 1 << 12, create=True)
        cons = ShmSpscConsumer(shm_path)
        try:
            assert cons.peekright() is None
        finally:
            cons.close()
            prod.close()

    def test_insert_batch_empty_list(self, shm_path: str) -> None:
        """Given empty list, When insert_batch called, Then returns True."""
        prod = ShmSpscProducer(shm_path, 1 << 12, create=True)
        try:
            assert prod.insert_batch([])
        finally:
            prod.close()

    def test_idempotent_close(self, shm_path: str) -> None:
        """Given closed producer, When close called again, Then no crash."""
        prod = ShmSpscProducer(shm_path, 1 << 12, create=True, unlink_on_close=True)
        prod.close()
        prod.close()
        assert not os.path.exists(shm_path)

    def test_consume_with_yield(self, shm_path: str) -> None:
        """Given payload, When consume called, Then works correctly."""
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
        """Given corrupted header, When consumer attaches, Then raises RuntimeError."""
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
        """Given producer in main process and consumer in separate process, Then all messages roundtrip."""
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
        """Given insert_char data, When consumed, Then roundtrips correctly."""
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
        """Given existing buffer, When attaching with create=False, Then works."""
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
        """Given producer as context manager, When exited, Then auto-closed."""
        with ShmSpscProducer(
            shm_path, 1 << 12, create=True, unlink_on_close=True
        ) as prod:
            assert prod.insert(b"ctx")
        assert not os.path.exists(shm_path)

    def test_context_manager_consumer(self, shm_path: str) -> None:
        """Given consumer as context manager, When exited, Then auto-closed."""
        prod = ShmSpscProducer(shm_path, 1 << 12, create=True)
        try:
            prod.insert(b"ctx")
            with ShmSpscConsumer(shm_path) as cons:
                assert cons.consume() == b"ctx"
        finally:
            prod.close()

    def test_len_property(self, shm_path: str) -> None:
        """Given various operations, When len checked, Then reflects state."""
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
        """Given populated buffer, When peekleft called, Then returns first item without consuming."""
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
        """Given populated buffer, When peekright called, Then returns last item without consuming."""
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
        """Given empty path, When creating ShmSpscConfig, Then raises ValueError."""
        with pytest.raises(ValueError):
            ShmSpscConfig(path="", capacity_bytes=1024)

    def test_config_validation_zero_capacity(self) -> None:
        """Given zero capacity, When creating ShmSpscConfig, Then raises ValueError."""
        with pytest.raises(ValueError):
            ShmSpscConfig(path="/tmp/x", capacity_bytes=0)

    def test_config_validation_zero_spin_wait(self) -> None:
        """Given spin_wait=0, When creating ShmSpscConfig, Then raises ValueError."""
        with pytest.raises(ValueError):
            ShmSpscConfig(path="/tmp/x", capacity_bytes=1024, spin_wait=0)

    def test_config_validation_unlink_without_create(self) -> None:
        """Given unlink_on_close=True and create=False, When creating ShmSpscConfig, Then raises ValueError."""
        with pytest.raises(ValueError):
            ShmSpscConfig(
                path="/tmp/x",
                capacity_bytes=1024,
                create=False,
                unlink_on_close=True,
            )

    def test_config_default(self) -> None:
        """Given ShmSpscConfig.default(), When called, Then returns valid config."""
        cfg = ShmSpscConfig.default()
        assert cfg.path == "/tmp/shm_ring.bin"
        assert cfg.capacity_bytes == 1 << 16
        assert cfg.create is True
        assert cfg.unlink_on_close is False
        assert cfg.spin_wait == 1024

    def test_config_kwargs(self) -> None:
        """Given config, When producer_kwargs/consumer_kwargs called, Then correct dicts."""
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
        """Given file smaller than 64 bytes, When consumer attaches, Then rejected."""
        with open(shm_path, "wb") as f:
            f.write(b"\x00" * 32)
        with pytest.raises(RuntimeError):
            ShmSpscConsumer(shm_path)

    # --- P1 Important ---

    def test_spin_wait_small(self, shm_path: str) -> None:
        """Given spin_wait=1, When producer/consumer used, Then works correctly."""
        prod = ShmSpscProducer(shm_path, 1 << 12, create=True, spin_wait=1)
        cons = ShmSpscConsumer(shm_path, spin_wait=1)
        try:
            assert prod.insert(b"small")
            assert cons.consume() == b"small"
        finally:
            cons.close()
            prod.close()

    def test_spin_wait_large(self, shm_path: str) -> None:
        """Given spin_wait=65536, When producer/consumer used, Then works correctly."""
        prod = ShmSpscProducer(shm_path, 1 << 12, create=True, spin_wait=65536)
        cons = ShmSpscConsumer(shm_path, spin_wait=65536)
        try:
            assert prod.insert(b"large")
            assert cons.consume() == b"large"
        finally:
            cons.close()
            prod.close()

    def test_mismatched_spin_wait(self, shm_path: str) -> None:
        """Given mismatched spin_wait, When producer/consumer used, Then interoperate."""
        prod = ShmSpscProducer(shm_path, 1 << 12, create=True, spin_wait=100)
        cons = ShmSpscConsumer(shm_path, spin_wait=10000)
        try:
            assert prod.insert(b"mismatch")
            assert cons.consume() == b"mismatch"
        finally:
            cons.close()
            prod.close()

    def test_insert_batch_exceeds_capacity(self, shm_path: str) -> None:
        """Given batch exceeding capacity, When inserted, Then returns False."""
        capacity = 1 << 8
        prod = ShmSpscProducer(shm_path, capacity, create=True)
        try:
            msgs = [b"x" * (capacity - 7)]
            assert not prod.insert_batch(msgs)
        finally:
            prod.close()

    def test_unlink_on_close_false(self, shm_path: str) -> None:
        """Given unlink_on_close=False, When closed, Then file persists."""
        prod = ShmSpscProducer(shm_path, 1 << 12, create=True, unlink_on_close=False)
        prod.close()
        assert os.path.exists(shm_path)
        os.unlink(shm_path)

    def test_timestamp_properties(self, shm_path: str) -> None:
        """Given various operations, When timestamps checked, Then monotonic."""
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
        """Given one thread producing and one consuming 1000 messages, Then all received."""
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
        """Given empty buffer, When consume_all called, Then returns empty list."""
        prod = ShmSpscProducer(shm_path, 1 << 12, create=True)
        cons = ShmSpscConsumer(shm_path)
        try:
            assert cons.consume_all() == []
        finally:
            cons.close()
            prod.close()

    # --- P2 Nice-to-have ---

    def test_invalid_path_oserror(self) -> None:
        """Given invalid path, When creating producer, Then raises OSError."""
        with pytest.raises(OSError):
            ShmSpscProducer("/nonexistent/dir/file", 1 << 12, create=True)

    def test_capacity_bytes_zero_handled(self, shm_path: str) -> None:
        """Given capacity_bytes=0, When creating producer, Then uses pow2_at_least(1)."""
        prod = ShmSpscProducer(shm_path, 0, create=True, unlink_on_close=False)
        try:
            prod.close()
            # Header (64 bytes) + 1 byte capacity
            assert os.path.getsize(shm_path) == 65
        finally:
            if os.path.exists(shm_path):
                os.unlink(shm_path)

    def test_consume_iterable(self, shm_path: str) -> None:
        """Given populated buffer, When consume_iterable called, Then yields FIFO."""
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
        """Given async consumer, When data available, Then returns asynchronously."""
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
        """Given async iterable consumer, When data available, Then yields asynchronously."""
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

    def test_unwrapped_matches_consume_all(self, shm_path: str) -> None:
        """Given populated buffer, When unwrapped and consume_all called, Then same results."""
        prod = ShmSpscProducer(shm_path, 1 << 12, create=True)
        cons = ShmSpscConsumer(shm_path)
        try:
            msgs = [b"a", b"b", b"c"]
            prod.insert_batch(msgs)
            unw = cons.unwrapped()
            assert unw == msgs
            assert len(cons) == 3
            assert cons.consume_all() == msgs
        finally:
            cons.close()
            prod.close()

    def test_unwrapped_empty(self, shm_path: str) -> None:
        """Given empty buffer, When unwrapped called, Then returns empty list."""
        prod = ShmSpscProducer(shm_path, 1 << 12, create=True)
        cons = ShmSpscConsumer(shm_path)
        try:
            assert cons.unwrapped() == []
        finally:
            cons.close()
            prod.close()

    def test_contains(self, shm_path: str) -> None:
        """Given populated buffer, When contains checked, Then correct results."""
        prod = ShmSpscProducer(shm_path, 1 << 12, create=True)
        cons = ShmSpscConsumer(shm_path)
        try:
            prod.insert(b"hello")
            assert cons.contains(b"hello")
            assert b"hello" in cons
            assert not cons.contains(b"missing")
            assert b"missing" not in cons
        finally:
            cons.close()
            prod.close()

    def test_consumer_is_empty(self, shm_path: str) -> None:
        """Given various states, When is_empty checked, Then reflects buffer state."""
        prod = ShmSpscProducer(shm_path, 1 << 12, create=True)
        cons = ShmSpscConsumer(shm_path)
        try:
            assert cons.is_empty()
            prod.insert(b"x")
            assert not cons.is_empty()
            cons.consume()
            assert cons.is_empty()
        finally:
            cons.close()
            prod.close()

    def test_consumer_is_full(self, shm_path: str) -> None:
        """Given full buffer, When is_full checked, Then returns True."""
        capacity = 1 << 8
        prod = ShmSpscProducer(shm_path, capacity, create=True)
        cons = ShmSpscConsumer(shm_path)
        try:
            assert not cons.is_full()
            # Fill buffer with messages that leave less than 8 bytes free
            msg = b"x" * (capacity // 4 - 8)
            for _ in range(8):
                prod.insert(msg)
            assert cons.is_full()
        finally:
            cons.close()
            prod.close()

    def test_consumer_clear(self, shm_path: str) -> None:
        """Given populated buffer, When clear called, Then emptied."""
        prod = ShmSpscProducer(shm_path, 1 << 12, create=True)
        cons = ShmSpscConsumer(shm_path)
        try:
            prod.insert_batch([b"a", b"b", b"c"])
            cons.clear()
            assert cons.is_empty()
            assert len(cons) == 0
        finally:
            cons.close()
            prod.close()

    def test_producer_is_empty(self, shm_path: str) -> None:
        """Given various states, When producer is_empty checked, Then reflects buffer state."""
        prod = ShmSpscProducer(shm_path, 1 << 12, create=True)
        cons = ShmSpscConsumer(shm_path)
        try:
            assert prod.is_empty()
            prod.insert(b"x")
            assert not prod.is_empty()
            cons.consume()
            assert prod.is_empty()
        finally:
            cons.close()
            prod.close()

    def test_producer_is_full(self, shm_path: str) -> None:
        """Given full buffer, When producer is_full checked, Then returns True."""
        capacity = 1 << 8
        prod = ShmSpscProducer(shm_path, capacity, create=True)
        try:
            assert not prod.is_full()
            msg = b"x" * (capacity // 4 - 8)
            for _ in range(8):
                prod.insert(msg)
            assert prod.is_full()
        finally:
            prod.close()
