"""MPSC shared-memory ring buffer tests.

Exercises multi-producer single-consumer behavior, batch semantics, and
header validation for the sharded sub-ring architecture.
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
    ShmMpscConsumer,
    ShmMpscProducer,
    ShmMpscConfig,
)

pytestmark = pytest.mark.skipif(
    os.name != "posix",
    reason="Shared memory ringbuffer requires POSIX support",
)


@pytest.fixture()
def shm_path(tmp_path: Path) -> str:
    """Build a temporary backing file path for MPSC SHM ringbuffer tests.

    Args:
        tmp_path: pytest-provided temporary directory path.

    Returns:
        String path to the backing file.
    """
    return str(tmp_path / "shm_mpsc_ring.bin")


def _mpsc_producer_proc(
    path: str, capacity: int, num_rings: int, msgs: list[bytes]
) -> None:
    """Producer helper for multiprocess tests.

    Args:
        path: Filesystem path to the shared memory ringbuffer file.
        capacity: Capacity used at creation time.
        num_rings: Number of sub-rings used at creation time.
        msgs: List of byte messages to insert.
    """
    prod = ShmMpscProducer(path, capacity, num_rings=num_rings, create=False)
    try:
        for msg in msgs:
            prod.insert(msg)
    finally:
        prod.close()


def _mpsc_consumer_proc(path: str, n: int, q: mp.Queue) -> None:
    """Consumer helper for multiprocess tests.

    Args:
        path: Filesystem path to the shared memory ringbuffer file.
        n: Number of messages to consume.
        q: Multiprocessing queue used to return results.
    """
    cons = ShmMpscConsumer(path, spin_wait=4096)
    try:
        got: list[bytes] = []
        for _ in range(n):
            got.append(cons.consume())
        checksum = sum(len(x) + (x[0] if x else 0) for x in got)
        q.put((len(got), checksum))
    finally:
        cons.close()


class TestMpscSharedBytesRingBuffer:
    """Tests for MpscSharedBytesRingBuffer implementation."""

    # --- Basic functionality ---

    def test_basic_send_receive(self, shm_path: str) -> None:
        """Send one payload and confirm it round-trips correctly."""
        prod = ShmMpscProducer(
            shm_path, 1 << 16, num_rings=4, create=True, unlink_on_close=True
        )
        cons = ShmMpscConsumer(shm_path)
        try:
            payload = b"hello-world"
            assert prod.insert(payload)
            got = cons.consume()
            assert got == payload
        finally:
            cons.close()
            prod.close()
            assert not os.path.exists(shm_path)

    def test_multiple_producers_single_consumer(self, shm_path: str) -> None:
        """Two producers insert; single consumer receives both messages."""
        prod1 = ShmMpscProducer(
            shm_path, 1 << 16, num_rings=4, create=True, unlink_on_close=False
        )
        prod2 = ShmMpscProducer(shm_path, 1 << 16, num_rings=4, create=False)
        cons = ShmMpscConsumer(shm_path)
        try:
            assert prod1.insert(b"from-prod1")
            assert prod2.insert(b"from-prod2")
            msgs = {cons.consume(), cons.consume()}
            assert msgs == {b"from-prod1", b"from-prod2"}
        finally:
            cons.close()
            prod1.close()
            prod2.close()
            if os.path.exists(shm_path):
                os.unlink(shm_path)

    def test_message_ordering_within_subring(self, shm_path: str) -> None:
        """With one ring, ordering is preserved like SPSC."""
        prod = ShmMpscProducer(
            shm_path, 1 << 14, num_rings=1, create=True, unlink_on_close=True
        )
        cons = ShmMpscConsumer(shm_path)
        try:
            msgs = [f"m{i}".encode() for i in range(100)]
            for m in msgs:
                assert prod.insert(m)
            got = cons.consume_all()
            assert got == msgs
        finally:
            cons.close()
            prod.close()

    def test_batch_and_drain(self, shm_path: str) -> None:
        """Insert a batch and drain in order on a single sub-ring."""
        prod = ShmMpscProducer(
            shm_path, 1 << 15, num_rings=1, create=True, unlink_on_close=True
        )
        cons = ShmMpscConsumer(shm_path)
        try:
            msgs = [f"m{i}".encode() for i in range(1000)]
            assert prod.insert_batch(msgs)
            got = cons.consume_all()
            assert got == msgs
        finally:
            cons.close()
            prod.close()

    # --- Edge cases ---

    def test_empty_payload_insert(self, shm_path: str) -> None:
        """Empty payload should insert and consume correctly."""
        prod = ShmMpscProducer(
            shm_path, 1 << 12, num_rings=2, create=True, unlink_on_close=True
        )
        cons = ShmMpscConsumer(shm_path)
        try:
            assert prod.insert(b"")
            got = cons.consume()
            assert got == b""
        finally:
            cons.close()
            prod.close()

    def test_exact_capacity_message(self, shm_path: str) -> None:
        """Insert a message of exactly capacity - 8 bytes (the max)."""
        capacity = 1 << 12
        prod = ShmMpscProducer(
            shm_path, capacity, num_rings=1, create=True, unlink_on_close=True
        )
        cons = ShmMpscConsumer(shm_path)
        try:
            msg = b"x" * (capacity - 8)
            assert prod.insert(msg)
            got = cons.consume()
            assert got == msg
        finally:
            cons.close()
            prod.close()

    def test_oversize_rejected(self, shm_path: str) -> None:
        """Reject inserts that exceed capacity."""
        capacity = 1 << 12
        prod = ShmMpscProducer(
            shm_path, capacity, num_rings=1, create=True, unlink_on_close=True
        )
        try:
            oversize = b"z" * (capacity - 7)
            assert not prod.insert(oversize)
        finally:
            prod.close()

    def test_insert_overwrites_oldest(self, shm_path: str) -> None:
        """Ensure overwrites drop oldest items when capacity is exceeded."""
        capacity = 1 << 12
        prod = ShmMpscProducer(
            shm_path, capacity, num_rings=1, create=True, unlink_on_close=True
        )
        cons = ShmMpscConsumer(shm_path)
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

    def test_insert_batch_empty_list(self, shm_path: str) -> None:
        """insert_batch with empty list returns True."""
        prod = ShmMpscProducer(
            shm_path, 1 << 12, num_rings=2, create=True, unlink_on_close=True
        )
        try:
            assert prod.insert_batch([])
        finally:
            prod.close()

    def test_consume_all_empty(self, shm_path: str) -> None:
        """consume_all on empty buffer returns empty list."""
        prod = ShmMpscProducer(
            shm_path, 1 << 12, num_rings=2, create=True, unlink_on_close=True
        )
        cons = ShmMpscConsumer(shm_path)
        try:
            assert cons.consume_all() == []
        finally:
            cons.close()
            prod.close()

    def test_peekleft_empty_returns_none(self, shm_path: str) -> None:
        """peekleft on empty buffer returns None."""
        prod = ShmMpscProducer(
            shm_path, 1 << 12, num_rings=2, create=True, unlink_on_close=True
        )
        cons = ShmMpscConsumer(shm_path)
        try:
            assert cons.peekleft() is None
        finally:
            cons.close()
            prod.close()

    def test_peekright_empty_returns_none(self, shm_path: str) -> None:
        """peekright on empty buffer returns None."""
        prod = ShmMpscProducer(
            shm_path, 1 << 12, num_rings=2, create=True, unlink_on_close=True
        )
        cons = ShmMpscConsumer(shm_path)
        try:
            assert cons.peekright() is None
        finally:
            cons.close()
            prod.close()

    def test_peekleft_nonempty(self, shm_path: str) -> None:
        """peekleft returns an item without consuming it.

        Uses a single ring so ordering is deterministic.
        """
        prod = ShmMpscProducer(
            shm_path, 1 << 12, num_rings=1, create=True, unlink_on_close=True
        )
        cons = ShmMpscConsumer(shm_path)
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

        Uses a single ring so ordering is deterministic.
        """
        prod = ShmMpscProducer(
            shm_path, 1 << 12, num_rings=1, create=True, unlink_on_close=True
        )
        cons = ShmMpscConsumer(shm_path)
        try:
            assert prod.insert(b"first")
            assert prod.insert(b"second")
            assert cons.peekright() == b"second"
            assert cons.consume() == b"first"
            assert cons.consume() == b"second"
        finally:
            cons.close()
            prod.close()

    # --- Config validation ---

    def test_config_validation_empty_path(self) -> None:
        """ShmMpscConfig with empty path raises ValueError."""
        with pytest.raises(ValueError):
            ShmMpscConfig(path="", capacity_bytes=1024)

    def test_config_validation_zero_capacity(self) -> None:
        """ShmMpscConfig with zero capacity raises ValueError."""
        with pytest.raises(ValueError):
            ShmMpscConfig(path="/tmp/x", capacity_bytes=0)

    def test_config_validation_negative_num_rings(self) -> None:
        """ShmMpscConfig with negative num_rings raises ValueError."""
        with pytest.raises(ValueError):
            ShmMpscConfig(path="/tmp/x", capacity_bytes=1024, num_rings=-1)

    def test_config_validation_zero_spin_wait(self) -> None:
        """ShmMpscConfig with spin_wait=0 raises ValueError."""
        with pytest.raises(ValueError):
            ShmMpscConfig(path="/tmp/x", capacity_bytes=1024, spin_wait=0)

    def test_config_validation_unlink_without_create(self) -> None:
        """ShmMpscConfig with unlink_on_close=True and create=False raises ValueError."""
        with pytest.raises(ValueError):
            ShmMpscConfig(
                path="/tmp/x",
                capacity_bytes=1024,
                create=False,
                unlink_on_close=True,
            )

    def test_config_default(self) -> None:
        """ShmMpscConfig.default() returns valid config with expected defaults."""
        cfg = ShmMpscConfig.default()
        assert cfg.path == "/tmp/shm_mpsc_ring.bin"
        assert cfg.capacity_bytes == 1 << 20
        assert cfg.num_rings == 0
        assert cfg.create is True
        assert cfg.unlink_on_close is False
        assert cfg.spin_wait == 1024

    def test_config_kwargs(self) -> None:
        """producer_kwargs and consumer_kwargs return correct dicts."""
        cfg = ShmMpscConfig(
            path="/tmp/x", capacity_bytes=2048, num_rings=2, spin_wait=512
        )
        pkw = cfg.producer_kwargs()
        assert pkw["path"] == "/tmp/x"
        assert pkw["capacity_bytes"] == 2048
        assert pkw["num_rings"] == 2
        assert pkw["spin_wait"] == 512
        ckw = cfg.consumer_kwargs()
        assert ckw["path"] == "/tmp/x"
        assert ckw["spin_wait"] == 512
        assert "capacity_bytes" not in ckw

    def test_num_rings_auto_detects(self, shm_path: str) -> None:
        """num_rings=0 auto-detects CPU count."""
        prod = ShmMpscProducer(
            shm_path, 1 << 16, num_rings=0, create=True, unlink_on_close=True
        )
        try:
            import os as _os

            expected = _os.cpu_count() or 4
            assert prod.num_rings == expected
        finally:
            prod.close()

    def test_num_rings_one(self, shm_path: str) -> None:
        """num_rings=1 degrades to SPSC-like behavior."""
        prod = ShmMpscProducer(
            shm_path, 1 << 16, num_rings=1, create=True, unlink_on_close=True
        )
        cons = ShmMpscConsumer(shm_path)
        try:
            assert prod.insert(b"solo")
            assert cons.consume() == b"solo"
        finally:
            cons.close()
            prod.close()

    # --- Resource management ---

    def test_idempotent_close(self, shm_path: str) -> None:
        """Calling close twice should not crash."""
        prod = ShmMpscProducer(
            shm_path, 1 << 12, num_rings=2, create=True, unlink_on_close=True
        )
        prod.close()
        prod.close()
        assert not os.path.exists(shm_path)

    def test_context_manager_producer(self, shm_path: str) -> None:
        """Use producer as a context manager and verify auto-close."""
        with ShmMpscProducer(
            shm_path, 1 << 12, num_rings=2, create=True, unlink_on_close=True
        ) as prod:
            assert prod.insert(b"ctx")
        assert not os.path.exists(shm_path)

    def test_context_manager_consumer(self, shm_path: str) -> None:
        """Use consumer as a context manager and verify auto-close."""
        prod = ShmMpscProducer(shm_path, 1 << 12, num_rings=2, create=True)
        try:
            prod.insert(b"ctx")
            with ShmMpscConsumer(shm_path) as cons:
                assert cons.consume() == b"ctx"
        finally:
            prod.close()

    def test_len_property(self, shm_path: str) -> None:
        """Verify len(producer) behavior across inserts, consumes, and overwrites."""
        capacity = 1 << 12
        prod = ShmMpscProducer(
            shm_path, capacity, num_rings=1, create=True, unlink_on_close=True
        )
        cons = ShmMpscConsumer(shm_path)
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
        finally:
            cons.close()
            prod.close()

    def test_unlink_on_close_false(self, shm_path: str) -> None:
        """Close producer with unlink_on_close=False; file still exists."""
        prod = ShmMpscProducer(
            shm_path, 1 << 12, num_rings=2, create=True, unlink_on_close=False
        )
        prod.close()
        assert os.path.exists(shm_path)
        os.unlink(shm_path)

    # --- Error handling ---

    def test_invalid_path_oserror(self) -> None:
        """Invalid path raises OSError."""
        with pytest.raises(OSError):
            ShmMpscProducer("/nonexistent/dir/file", 1 << 12, create=True)

    def test_truncated_file_rejected(self, shm_path: str) -> None:
        """File smaller than 64 bytes is rejected on consumer attach."""
        with open(shm_path, "wb") as f:
            f.write(b"\x00" * 32)
        with pytest.raises(RuntimeError):
            ShmMpscConsumer(shm_path)

    def test_corrupted_header_rejected(self, shm_path: str) -> None:
        """Reject attaching to a ringbuffer with a corrupted header."""
        prod = ShmMpscProducer(
            shm_path, 1 << 12, num_rings=2, create=True, unlink_on_close=False
        )
        prod.close()
        try:
            with open(shm_path, "r+b") as f:
                f.seek(8)
                f.write(struct.pack("Q", 0))
            with pytest.raises(RuntimeError):
                ShmMpscConsumer(shm_path)
        finally:
            if os.path.exists(shm_path):
                os.unlink(shm_path)

    def test_create_false_attach(self, shm_path: str) -> None:
        """Create with create=True, then attach a second producer with create=False."""
        prod1 = ShmMpscProducer(shm_path, 1 << 12, num_rings=2, create=True)
        try:
            prod2 = ShmMpscProducer(shm_path, 1 << 12, num_rings=2, create=False)
            try:
                assert prod2.insert(b"attached")
                cons = ShmMpscConsumer(shm_path)
                try:
                    assert cons.consume() == b"attached"
                finally:
                    cons.close()
            finally:
                prod2.close()
        finally:
            prod1.close()
            if os.path.exists(shm_path):
                os.unlink(shm_path)

    def test_capacity_bytes_zero_handled(self, shm_path: str) -> None:
        """capacity_bytes=0 uses pow2_at_least(1) and does not crash."""
        prod = ShmMpscProducer(
            shm_path, 0, num_rings=1, create=True, unlink_on_close=False
        )
        try:
            prod.close()
            # Global header (64) + sub header (64) + 1 byte data, aligned stride = 128
            # total = 64 + 128 = 192
            assert os.path.getsize(shm_path) == 192
        finally:
            if os.path.exists(shm_path):
                os.unlink(shm_path)

    # --- Timestamp properties ---

    def test_timestamp_properties(self, shm_path: str) -> None:
        """Timestamps are non-zero and monotonically increase."""
        prod = ShmMpscProducer(
            shm_path, 1 << 12, num_rings=2, create=True, unlink_on_close=True
        )
        cons = ShmMpscConsumer(shm_path)
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

    # --- Multi-producer scenarios ---

    def test_two_producers_in_threads(self, shm_path: str) -> None:
        """Two threads producing via separate instances, one thread consuming."""
        n = 500
        prod1 = ShmMpscProducer(
            shm_path, 1 << 18, num_rings=4, create=True, unlink_on_close=False
        )
        prod2 = ShmMpscProducer(shm_path, 1 << 18, num_rings=4, create=False)
        cons = ShmMpscConsumer(shm_path)
        received: list[bytes] = []

        def p1() -> None:
            for i in range(n):
                prod1.insert(f"p1-{i}".encode())

        def p2() -> None:
            for i in range(n):
                prod2.insert(f"p2-{i}".encode())

        def c() -> None:
            for _ in range(2 * n):
                received.append(cons.consume())

        tc = threading.Thread(target=c)
        tc.start()
        t1 = threading.Thread(target=p1)
        t2 = threading.Thread(target=p2)
        t1.start()
        t2.start()
        t1.join(timeout=30)
        t2.join(timeout=30)
        tc.join(timeout=30)
        assert len(received) == 2 * n
        p1_msgs = [m for m in received if m.startswith(b"p1-")]
        p2_msgs = [m for m in received if m.startswith(b"p2-")]
        assert len(p1_msgs) == n
        assert len(p2_msgs) == n
        cons.close()
        prod1.close()
        prod2.close()
        if os.path.exists(shm_path):
            os.unlink(shm_path)

    def test_different_message_sizes(self, shm_path: str) -> None:
        """Producers with different message sizes."""
        prod = ShmMpscProducer(
            shm_path, 1 << 16, num_rings=4, create=True, unlink_on_close=True
        )
        cons = ShmMpscConsumer(shm_path)
        try:
            msgs = [b"a", b"b" * 10, b"c" * 100, b"d" * 1000]
            for m in msgs:
                assert prod.insert(m)
            got = cons.consume_all()
            assert set(got) == set(msgs)
        finally:
            cons.close()
            prod.close()

    def test_four_producers_multiprocess(self, shm_path: str) -> None:
        """Four producers in separate processes, consumer in main process."""
        n_per_prod = 500
        num_prods = 4
        total = n_per_prod * num_prods
        prod_init = ShmMpscProducer(shm_path, 1 << 20, num_rings=4, create=True)
        prod_init.close()

        procs = []
        for pid in range(num_prods):
            msgs = [f"p{pid}-{i}".encode() for i in range(n_per_prod)]
            p = mp.Process(
                target=_mpsc_producer_proc,
                args=(shm_path, 1 << 20, 4, msgs),
            )
            p.start()
            procs.append(p)

        cons = ShmMpscConsumer(shm_path, spin_wait=4096)
        try:
            received: list[bytes] = []
            for _ in range(total):
                received.append(cons.consume())
            assert len(received) == total
        finally:
            cons.close()

        for p in procs:
            p.join(timeout=30)
            assert p.exitcode == 0

        if os.path.exists(shm_path):
            os.unlink(shm_path)

    def test_eight_producers_stress(self, shm_path: str) -> None:
        """Eight producers in separate processes (stress test)."""
        n_per_prod = 1000
        num_prods = 8
        total = n_per_prod * num_prods
        cons = None
        procs = []
        try:
            prod_init = ShmMpscProducer(shm_path, 1 << 22, num_rings=8, create=True)
            prod_init.close()

            rng = random.Random(1337)
            for pid in range(num_prods):
                msgs = [
                    bytes(rng.randrange(0, 256) for _ in range(rng.randint(1, 128)))
                    for _ in range(n_per_prod)
                ]
                p = mp.Process(
                    target=_mpsc_producer_proc,
                    args=(shm_path, 1 << 22, 8, msgs),
                )
                p.start()
                procs.append(p)

            cons = ShmMpscConsumer(shm_path, spin_wait=4096)
            received = 0
            for _ in range(total):
                cons.consume()
                received += 1
            assert received == total

            for p in procs:
                p.join(timeout=60)
                assert p.exitcode == 0
        finally:
            if cons is not None:
                cons.close()
            for p in procs:
                if p.is_alive():
                    p.join(timeout=5)
            if os.path.exists(shm_path):
                os.unlink(shm_path)

    # --- Multi-process scenarios ---

    def test_multiprocess_consumer_producers_main(self, shm_path: str) -> None:
        """Consumer in separate process, producers in main process."""
        n = 1000
        prod = ShmMpscProducer(shm_path, 1 << 20, num_rings=4, create=True)
        try:
            q: mp.Queue = mp.Queue()
            p = mp.Process(target=_mpsc_consumer_proc, args=(shm_path, n, q))
            p.start()

            for i in range(n):
                prod.insert(str(i).encode())

            p.join(timeout=30)
            assert p.exitcode == 0
            got_n, checksum = q.get(timeout=2)
            assert got_n == n
            assert isinstance(checksum, int) and checksum >= 0
        finally:
            prod.close()
            if os.path.exists(shm_path):
                os.unlink(shm_path)

    def test_all_separate_processes(self, shm_path: str) -> None:
        """All producers and consumer in separate processes."""
        n_per_prod = 250
        num_prods = 4
        total = n_per_prod * num_prods
        prod_init = ShmMpscProducer(shm_path, 1 << 20, num_rings=4, create=True)
        prod_init.close()

        procs = []
        for pid in range(num_prods):
            msgs = [f"p{pid}-{i}".encode() for i in range(n_per_prod)]
            p = mp.Process(
                target=_mpsc_producer_proc,
                args=(shm_path, 1 << 20, 4, msgs),
            )
            p.start()
            procs.append(p)

        q: mp.Queue = mp.Queue()
        cons_p = mp.Process(target=_mpsc_consumer_proc, args=(shm_path, total, q))
        cons_p.start()

        for p in procs:
            p.join(timeout=30)
            assert p.exitcode == 0

        cons_p.join(timeout=30)
        assert cons_p.exitcode == 0
        got_n, checksum = q.get(timeout=2)
        assert got_n == total
        assert isinstance(checksum, int) and checksum >= 0

        if os.path.exists(shm_path):
            os.unlink(shm_path)

    # --- Performance / stress ---

    def test_high_throughput_100k(self, shm_path: str) -> None:
        """High-throughput test: 100K messages across 4 producers."""
        n = 100_000
        num_prods = 4
        prod_init = ShmMpscProducer(shm_path, 1 << 22, num_rings=4, create=True)
        prod_init.close()

        procs = []
        for pid in range(num_prods):
            msgs = [f"p{pid}-{i}".encode() for i in range(n // num_prods)]
            p = mp.Process(
                target=_mpsc_producer_proc,
                args=(shm_path, 1 << 22, 4, msgs),
            )
            p.start()
            procs.append(p)

        cons = ShmMpscConsumer(shm_path, spin_wait=4096)
        try:
            received = 0
            for _ in range(n):
                cons.consume()
                received += 1
            assert received == n
        finally:
            cons.close()

        for p in procs:
            p.join(timeout=60)
            assert p.exitcode == 0

        if os.path.exists(shm_path):
            os.unlink(shm_path)

    def test_large_messages_16kb(self, shm_path: str) -> None:
        """Large message test: 16KB messages."""
        capacity = 1 << 16
        prod = ShmMpscProducer(
            shm_path, capacity, num_rings=2, create=True, unlink_on_close=True
        )
        cons = ShmMpscConsumer(shm_path)
        try:
            msg = b"x" * (16 * 1024)
            assert prod.insert(msg)
            got = cons.consume()
            assert got == msg
        finally:
            cons.close()
            prod.close()

    def test_insert_batch_exceeds_capacity(self, shm_path: str) -> None:
        """Batch whose total size exceeds capacity returns False."""
        capacity = 1 << 8
        prod = ShmMpscProducer(
            shm_path, capacity, num_rings=1, create=True, unlink_on_close=True
        )
        try:
            msgs = [b"x" * (capacity - 7)]
            assert not prod.insert_batch(msgs)
        finally:
            prod.close()

    def test_spin_wait_small(self, shm_path: str) -> None:
        """Producer and consumer with spin_wait=1 work correctly."""
        prod = ShmMpscProducer(
            shm_path,
            1 << 12,
            num_rings=2,
            create=True,
            spin_wait=1,
            unlink_on_close=True,
        )
        cons = ShmMpscConsumer(shm_path, spin_wait=1)
        try:
            assert prod.insert(b"small")
            assert cons.consume() == b"small"
        finally:
            cons.close()
            prod.close()

    def test_spin_wait_large(self, shm_path: str) -> None:
        """Producer and consumer with spin_wait=65536 work correctly."""
        prod = ShmMpscProducer(
            shm_path,
            1 << 12,
            num_rings=2,
            create=True,
            spin_wait=65536,
            unlink_on_close=True,
        )
        cons = ShmMpscConsumer(shm_path, spin_wait=65536)
        try:
            assert prod.insert(b"large")
            assert cons.consume() == b"large"
        finally:
            cons.close()
            prod.close()

    def test_mismatched_spin_wait(self, shm_path: str) -> None:
        """Producer and consumer with different spin_wait values interoperate."""
        prod = ShmMpscProducer(
            shm_path,
            1 << 12,
            num_rings=2,
            create=True,
            spin_wait=100,
            unlink_on_close=True,
        )
        cons = ShmMpscConsumer(shm_path, spin_wait=10000)
        try:
            assert prod.insert(b"mismatch")
            assert cons.consume() == b"mismatch"
        finally:
            cons.close()
            prod.close()

    def test_insert_char_roundtrip(self, shm_path: str) -> None:
        """insert_char(b'hello', 5) should round-trip through consume()."""
        prod = ShmMpscProducer(
            shm_path, 1 << 12, num_rings=2, create=True, unlink_on_close=True
        )
        cons = ShmMpscConsumer(shm_path)
        try:
            assert prod.insert_char(b"hello", 5)
            got = cons.consume()
            assert got == b"hello"
        finally:
            cons.close()
            prod.close()

    def test_consume_iterable(self, shm_path: str) -> None:
        """consume_iterable yields items in FIFO order."""
        prod = ShmMpscProducer(
            shm_path, 1 << 12, num_rings=2, create=True, unlink_on_close=True
        )
        cons = ShmMpscConsumer(shm_path)
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
        """aconsume returns items asynchronously."""
        prod = ShmMpscProducer(
            shm_path, 1 << 12, num_rings=2, create=True, unlink_on_close=True
        )
        cons = ShmMpscConsumer(shm_path)
        try:
            prod.insert(b"async")
            result = await cons.aconsume()
            assert result == b"async"
        finally:
            cons.close()
            prod.close()

    @pytest.mark.asyncio
    async def test_aconsume_iterable(self, shm_path: str) -> None:
        """aconsume_iterable yields items asynchronously."""
        prod = ShmMpscProducer(
            shm_path, 1 << 12, num_rings=2, create=True, unlink_on_close=True
        )
        cons = ShmMpscConsumer(shm_path)
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
        """unwrapped() returns contents without consuming."""
        prod = ShmMpscProducer(
            shm_path, 1 << 14, num_rings=1, create=True, unlink_on_close=True
        )
        cons = ShmMpscConsumer(shm_path)
        try:
            msgs = [b"a", b"b", b"c"]
            prod.insert_batch(msgs)
            unw = cons.unwrapped()
            # With a single ring, order should match
            assert unw == msgs
            assert len(cons) == 3
            assert cons.consume_all() == msgs
        finally:
            cons.close()
            prod.close()

    def test_unwrapped_empty(self, shm_path: str) -> None:
        """unwrapped() on empty buffer returns empty list."""
        prod = ShmMpscProducer(
            shm_path, 1 << 12, num_rings=2, create=True, unlink_on_close=True
        )
        cons = ShmMpscConsumer(shm_path)
        try:
            assert cons.unwrapped() == []
        finally:
            cons.close()
            prod.close()

    def test_contains(self, shm_path: str) -> None:
        """contains() and __contains__() work correctly."""
        prod = ShmMpscProducer(
            shm_path, 1 << 12, num_rings=1, create=True, unlink_on_close=True
        )
        cons = ShmMpscConsumer(shm_path)
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
        """Consumer is_empty() reflects buffer state."""
        prod = ShmMpscProducer(
            shm_path, 1 << 12, num_rings=2, create=True, unlink_on_close=True
        )
        cons = ShmMpscConsumer(shm_path)
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
        """Consumer is_full() reflects when no space remains."""
        capacity = 1 << 8
        prod = ShmMpscProducer(
            shm_path, capacity, num_rings=1, create=True, unlink_on_close=True
        )
        cons = ShmMpscConsumer(shm_path)
        try:
            assert not cons.is_full()
            msg = b"x" * (capacity // 4 - 8)
            for _ in range(8):
                prod.insert(msg)
            assert cons.is_full()
        finally:
            cons.close()
            prod.close()

    def test_consumer_clear(self, shm_path: str) -> None:
        """clear() drains the buffer."""
        prod = ShmMpscProducer(
            shm_path, 1 << 12, num_rings=2, create=True, unlink_on_close=True
        )
        cons = ShmMpscConsumer(shm_path)
        try:
            prod.insert_batch([b"a", b"b", b"c"])
            cons.clear()
            assert cons.is_empty()
            assert len(cons) == 0
        finally:
            cons.close()
            prod.close()

    def test_producer_is_empty(self, shm_path: str) -> None:
        """Producer is_empty() reflects buffer state."""
        prod = ShmMpscProducer(
            shm_path, 1 << 12, num_rings=2, create=True, unlink_on_close=True
        )
        cons = ShmMpscConsumer(shm_path)
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
        """Producer is_full() reflects when no space remains."""
        capacity = 1 << 8
        prod = ShmMpscProducer(
            shm_path, capacity, num_rings=1, create=True, unlink_on_close=True
        )
        try:
            assert not prod.is_full()
            msg = b"x" * (capacity // 4 - 8)
            for _ in range(8):
                prod.insert(msg)
            assert prod.is_full()
        finally:
            prod.close()
