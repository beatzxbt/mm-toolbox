"""Shared-memory ring buffer memory utility tests.

Layer 1 and 2 tests: exercises wrap-around paths in write_u64_le, read_u64_le,
copy_into_ring, and copy_from_ring by constructing small ring buffers that force
data to cross the capacity boundary.
"""

from __future__ import annotations

import os

import pytest

from mm_toolbox.ringbuffer.shm import (
    ShmSpscConsumer,
    ShmSpscProducer,
)

pytestmark = pytest.mark.skipif(
    os.name != "posix",
    reason="Shared memory ringbuffer requires POSIX support",
)


class TestMemoryWraparoundPaths:
    """Layer 1 — Wrap-around paths for memory.pyx helpers.

    Uses tiny SPSC ring buffers to force headers and payloads to cross the
    ring boundary, exercising the wrap branches in write_u64_le, read_u64_le,
    copy_into_ring, and copy_from_ring.
    """

    def test_write_u64_le_wraparound(self, tmp_path):
        """Given write_pos near boundary, When header written, Then wrap path hit.

        With capacity=16, a 6-byte message occupies 14 bytes (write_pos 0→14).
        The next 1-byte message needs 9 bytes; the oldest message is dropped
        to make room.  The new header starts at ring position 14; 14+8=22 >
        16 so write_u64_le takes the byte-by-byte wrap branch.
        """
        path = str(tmp_path / "wrap.bin")
        prod = ShmSpscProducer(path, 16, create=True, unlink_on_close=True)
        cons = ShmSpscConsumer(path)
        try:
            assert prod.insert(b"123456")  # 14 bytes total
            assert prod.insert(b"x")  # 9 bytes total, header wraps
            # First message was dropped to make room; only second remains
            assert cons.consume() == b"x"
        finally:
            cons.close()
            prod.close()

    def test_read_u64_le_wraparound(self, tmp_path):
        """Given read_pos near boundary, When header read, Then wrap path hit.

        Same setup as write_u64_le_wraparound; consuming the remaining message
        reads a header starting at ring position 14, triggering read_u64_le
        wrap.
        """
        path = str(tmp_path / "wrap.bin")
        prod = ShmSpscProducer(path, 16, create=True, unlink_on_close=True)
        cons = ShmSpscConsumer(path)
        try:
            assert prod.insert(b"123456")
            assert prod.insert(b"x")
            assert cons.consume() == b"x"
        finally:
            cons.close()
            prod.close()

    def test_copy_into_ring_wraparound(self, tmp_path):
        """Given payload crosses boundary, When copied in, Then wrap path hit.

        With capacity=32, a 15-byte message occupies 23 bytes (write_pos 0→23).
        The next 8-byte message needs 16 bytes; the oldest is dropped.  The
        new payload starts at ring position 31; endspace = 32-31 = 1, payload
        = 8 > 1, so copy_into_ring wraps.  The header starts at position 23 and
        23+8=31 ≤ 32, so header does not wrap, isolating the payload wrap path.
        """
        path = str(tmp_path / "wrap.bin")
        prod = ShmSpscProducer(path, 32, create=True, unlink_on_close=True)
        cons = ShmSpscConsumer(path)
        try:
            assert prod.insert(b"123456789012345")  # 15 bytes, 23 total
            assert prod.insert(b"12345678")  # 8 bytes, payload wraps
            assert cons.consume() == b"12345678"
        finally:
            cons.close()
            prod.close()

    def test_copy_from_ring_wraparound(self, tmp_path):
        """Given payload crosses boundary, When copied out, Then wrap path hit.

        Same as copy_into_ring_wraparound; consuming the remaining message
        reads an 8-byte payload from ring position 31, triggering
        copy_from_ring wrap.
        """
        path = str(tmp_path / "wrap.bin")
        prod = ShmSpscProducer(path, 32, create=True, unlink_on_close=True)
        cons = ShmSpscConsumer(path)
        try:
            assert prod.insert(b"123456789012345")
            assert prod.insert(b"12345678")
            assert cons.consume() == b"12345678"
        finally:
            cons.close()
            prod.close()

    def test_multiple_wraparounds(self, tmp_path):
        """Given many messages, When buffer wraps repeatedly, Then data intact.

        Fills and drains a tiny buffer many times to stress repeated wrap
        behaviour in all four memory helpers.
        """
        path = str(tmp_path / "wrap.bin")
        prod = ShmSpscProducer(path, 16, create=True, unlink_on_close=True)
        cons = ShmSpscConsumer(path)
        try:
            for i in range(20):
                msg = f"msg{i:02d}".encode()  # 5 bytes
                assert prod.insert(msg)
                assert cons.consume() == msg
        finally:
            cons.close()
            prod.close()


class TestMemoryEdgeCases:
    """Layer 2 — Edge cases that exercise memory helper boundary conditions."""

    def test_exact_boundary_header_no_wrap(self, tmp_path):
        """Given header aligned to capacity-8, When written, Then fast path used.

        With capacity=32, write_pos=23 gives ring position 23. The next
        message header starts at 23, 23+8=31 ≤ 32, so the fast memcpy path is
        used (no wrap).
        """
        path = str(tmp_path / "wrap.bin")
        prod = ShmSpscProducer(path, 32, create=True, unlink_on_close=True)
        cons = ShmSpscConsumer(path)
        try:
            # message of 15 bytes -> total 23 bytes, write_pos goes 0->23
            assert prod.insert(b"123456789012345")
            # next message (1 byte) header at ring pos 23, 23+8=31 ≤ 32 -> no wrap
            assert prod.insert(b"x")
            assert cons.consume() == b"123456789012345"
            assert cons.consume() == b"x"
        finally:
            cons.close()
            prod.close()

    def test_zero_length_payload_wrap(self, tmp_path):
        """Given zero-length payload, When inserted at boundary, Then works.

        A 0-byte message occupies exactly 8 bytes (header only).  With
        capacity=16, inserting two 0-byte messages puts the second header at
        ring position 8, which does not wrap.  Verify no crash and round-trip ok.
        """
        path = str(tmp_path / "wrap.bin")
        prod = ShmSpscProducer(path, 16, create=True, unlink_on_close=True)
        cons = ShmSpscConsumer(path)
        try:
            assert prod.insert(b"")
            assert prod.insert(b"")
            assert cons.consume() == b""
            assert cons.consume() == b""
        finally:
            cons.close()
            prod.close()
