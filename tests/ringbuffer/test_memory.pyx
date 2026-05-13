# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
"""
Layered tests for shared-memory ring buffer memory utilities.

Tests align_up, pow2_at_least, write_u64_le, read_u64_le, copy_into_ring,
and copy_from_ring with comprehensive edge cases and boundary values.
"""
from __future__ import annotations

from libc.stdint cimport uint64_t as u64
from libc.stdlib cimport malloc, free

from mm_toolbox.ringbuffer.shm.memory cimport (
    align_up,
    pow2_at_least,
    write_u64_le,
    read_u64_le,
    copy_into_ring,
    copy_from_ring,
)


# =============================================================================
# LAYER 1: Primitives - align_up and pow2_at_least
# =============================================================================

# ---------------------------------------------------------------------------
# align_up tests
# ---------------------------------------------------------------------------

cpdef void test_align_up_zero():
    """Given x=0, When aligned up, Then result is 0."""
    assert align_up(0, 8) == 0


cpdef void test_align_up_less_than_align():
    """Given x < a, When aligned up, Then result is a."""
    assert align_up(3, 8) == 8
    assert align_up(7, 8) == 8
    assert align_up(1, 4096) == 4096


cpdef void test_align_up_equals_align():
    """Given x == a, When aligned up, Then result is x."""
    assert align_up(8, 8) == 8
    assert align_up(4096, 4096) == 4096


cpdef void test_align_up_greater_than_align():
    """Given x > a, When aligned up, Then result is next multiple."""
    assert align_up(9, 8) == 16
    assert align_up(15, 8) == 16
    assert align_up(17, 8) == 24
    assert align_up(4097, 4096) == 8192


cpdef void test_align_up_align_one():
    """Given a=1, When aligned up, Then result equals x."""
    assert align_up(0, 1) == 0
    assert align_up(5, 1) == 5
    assert align_up(100, 1) == 100


# ---------------------------------------------------------------------------
# pow2_at_least tests
# ---------------------------------------------------------------------------

cpdef void test_pow2_at_least_zero():
    """Given v=0, When computing pow2_at_least, Then result is 1."""
    assert pow2_at_least(0) == 1


cpdef void test_pow2_at_least_one():
    """Given v=1, When computing pow2_at_least, Then result is 1."""
    assert pow2_at_least(1) == 1


cpdef void test_pow2_at_least_two():
    """Given v=2, When computing pow2_at_least, Then result is 2."""
    assert pow2_at_least(2) == 2


cpdef void test_pow2_at_least_three():
    """Given v=3, When computing pow2_at_least, Then result is 4."""
    assert pow2_at_least(3) == 4


cpdef void test_pow2_at_least_already_power_of_two():
    """Given v is already a power of two, When computing pow2_at_least, Then result is v."""
    assert pow2_at_least(4) == 4
    assert pow2_at_least(8) == 8
    assert pow2_at_least(16) == 16
    assert pow2_at_least(1024) == 1024
    assert pow2_at_least(2147483648) == 2147483648


cpdef void test_pow2_at_least_large_value():
    """Given v=UINT64_MAX/2, When computing pow2_at_least, Then result is 2^63."""
    cdef u64 half_max = 9223372036854775807  # UINT64_MAX / 2
    assert pow2_at_least(half_max) == 9223372036854775808  # 2^63


cpdef void test_pow2_at_least_just_above_power_of_two():
    """Given v just above a power of two, When computing pow2_at_least, Then result is next power of two."""
    assert pow2_at_least(5) == 8
    assert pow2_at_least(9) == 16
    assert pow2_at_least(17) == 32


# =============================================================================
# LAYER 2: Composite Components - write_u64_le and read_u64_le
# =============================================================================

cpdef void test_write_read_u64_le_basic():
    """Given aligned position, When writing and reading u64, Then value matches."""
    cdef unsigned char* buf = <unsigned char*>malloc(64)
    assert buf != NULL
    try:
        write_u64_le(buf, 0, 63, 0x0102030405060708)
        assert read_u64_le(buf, 0, 63) == 0x0102030405060708
    finally:
        free(buf)


cpdef void test_write_read_u64_le_different_values():
    """Given various u64 values, When written and read back, Then values match."""
    cdef unsigned char* buf = <unsigned char*>malloc(64)
    assert buf != NULL
    try:
        # Zero
        write_u64_le(buf, 0, 63, 0)
        assert read_u64_le(buf, 0, 63) == 0

        # Max uint64
        write_u64_le(buf, 8, 63, 18446744073709551615)
        assert read_u64_le(buf, 8, 63) == 18446744073709551615

        # All ones in each byte position
        write_u64_le(buf, 16, 63, 0xFF00000000000000)
        assert read_u64_le(buf, 16, 63) == 0xFF00000000000000

        write_u64_le(buf, 24, 63, 0x00000000000000FF)
        assert read_u64_le(buf, 24, 63) == 0x00000000000000FF
    finally:
        free(buf)


cpdef void test_write_read_u64_le_wraparound():
    """Given position near end of buffer, When writing 8 bytes, Then wrap-around path used and value matches."""
    cdef unsigned char* buf = <unsigned char*>malloc(16)
    assert buf != NULL
    try:
        # Position 12 in a 16-byte buffer: 12+8=20 > 16, so wrap path is used
        write_u64_le(buf, 12, 15, 0xDEADBEEFCAFEBABE)
        assert read_u64_le(buf, 12, 15) == 0xDEADBEEFCAFEBABE
    finally:
        free(buf)


cpdef void test_write_read_u64_le_boundary_capacity_minus_8():
    """Given position at capacity-8, When writing 8 bytes, Then fast path used and value matches."""
    cdef unsigned char* buf = <unsigned char*>malloc(32)
    assert buf != NULL
    try:
        # Position 24 in a 32-byte buffer: 24+8=32 <= 32, so fast path is used
        write_u64_le(buf, 24, 31, 0x1122334455667788)
        assert read_u64_le(buf, 24, 31) == 0x1122334455667788
    finally:
        free(buf)


cpdef void test_write_read_u64_le_multiple_wrap_positions():
    """Given various wrap positions, When writing and reading, Then all values match."""
    cdef unsigned char* buf = <unsigned char*>malloc(16)
    assert buf != NULL
    try:
        # Test wrap at different offsets
        write_u64_le(buf, 15, 15, 0xAABBCCDDEEFF0011)
        assert read_u64_le(buf, 15, 15) == 0xAABBCCDDEEFF0011

        write_u64_le(buf, 14, 15, 0x1122334455667788)
        assert read_u64_le(buf, 14, 15) == 0x1122334455667788

        write_u64_le(buf, 9, 15, 0x0102030405060708)
        assert read_u64_le(buf, 9, 15) == 0x0102030405060708
    finally:
        free(buf)


# =============================================================================
# LAYER 3: Mini-Integration - copy_into_ring and copy_from_ring
# =============================================================================

cpdef void test_copy_into_ring_no_wrap():
    """Given n < endspace, When copying into ring, Then single memcpy used and data intact."""
    cdef unsigned char* ring = <unsigned char*>malloc(32)
    cdef unsigned char* src = <unsigned char*>malloc(10)
    assert ring != NULL
    assert src != NULL
    try:
        # Initialize source
        for i in range(10):
            src[i] = <unsigned char>(i + 1)

        # Copy 10 bytes starting at position 0 in 32-byte ring (endspace=32)
        copy_into_ring(ring, 0, 31, src, 10, 32)

        # Verify
        for i in range(10):
            assert ring[i] == <unsigned char>(i + 1)
    finally:
        free(ring)
        free(src)


cpdef void test_copy_into_ring_exact_endspace():
    """Given n == endspace, When copying into ring, Then single memcpy used and data intact."""
    cdef unsigned char* ring = <unsigned char*>malloc(32)
    cdef unsigned char* src = <unsigned char*>malloc(8)
    assert ring != NULL
    assert src != NULL
    try:
        for i in range(8):
            src[i] = <unsigned char>(i + 10)

        # Copy 8 bytes starting at position 24 in 32-byte ring (endspace=8)
        copy_into_ring(ring, 24, 31, src, 8, 32)

        for i in range(8):
            assert ring[24 + i] == <unsigned char>(i + 10)
    finally:
        free(ring)
        free(src)


cpdef void test_copy_into_ring_wrap():
    """Given n > endspace, When copying into ring, Then wrap-around used and data intact."""
    cdef unsigned char* ring = <unsigned char*>malloc(16)
    cdef unsigned char* src = <unsigned char*>malloc(10)
    assert ring != NULL
    assert src != NULL
    try:
        for i in range(10):
            src[i] = <unsigned char>(i + 20)

        # Copy 10 bytes starting at position 12 in 16-byte ring (endspace=4)
        copy_into_ring(ring, 12, 15, src, 10, 16)

        # First 4 bytes at ring[12..15]
        for i in range(4):
            assert ring[12 + i] == <unsigned char>(i + 20)
        # Remaining 6 bytes at ring[0..5]
        for i in range(6):
            assert ring[i] == <unsigned char>(4 + i + 20)
    finally:
        free(ring)
        free(src)


cpdef void test_copy_into_ring_zero():
    """Given n=0, When copying into ring, Then no change."""
    cdef unsigned char* ring = <unsigned char*>malloc(16)
    cdef unsigned char* src = <unsigned char*>malloc(4)
    assert ring != NULL
    assert src != NULL
    try:
        # Initialize ring with known values
        for i in range(16):
            ring[i] = <unsigned char>(i + 1)

        # Zero-length copy should be no-op
        copy_into_ring(ring, 0, 15, src, 0, 16)

        for i in range(16):
            assert ring[i] == <unsigned char>(i + 1)
    finally:
        free(ring)
        free(src)


cpdef void test_copy_from_ring_no_wrap():
    """Given n < endspace, When copying from ring, Then single memcpy used and data matches."""
    cdef unsigned char* ring = <unsigned char*>malloc(32)
    cdef unsigned char* dst = <unsigned char*>malloc(10)
    assert ring != NULL
    assert dst != NULL
    try:
        for i in range(32):
            ring[i] = <unsigned char>(i + 1)

        # Read 10 bytes starting at position 0 in 32-byte ring (endspace=32)
        copy_from_ring(dst, ring, 0, 31, 10, 32)

        for i in range(10):
            assert dst[i] == <unsigned char>(i + 1)
    finally:
        free(ring)
        free(dst)


cpdef void test_copy_from_ring_exact_endspace():
    """Given n == endspace, When copying from ring, Then single memcpy used and data matches."""
    cdef unsigned char* ring = <unsigned char*>malloc(32)
    cdef unsigned char* dst = <unsigned char*>malloc(8)
    assert ring != NULL
    assert dst != NULL
    try:
        for i in range(32):
            ring[i] = <unsigned char>(i + 50)

        # Read 8 bytes starting at position 24 in 32-byte ring (endspace=8)
        copy_from_ring(dst, ring, 24, 31, 8, 32)

        for i in range(8):
            assert dst[i] == <unsigned char>(24 + i + 50)
    finally:
        free(ring)
        free(dst)


cpdef void test_copy_from_ring_wrap():
    """Given n > endspace, When copying from ring, Then wrap-around used and data matches."""
    cdef unsigned char* ring = <unsigned char*>malloc(16)
    cdef unsigned char* dst = <unsigned char*>malloc(10)
    assert ring != NULL
    assert dst != NULL
    try:
        for i in range(16):
            ring[i] = <unsigned char>(i + 100)

        # Read 10 bytes starting at position 12 in 16-byte ring (endspace=4)
        copy_from_ring(dst, ring, 12, 15, 10, 16)

        # First 4 bytes from ring[12..15]
        for i in range(4):
            assert dst[i] == <unsigned char>(12 + i + 100)
        # Remaining 6 bytes from ring[0..5]
        for i in range(6):
            assert dst[4 + i] == <unsigned char>(i + 100)
    finally:
        free(ring)
        free(dst)


cpdef void test_copy_from_ring_zero():
    """Given n=0, When copying from ring, Then destination unchanged."""
    cdef unsigned char* ring = <unsigned char*>malloc(16)
    cdef unsigned char* dst = <unsigned char*>malloc(4)
    assert ring != NULL
    assert dst != NULL
    try:
        for i in range(4):
            dst[i] = <unsigned char>(0xFF)

        # Zero-length copy should be no-op
        copy_from_ring(dst, ring, 0, 15, 0, 16)

        for i in range(4):
            assert dst[i] == <unsigned char>(0xFF)
    finally:
        free(ring)
        free(dst)


cpdef void test_copy_roundtrip():
    """Given data copied into ring, When copied back out, Then data matches original."""
    cdef unsigned char* ring = <unsigned char*>malloc(32)
    cdef unsigned char* src = <unsigned char*>malloc(20)
    cdef unsigned char* dst = <unsigned char*>malloc(20)
    assert ring != NULL
    assert src != NULL
    assert dst != NULL
    try:
        for i in range(20):
            src[i] = <unsigned char>(i + 1)

        # Copy in at position 20 (endspace=12, so wrap)
        copy_into_ring(ring, 20, 31, src, 20, 32)

        # Copy back out from same position
        copy_from_ring(dst, ring, 20, 31, 20, 32)

        for i in range(20):
            assert dst[i] == src[i]
    finally:
        free(ring)
        free(src)
        free(dst)
