"""
Shared-memory ring buffer memory utilities.

Provides low-level memory operations for shared-memory ring buffers including
alignment helpers, power-of-two rounding, little-endian u64 read/write with
wrap handling, and bulk copy operations into/from ring buffers.
"""
from libc.stddef cimport size_t
from libc.string cimport memcpy
from libc.stdint cimport uint64_t as u64


cdef size_t align_up(size_t x, size_t a) nogil:
    """Align x up to the next multiple of a.

    Args:
        x: Value to align.
        a: Alignment boundary (must be a power of two).

    Returns:
        The smallest multiple of a that is >= x.
    """
    return (x + (a - 1)) & ~(a - 1)


cdef u64 pow2_at_least(u64 v) nogil:
    """Return smallest power of two >= v.

    Args:
        v: Input value.

    Returns:
        Smallest power of two greater than or equal to v.
    """
    if v <= 1:
        return 1
    v -= 1
    v |= v >> 1
    v |= v >> 2
    v |= v >> 4
    v |= v >> 8
    v |= v >> 16
    v |= v >> 32
    return v + 1


cdef extern from *:
    u64 __atomic_load_n(u64* ptr, int order) noexcept nogil
    void __atomic_store_n(u64* ptr, u64 value, int order) noexcept nogil


cdef void write_u64_le(unsigned char* base, u64 idx, u64 mask, u64 val) nogil:
    """Atomic word storage, also supporting utility calls at byte offsets."""
    copy_into_ring(base, idx, mask, <const unsigned char*>&val, 8, mask + 1)


cdef u64 read_u64_le(const unsigned char* base, u64 idx, u64 mask) nogil:
    cdef u64 val = 0
    copy_from_ring(<unsigned char*>&val, base, idx, mask, 8, mask + 1)
    return val


cdef void copy_into_ring(unsigned char* ring, u64 start, u64 mask,
                         const unsigned char* src, size_t n, u64 capacity) nogil:
    """Copy through aligned release stores; capacity must be at least eight."""
    cdef u64 idx, word
    cdef size_t offset, take
    cdef u64* slot
    if capacity < 8:
        return
    while n:
        idx = start & mask
        offset = <size_t>(idx & 7)
        take = min(<size_t>8 - offset, n)
        slot = <u64*>(ring + (idx & ~<u64>7))
        word = __atomic_load_n(slot, 0) if offset or take < 8 else 0
        memcpy((<unsigned char*>&word) + offset, src, take)
        __atomic_store_n(slot, word, 3)
        start += take
        src += take
        n -= take


cdef void copy_from_ring(unsigned char* dst, const unsigned char* ring, u64 start,
                         u64 mask, size_t n, u64 capacity) nogil:
    """Speculative relaxed loads; callers validate before exposing a copy."""
    cdef u64 idx, word
    cdef size_t offset, take
    if capacity < 8:
        return
    while n:
        idx = start & mask
        offset = <size_t>(idx & 7)
        take = min(<size_t>8 - offset, n)
        word = __atomic_load_n(<u64*>(ring + (idx & ~<u64>7)), 0)
        memcpy(dst, (<const unsigned char*>&word) + offset, take)
        start += take
        dst += take
        n -= take
