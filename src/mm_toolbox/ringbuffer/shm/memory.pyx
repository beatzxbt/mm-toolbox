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


cdef void write_u64_le(unsigned char* base, u64 idx, u64 mask, u64 val) nogil:
    """Write u64 in little-endian with wrap handling.

    Args:
        base: Pointer to the ring buffer base.
        idx: Absolute index in the ring.
        mask: Capacity mask (capacity - 1, capacity must be power of two).
        val: Value to write.
    """
    cdef u64 pos = idx & mask
    cdef u64 capacity = mask + 1
    if pos + 8 <= capacity:
        # Unaligned access is safe and fast on x86_64 and ARM64
        memcpy(base + pos, &val, 8)
    else:
        base[(idx + 0) & mask] = <unsigned char>(val & 0xFF)
        base[(idx + 1) & mask] = <unsigned char>((val >> 8) & 0xFF)
        base[(idx + 2) & mask] = <unsigned char>((val >> 16) & 0xFF)
        base[(idx + 3) & mask] = <unsigned char>((val >> 24) & 0xFF)
        base[(idx + 4) & mask] = <unsigned char>((val >> 32) & 0xFF)
        base[(idx + 5) & mask] = <unsigned char>((val >> 40) & 0xFF)
        base[(idx + 6) & mask] = <unsigned char>((val >> 48) & 0xFF)
        base[(idx + 7) & mask] = <unsigned char>((val >> 56) & 0xFF)


cdef u64 read_u64_le(const unsigned char* base, u64 idx, u64 mask) nogil:
    """Read u64 in little-endian with wrap handling.

    Args:
        base: Pointer to the ring buffer base.
        idx: Absolute index in the ring.
        mask: Capacity mask (capacity - 1, capacity must be power of two).

    Returns:
        The u64 value at the given position.
    """
    cdef u64 pos = idx & mask
    cdef u64 capacity = mask + 1
    cdef u64 val
    if pos + 8 <= capacity:
        memcpy(&val, base + pos, 8)
        return val
    else:
        return (
            (<u64>base[(idx + 0) & mask])
            | (<u64>base[(idx + 1) & mask] << 8)
            | (<u64>base[(idx + 2) & mask] << 16)
            | (<u64>base[(idx + 3) & mask] << 24)
            | (<u64>base[(idx + 4) & mask] << 32)
            | (<u64>base[(idx + 5) & mask] << 40)
            | (<u64>base[(idx + 6) & mask] << 48)
            | (<u64>base[(idx + 7) & mask] << 56)
        )


cdef void copy_into_ring(unsigned char* ring, u64 start, u64 mask, const unsigned char* src, size_t n, u64 capacity) nogil:
    """Copy contiguous bytes into ring with wrap handling.

    Args:
        ring: Pointer to the ring buffer base.
        start: Absolute start index in the ring.
        mask: Capacity mask.
        src: Source buffer pointer.
        n: Number of bytes to copy.
        capacity: Ring buffer capacity.
    """
    if n == 0:
        return
    cdef u64 idx = start & mask
    cdef u64 endspace = capacity - idx
    if n <= endspace:
        memcpy(ring + idx, src, n)
    else:
        memcpy(ring + idx, src, <size_t>endspace)
        memcpy(ring, src + <size_t>endspace, n - <size_t>endspace)


cdef void copy_from_ring(unsigned char* dst, const unsigned char* ring, u64 start, u64 mask, size_t n, u64 capacity) nogil:
    """Copy contiguous bytes from ring with wrap handling.

    Args:
        dst: Destination buffer pointer.
        ring: Pointer to the ring buffer base.
        start: Absolute start index in the ring.
        mask: Capacity mask.
        n: Number of bytes to copy.
        capacity: Ring buffer capacity.
    """
    if n == 0:
        return
    cdef u64 idx = start & mask
    cdef u64 endspace = capacity - idx
    if n <= endspace:
        memcpy(dst, ring + idx, n)
    else:
        memcpy(dst, ring + idx, <size_t>endspace)
        memcpy(dst + <size_t>endspace, ring, n - <size_t>endspace)
