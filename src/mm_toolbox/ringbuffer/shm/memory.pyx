from libc.stddef cimport size_t
from libc.string cimport memcpy
from libc.stdint cimport uint64_t as u64


cdef size_t align_up(size_t x, size_t a) nogil:
    """Align x up to the next multiple of a."""
    return (x + (a - 1)) & ~(a - 1)


cdef u64 pow2_at_least(u64 v) nogil:
    """Return smallest power of two >= v."""
    cdef u64 p = 1
    if v <= 1:
        return 1
    while p < v:
        p <<= 1
    return p


cdef void write_u64_le(unsigned char* base, u64 idx, u64 mask, u64 val) nogil:
    """Write u64 in little-endian with wrap handling."""
    cdef u64 pos = idx & mask
    cdef u64 capacity = mask + 1
    if pos + 8 <= capacity:
        if pos & 7 == 0:
            (<u64*>(base + pos))[0] = val
        else:
            base[pos + 0] = <unsigned char>(val & 0xFF)
            base[pos + 1] = <unsigned char>((val >> 8) & 0xFF)
            base[pos + 2] = <unsigned char>((val >> 16) & 0xFF)
            base[pos + 3] = <unsigned char>((val >> 24) & 0xFF)
            base[pos + 4] = <unsigned char>((val >> 32) & 0xFF)
            base[pos + 5] = <unsigned char>((val >> 40) & 0xFF)
            base[pos + 6] = <unsigned char>((val >> 48) & 0xFF)
            base[pos + 7] = <unsigned char>((val >> 56) & 0xFF)
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
    """Read u64 in little-endian with wrap handling."""
    cdef u64 pos = idx & mask
    cdef u64 capacity = mask + 1
    if pos + 8 <= capacity:
        if pos & 7 == 0:
            return (<u64*>(base + pos))[0]
        else:
            return (
                (<u64>base[pos + 0])
                | (<u64>base[pos + 1] << 8)
                | (<u64>base[pos + 2] << 16)
                | (<u64>base[pos + 3] << 24)
                | (<u64>base[pos + 4] << 32)
                | (<u64>base[pos + 5] << 40)
                | (<u64>base[pos + 6] << 48)
                | (<u64>base[pos + 7] << 56)
            )
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
    """Copy contiguous bytes into ring with wrap handling."""
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
    """Copy contiguous bytes from ring with wrap handling."""
    if n == 0:
        return
    cdef u64 idx = start & mask
    cdef u64 endspace = capacity - idx
    if n <= endspace:
        memcpy(dst, ring + idx, n)
    else:
        memcpy(dst, ring + idx, <size_t>endspace)
        memcpy(dst + <size_t>endspace, ring, n - <size_t>endspace)
