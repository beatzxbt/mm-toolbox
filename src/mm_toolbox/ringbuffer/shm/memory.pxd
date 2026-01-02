from libc.stddef cimport size_t
from libc.string cimport memcpy
from libc.stdint cimport uint64_t as u64


cdef size_t align_up(size_t x, size_t a) nogil
cdef u64 pow2_at_least(u64 v) nogil
cdef void write_u64_le(unsigned char* base, u64 idx, u64 mask, u64 val) nogil
cdef u64 read_u64_le(const unsigned char* base, u64 idx, u64 mask) nogil
cdef void copy_into_ring(unsigned char* ring, u64 start, u64 mask, const unsigned char* src, size_t n, u64 capacity) nogil
cdef void copy_from_ring(unsigned char* dst, const unsigned char* ring, u64 start, u64 mask, size_t n, u64 capacity) nogil
