# cython: language_level=3

from libc.stdint cimport uint64_t as u64
from libc.stddef cimport size_t

# C library externs
cdef extern from "sys/mman.h":
    void* mmap(void* addr, size_t length, int prot, int flags, int fd, long offset)
    int munmap(void* addr, size_t length)
    int PROT_READ
    int PROT_WRITE
    int MAP_SHARED

cdef extern from "fcntl.h":
    int open(const char* path, int oflag, ...)
    int O_RDWR
    int O_CREAT

cdef extern from "unistd.h":
    int ftruncate(int fd, long length)
    int close(int fd)

cdef extern from "../../time/ctime_impl.h":
    long long c_time_monotonic_ns() nogil

from .header cimport ShmHeader

cdef extern from "c/shm_helpers.h":
    void shm_copy_from_ring(unsigned char* dst, const unsigned char* ring, u64 start,
                           u64 mask, size_t n, u64 capacity) nogil

cdef extern from "c/shm_core.h":
    ctypedef struct ShmProducerContext:
        ShmHeader* hdr
        unsigned char* data
        u64 capacity
        u64 mask
        u64 cached_read
        u64 cached_write

    ctypedef struct ShmConsumerContext:
        ShmHeader* hdr
        unsigned char* data
        u64 capacity
        u64 mask
        int spin_wait

    int shm_producer_insert(ShmProducerContext* ctx, const unsigned char* payload,
                           size_t payload_len, u64* dropped_out) nogil
    int shm_consumer_peek_available(ShmConsumerContext* ctx, u64* msg_len_out,
                                   u64* read_pos_out) nogil
    int shm_consumer_consume(ShmConsumerContext* ctx, unsigned char* dst,
                             u64 msg_len, u64 read_pos) nogil

# Constants
cdef u64 _MAGIC
cdef u64 _MPSC_MAGIC
cdef size_t _HEADER_ALIGN
cdef size_t _HEADER_SIZE
cdef size_t _MPSC_GLOBAL_HEADER_SIZE
cdef size_t _SUB_RING_HEADER_SIZE

# Base class
cdef class _ShmRingBase:
    cdef void* _base
    cdef size_t _map_len
    cdef int _fd
    cdef bint _owner
    cdef bint _unlink_on_close
    cdef int _spin_wait
    cdef object _path_py
    cdef const char* _path

    cdef inline void _close_map(self)
    cpdef void close(self)
