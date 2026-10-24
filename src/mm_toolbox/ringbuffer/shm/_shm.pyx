# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

"""
Common base for shared-memory ring buffers.

Provides constants, header sizes, and lifecycle management (mmap, unmap, close)
for shared-memory ring buffer implementations (SPSC and MPSC).
"""

import os
from cpython.bytearray cimport PyByteArray_Resize
from libc.string cimport memcpy
from libc.stdint cimport uint64_t as u64
from libc.stddef cimport size_t
from libc.errno cimport errno

from .header cimport ShmHeader, ShmMpscGlobalHeader, ShmSubRingHeader
from .memory cimport align_up, pow2_at_least
from .atomics cimport atomic_load_acquire

# Constants
cdef u64 _MAGIC = 0x3253484252  # 'SHBR'
cdef u64 _MPSC_MAGIC = 0x3253484D50  # 'SHMP'
cdef size_t _HEADER_ALIGN = 64
cdef size_t _HEADER_SIZE = align_up(sizeof(ShmHeader), _HEADER_ALIGN)
cdef size_t _MPSC_GLOBAL_HEADER_SIZE = align_up(sizeof(ShmMpscGlobalHeader), _HEADER_ALIGN)
cdef size_t _SUB_RING_HEADER_SIZE = align_up(sizeof(ShmSubRingHeader), _HEADER_ALIGN)


cdef class _ShmRingBase:
    """Common lifecycle for shared ring buffers.

    Manages mmap lifecycle, file descriptors, and cleanup for shared-memory
    ring buffer implementations.
    """

    def __cinit__(self) -> None:
        """Initialize the base shared ring buffer.

        Raises:
            OSError: If the platform is not POSIX.
        """
        if os.name != "posix":
            raise OSError("Shared memory ringbuffer is only supported on POSIX platforms")
        self._base = NULL
        self._map_len = 0
        self._fd = -1
        self._owner = False
        self._unlink_on_close = False
        self._spin_wait = 1024
        self._path_py = None
        self._path = NULL

    cdef inline void _ensure_open(self):
        if self._base == NULL:
            raise RuntimeError("Shared ring is closed")

    cdef void _release_lease(self) noexcept:
        pass

    cdef inline void _close_map(self):
        """Unmap and close backing file."""
        if self._base != NULL:
            self._release_lease()
            munmap(self._base, self._map_len)
            self._base = NULL
            self._map_len = 0
        if self._fd >= 0:
            close(self._fd)
            self._fd = -1

    def __dealloc__(self):
        """Deallocate the shared ring buffer, unmapping memory."""
        self._close_map()

    cpdef void close(self):
        """Close the shared ring buffer and optionally unlink the backing file.

        If this instance is the owner and unlink_on_close is enabled, the
        backing file is removed from the filesystem.
        """
        self._close_map()
        if self._owner and self._unlink_on_close and self._path_py is not None:
            try:
                os.unlink(self._path_py)
            except Exception:
                pass
        self._owner = False

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc, tb):
        """Exit context manager, closing the ring buffer."""
        self.close()


cdef object _try_consume(ShmConsumerContext* ctx, Py_ssize_t limit=-1):
    """Return only a successfully retired private copy; None means retry/empty."""
    cdef u64 length, read
    cdef int success
    cdef bytearray scratch
    cdef unsigned char* dst
    with nogil:
        success = shm_consumer_peek_available(ctx, &length, &read)
    if not success:
        return None
    if limit >= 0 and length > <u64>limit:
        raise ValueError(f"Message size {length} exceeds buffer size {limit}")
    scratch = bytearray(<Py_ssize_t>length)
    dst = <unsigned char*>scratch
    with nogil:
        success = shm_consumer_consume(ctx, dst, length, read)
    if not success:
        return None
    return bytes(scratch)


cdef int _try_consume_into(ShmConsumerContext* ctx, bytearray dst, bytearray scratch):
    """Reuse private scratch; failed retirement leaves caller dst untouched.

    Grow scratch only for a new largest payload. Hold the GIL across this
    operation so Python threads cannot concurrently reuse this private array.
    Other processes keep progressing through the native lock-free algorithm.
    """
    cdef u64 length, read
    if not shm_consumer_peek_available(ctx, &length, &read):
        return -1
    if length > <u64>len(dst):
        raise ValueError(f"Message size {length} exceeds buffer size {len(dst)}")
    if length > <u64>len(scratch):
        PyByteArray_Resize(scratch, <Py_ssize_t>length)
    if not shm_consumer_consume(ctx, <unsigned char*>scratch, length, read):
        return -1
    memcpy(<unsigned char*>dst, <const unsigned char*>scratch, <size_t>length)
    return <int>length


cdef list _snapshot(ShmConsumerContext* ctx):
    """Coherent per-ring snapshot. No speculative word escapes validation."""
    cdef u64 read, write, pos, length
    cdef int success, valid
    cdef list result
    cdef bytearray scratch
    cdef unsigned char* dst
    while True:
        with nogil:
            success = shm_snapshot_begin(ctx, &read, &write)
        if not success:
            return []
        result = []
        pos = read
        valid = 1
        while pos < write:
            with nogil:
                success = shm_snapshot_record(ctx, pos, write, &length)
            if not success:
                valid = 0
                break
            scratch = bytearray(<Py_ssize_t>length)
            dst = <unsigned char*>scratch
            with nogil:
                shm_copy_from_ring(dst, ctx.data, pos + 8, ctx.mask, length, ctx.capacity)
            result.append(bytes(scratch))
            pos += shm_record_span(length)
        with nogil:
            success = shm_snapshot_valid(ctx, read)
        if success:
            return result if valid else []


cdef bint _insert_batch(ShmProducerContext* ctx, list items, u64* dropped):
    """Validate the complete padded batch before reserving or retiring data."""
    cdef u64 total = 0, span, pos
    cdef bytes item
    dropped[0] = 0
    for item in items:
        span = shm_record_span(<u64>len(item))
        if not span or span > ctx.capacity - total:
            return False
        total += span
    if not total:
        return True
    if not shm_producer_reserve(ctx, total, dropped):
        return False
    pos = ctx.cached_write
    for item in items:
        shm_producer_write(ctx, <const unsigned char*>item, len(item), pos)
        pos += shm_record_span(<u64>len(item))
    shm_producer_publish(ctx, pos)
    return True


cdef u64 _count(ShmHeader* hdr, unsigned char* data, u64 capacity):
    cdef ShmConsumerContext ctx
    cdef u64 result
    ctx.hdr = hdr
    ctx.data = data
    ctx.capacity = capacity
    ctx.mask = capacity - 1
    ctx.spin_wait = 0
    with nogil:
        result = shm_message_count(&ctx)
    return result


cdef bint _full(ShmHeader* hdr, u64 capacity):
    cdef u64 read, write
    with nogil:
        read = atomic_load_acquire(&hdr.read_pos)
        write = atomic_load_acquire(&hdr.write_pos)
    return read <= write and write - read <= capacity and capacity - (write - read) < 8
