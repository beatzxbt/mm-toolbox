# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

"""SPSC shared-memory bytes ring buffers."""

import ctypes
import os
import stat
from libc.stdint cimport uint64_t as u64
from libc.string cimport memcpy
from libc.stddef cimport size_t
from libc.errno cimport errno
from libc.stdlib cimport malloc, free

_libc = ctypes.CDLL(None)
_sched_yield = _libc.sched_yield

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
    int O_EXCL
    int O_NOFOLLOW
    int O_CLOEXEC

cdef extern from "unistd.h":
    int ftruncate(int fd, long length)
    int close(int fd)

cdef extern from "../../time/ctime_impl.h":
    long long c_time_monotonic_ns() nogil

from .header cimport ShmHeader
from ._shm cimport (
    _ShmRingBase, _MAGIC, _HEADER_SIZE, _HEADER_ALIGN,
    ShmProducerContext, ShmConsumerContext, shm_producer_claim,
    shm_producer_release, shm_producer_insert, _try_consume, _try_consume_into, _snapshot,
    _insert_batch, _count, _full,
)
from .atomics cimport atomic_load_acquire, atomic_store_release
from .memory cimport pow2_at_least


cdef class _SharedBytesRing(_ShmRingBase):
    """Common mapping and header state for SPSC shared ring buffers.

    Manages mmap lifecycle, header initialization, and attachment for
    single-producer single-consumer shared-memory queues.
    """

    cdef ShmHeader* _hdr
    cdef unsigned char* _data
    cdef u64 _capacity
    cdef u64 _mask
    cdef u64 _cached_read
    cdef u64 _cached_write

    def __cinit__(self) -> None:
        self._hdr = NULL
        self._data = NULL
        self._capacity = 0
        self._mask = 0
        self._cached_read = 0
        self._cached_write = 0

    cdef void _map_create(self, bytes path_b, u64 capacity_bytes, bint unlink_on_close, int spin_wait):
        """Create and initialize a shared ring."""
        cdef:
            int fd
            size_t cap = <size_t>pow2_at_least(capacity_bytes if capacity_bytes >= 8 else 8)
            size_t total_len = _HEADER_SIZE + cap
            void* base
        fd = open(path_b, O_CREAT | O_RDWR | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0o600)
        if fd < 0:
            raise OSError(errno, "open failed for shared ring")
        if ftruncate(fd, <long long>total_len) != 0:
            close(fd)
            raise OSError(errno, "ftruncate failed for shared ring")
        os.fsync(fd)
        base = mmap(NULL, total_len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)
        if <long>base == -1:
            close(fd)
            try:
                os.unlink(path_b)
            except Exception:
                pass
            finally:
                raise OSError(errno, "mmap failed for shared ring")

        self._fd = fd
        self._map_len = total_len
        self._base = base
        self._hdr = <ShmHeader*>base
        self._data = <unsigned char*>base + _HEADER_SIZE
        self._owner = True
        self._unlink_on_close = unlink_on_close
        self._spin_wait = spin_wait if spin_wait > 0 else 1024
        self._capacity = <u64>cap
        self._mask = <u64>(cap - 1)
        self._cached_read = 0
        self._cached_write = 0

        self._hdr.capacity = <u64>cap
        self._hdr.mask = <u64>(cap - 1)
        self._hdr.write_pos = 0
        self._hdr.read_pos = 0
        self._hdr.producer_claim = 0
        self._hdr.dropped_count = 0
        self._hdr.latest_insert_time_ns = 0
        self._hdr.latest_consume_time_ns = 0
        atomic_store_release(&self._hdr.magic, _MAGIC)

    cdef void _map_attach(self, bytes path_b, int spin_wait):
        """Attach to an existing shared ring."""
        cdef:
            int fd
            void* base
            size_t total_len
            u64 capacity
            u64 mask
            object backing_len
        fd = open(path_b, O_RDWR | O_NOFOLLOW | O_CLOEXEC, 0o600)
        if fd < 0:
            raise OSError(errno, "open failed for shared ring")
        try:
            st = os.fstat(fd)
            backing_len = st.st_size
        except Exception:
            close(fd)
            raise
        if not stat.S_ISREG(st.st_mode):
            close(fd)
            raise RuntimeError("Shared ring backing file is not a regular file")
        if stat.S_IMODE(st.st_mode) != 0o600:
            close(fd)
            raise RuntimeError("Shared ring backing file has incorrect permissions")
        if backing_len < _HEADER_SIZE:
            close(fd)
            raise RuntimeError("Shared ring backing file too small for header")
        base = mmap(NULL, _HEADER_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)
        if <long>base == -1:
            close(fd)
            raise OSError(errno, "mmap header failed for shared ring")
        cdef ShmHeader* tmp_hdr = <ShmHeader*>base
        if atomic_load_acquire(&tmp_hdr.magic) != _MAGIC:
            munmap(base, _HEADER_SIZE)
            close(fd)
            raise RuntimeError("Shared ring header mismatch")
        capacity = tmp_hdr.capacity
        mask = tmp_hdr.mask
        if capacity < 8 or (capacity & (capacity - 1)) != 0 or mask != capacity - 1:
            munmap(base, _HEADER_SIZE)
            close(fd)
            raise RuntimeError("Shared ring header invalid")
        if capacity > <u64>((<size_t>-1) - _HEADER_SIZE):
            munmap(base, _HEADER_SIZE)
            close(fd)
            raise RuntimeError("Shared ring header invalid")

        total_len = _HEADER_SIZE + <size_t>capacity
        try:
            backing_len = os.fstat(fd).st_size
        except Exception:
            munmap(base, _HEADER_SIZE)
            close(fd)
            raise
        if backing_len < total_len:
            munmap(base, _HEADER_SIZE)
            close(fd)
            raise RuntimeError("Shared ring backing file too small for header capacity")
        munmap(base, _HEADER_SIZE)

        base = mmap(NULL, total_len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)
        if <long>base == -1:
            close(fd)
            raise OSError(errno, "mmap failed for shared ring")

        self._fd = fd
        self._map_len = total_len
        self._base = base
        self._hdr = <ShmHeader*>base
        self._data = <unsigned char*>base + _HEADER_SIZE
        self._owner = False
        self._unlink_on_close = False
        self._spin_wait = spin_wait if spin_wait > 0 else 1024
        self._capacity = capacity
        self._mask = mask
        self._cached_read = atomic_load_acquire(&self._hdr.read_pos)
        self._cached_write = atomic_load_acquire(&self._hdr.write_pos)

    def __len__(self) -> int:
        self._ensure_open()
        return <Py_ssize_t>_count(self._hdr, self._data, self._capacity)

    @property
    def dropped_count(self) -> int:
        """Number of records retired by overwrite since creation."""
        self._ensure_open()
        return atomic_load_acquire(&self._hdr.dropped_count)

    @property
    def latest_insert_time_ns(self) -> int:
        self._ensure_open()
        return atomic_load_acquire(&self._hdr.latest_insert_time_ns)

    @property
    def latest_consume_time_ns(self) -> int:
        self._ensure_open()
        return atomic_load_acquire(&self._hdr.latest_consume_time_ns)


cdef class ShmSpscProducer(_SharedBytesRing):
    """One exclusively leased writer. Close releases the lease for reuse.

    An instance is used by one thread/process at a time. A forked child must
    attach its own producer; inherited producer objects cannot write/release.
    Records overwrite the oldest entries and use eight-byte padded spans.
    """
    cdef ShmProducerContext _prod_ctx
    cdef u64 _lease_owner
    cdef u64 _last_insert_dropped
    cdef u64 _failed_inserts

    def __cinit__(self, path: str, int capacity_bytes, *, bint create=True,
                  bint unlink_on_close=False, int spin_wait=1024) -> None:
        if capacity_bytes < 0:
            raise ValueError("capacity_bytes must be >= 0")
        self._path_py = path
        path_b = path.encode()
        if create:
            self._map_create(path_b, <u64>capacity_bytes, unlink_on_close, spin_wait)
        else:
            self._map_attach(path_b, spin_wait)
        if not shm_producer_claim(self._hdr, <u64>os.getpid()):
            raise RuntimeError("Shared ring already has an active producer")
        self._lease_owner = <u64>os.getpid()
        self._prod_ctx.hdr = self._hdr
        self._prod_ctx.data = self._data
        self._prod_ctx.capacity = self._capacity
        self._prod_ctx.mask = self._mask
        self._prod_ctx.cached_read = atomic_load_acquire(&self._hdr.read_pos)
        self._prod_ctx.cached_write = atomic_load_acquire(&self._hdr.write_pos)

    cdef void _release_lease(self) noexcept:
        if self._base != NULL and self._lease_owner:
            if self._lease_owner == <u64>os.getpid():
                shm_producer_release(self._hdr, self._lease_owner)
            self._lease_owner = 0

    def __dealloc__(self):
        self._release_lease()

    cdef inline void _ensure_writer(self):
        self._ensure_open()
        if self._lease_owner != <u64>os.getpid():
            raise RuntimeError("Inherited producer cannot write; attach a new producer")

    cpdef bint insert(self, bytes item):
        """Insert and report success. Inspect last_insert_dropped for loss."""
        cdef int success
        self._ensure_writer()
        # Keep the GIL across the writer operation, so Python threads sharing
        # this instance cannot become concurrent writers while a copy runs.
        success = shm_producer_insert(&self._prod_ctx, <const unsigned char*>item,
                                      len(item), &self._last_insert_dropped)
        if not success:
            self._failed_inserts += 1
        return success == 1

    cdef bint insert_char(self, const char* data, size_t n):
        cdef int success
        self._ensure_writer()
        success = shm_producer_insert(&self._prod_ctx, <const unsigned char*>data,
                                      n, &self._last_insert_dropped)
        if not success:
            self._failed_inserts += 1
        return success == 1

    cpdef bint insert_batch(self, list[bytes] items):
        cdef bint success
        self._ensure_writer()
        success = _insert_batch(&self._prod_ctx, items, &self._last_insert_dropped)
        if not success:
            self._failed_inserts += 1
        return success

    @property
    def last_insert_dropped(self) -> int:
        return self._last_insert_dropped

    @property
    def failed_inserts(self) -> int:
        return self._failed_inserts

    cpdef bint is_empty(self):
        return len(self) == 0

    cpdef bint is_full(self):
        self._ensure_open()
        return _full(self._hdr, self._capacity)


cdef class ShmSpscConsumer(_SharedBytesRing):
    """Consumer returning only copies whose retirement CAS succeeded."""
    cdef ShmConsumerContext _cons_ctx
    cdef bytearray _scratch

    def __cinit__(self, path: str, *, int spin_wait=1024) -> None:
        self._path_py = path
        self._map_attach(path.encode(), spin_wait)
        self._cons_ctx.hdr = self._hdr
        self._cons_ctx.data = self._data
        self._cons_ctx.capacity = self._capacity
        self._cons_ctx.mask = self._mask
        self._cons_ctx.spin_wait = self._spin_wait
        self._scratch = bytearray()

    cpdef bytes consume(self):
        cdef object result
        cdef int spins = 0
        while True:
            self._ensure_open()
            result = _try_consume(&self._cons_ctx)
            if result is not None:
                return result
            spins += 1
            if spins >= self._spin_wait:
                _sched_yield()
                spins = 0

    cpdef int consume_into(self, bytearray dst):
        """Copy via reusable scratch; destination changes only after retirement."""
        cdef int copied, spins = 0
        while True:
            self._ensure_open()
            copied = _try_consume_into(&self._cons_ctx, dst, self._scratch)
            if copied >= 0:
                return copied
            spins += 1
            if spins >= self._spin_wait:
                _sched_yield()
                spins = 0

    cpdef int consume_all_into(self, list[bytearray] buffers):
        cdef bytearray dst
        cdef int copied, total = 0
        self._ensure_open()
        for dst in buffers:
            copied = _try_consume_into(&self._cons_ctx, dst, self._scratch)
            if copied < 0:
                break
            total += 1
        return total

    cpdef list consume_all(self):
        cdef list result = []
        cdef object item
        self._ensure_open()
        while True:
            item = _try_consume(&self._cons_ctx)
            if item is None:
                return result
            result.append(item)

    cpdef list unwrapped(self):
        self._ensure_open()
        return _snapshot(&self._cons_ctx)

    cpdef object peekleft(self):
        cdef list items = self.unwrapped()
        return items[0] if items else None

    cpdef object peekright(self):
        cdef list items = self.unwrapped()
        return items[len(items) - 1] if items else None

    cpdef bint contains(self, bytes item):
        return item in self.unwrapped()

    def __contains__(self, bytes item):
        return self.contains(item)

    def consume_iterable(self):
        while True:
            yield self.consume()

    async def aconsume(self):
        import asyncio
        cdef object result
        while True:
            self._ensure_open()
            result = _try_consume(&self._cons_ctx)
            if result is not None:
                return result
            await asyncio.sleep(0)

    async def aconsume_iterable(self):
        while True:
            yield await self.aconsume()

    cpdef bint is_empty(self):
        return len(self) == 0

    cpdef bint is_full(self):
        self._ensure_open()
        return _full(self._hdr, self._capacity)

    cpdef void clear(self):
        self.consume_all()
