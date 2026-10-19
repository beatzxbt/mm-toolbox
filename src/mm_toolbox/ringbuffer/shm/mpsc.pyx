# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

"""MPSC queues composed of exclusively leased lock-free SPSC shards."""

import ctypes
import os
import stat
from libc.stdint cimport uint64_t as u64
from libc.stddef cimport size_t
from libc.errno cimport errno
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy

from .header cimport ShmHeader, ShmMpscGlobalHeader, ShmSubRingHeader
from ._shm cimport (
    _ShmRingBase, _MAGIC, _MPSC_MAGIC, _MPSC_GLOBAL_HEADER_SIZE,
    _SUB_RING_HEADER_SIZE, _HEADER_ALIGN, ShmProducerContext, ShmConsumerContext,
    shm_producer_claim, shm_producer_release, shm_producer_insert,
    _try_consume, _try_consume_into, _snapshot, _insert_batch, _count, _full,
)
from .atomics cimport atomic_add, atomic_load_acquire, atomic_store_release
from .memory cimport align_up, pow2_at_least

cdef extern from "sys/mman.h":
    void* mmap(void*, size_t, int, int, int, long)
    int munmap(void*, size_t)
    int PROT_READ
    int PROT_WRITE
    int MAP_SHARED

cdef extern from "fcntl.h":
    int open(const char*, int, ...)
    int O_RDWR
    int O_CREAT
    int O_EXCL
    int O_NOFOLLOW
    int O_CLOEXEC

cdef extern from "unistd.h":
    int ftruncate(int, long)
    int close(int)

_libc = ctypes.CDLL(None)
_sched_yield = _libc.sched_yield


cdef class _MpscRing(_ShmRingBase):
    cdef ShmMpscGlobalHeader* _ghdr
    cdef u64 _num_rings
    cdef u64 _per_ring_capacity
    cdef size_t _ring_stride
    cdef ShmSubRingHeader** _sub_hdrs
    cdef unsigned char** _sub_datas
    cdef ShmProducerContext* _prod_ctxs
    cdef ShmConsumerContext* _cons_ctxs

    def __cinit__(self):
        self._ghdr = NULL
        self._sub_hdrs = NULL
        self._sub_datas = NULL
        self._prod_ctxs = NULL
        self._cons_ctxs = NULL

    cdef void _layout(self, u64 nr, u64 capacity):
        if nr == 0 or capacity < 8 or capacity & (capacity - 1):
            raise RuntimeError("MPSC shared ring header invalid")
        if capacity > <u64>((<size_t>-1) - _SUB_RING_HEADER_SIZE - _HEADER_ALIGN):
            raise RuntimeError("MPSC shared ring header invalid")
        self._ring_stride = align_up(_SUB_RING_HEADER_SIZE + <size_t>capacity, _HEADER_ALIGN)
        if nr > <u64>(((<size_t>-1) - _MPSC_GLOBAL_HEADER_SIZE) // self._ring_stride):
            raise RuntimeError("MPSC shared ring header invalid")
        self._num_rings = nr
        self._per_ring_capacity = capacity

    cdef void _contexts(self, bint initialize):
        cdef u64 i
        cdef ShmSubRingHeader* hdr
        cdef unsigned char* data
        cdef size_t nr = <size_t>self._num_rings
        self._sub_hdrs = <ShmSubRingHeader**>malloc(nr * sizeof(ShmSubRingHeader*))
        self._sub_datas = <unsigned char**>malloc(nr * sizeof(unsigned char*))
        self._prod_ctxs = <ShmProducerContext*>malloc(nr * sizeof(ShmProducerContext))
        self._cons_ctxs = <ShmConsumerContext*>malloc(nr * sizeof(ShmConsumerContext))
        if (self._sub_hdrs == NULL or self._sub_datas == NULL or
                self._prod_ctxs == NULL or self._cons_ctxs == NULL):
            raise MemoryError("Failed to allocate MPSC sub-ring arrays")
        for i in range(self._num_rings):
            hdr = <ShmSubRingHeader*>(<unsigned char*>self._base + _MPSC_GLOBAL_HEADER_SIZE + i * self._ring_stride)
            data = <unsigned char*>hdr + _SUB_RING_HEADER_SIZE
            self._sub_hdrs[i] = hdr
            self._sub_datas[i] = data
            if initialize:
                hdr.capacity = self._per_ring_capacity
                hdr.mask = self._per_ring_capacity - 1
                hdr.write_pos = 0
                hdr.read_pos = 0
                hdr.producer_claim = 0
                hdr.dropped_count = 0
                hdr.latest_insert_time_ns = 0
                hdr.latest_consume_time_ns = 0
                atomic_store_release(&hdr.magic, _MAGIC)
            elif (atomic_load_acquire(&hdr.magic) != _MAGIC or
                  hdr.capacity != self._per_ring_capacity or hdr.mask != self._per_ring_capacity - 1):
                raise RuntimeError("MPSC shared ring sub-header invalid")
            self._prod_ctxs[i].hdr = hdr
            self._prod_ctxs[i].data = data
            self._prod_ctxs[i].capacity = self._per_ring_capacity
            self._prod_ctxs[i].mask = self._per_ring_capacity - 1
            self._prod_ctxs[i].cached_read = atomic_load_acquire(&hdr.read_pos)
            self._prod_ctxs[i].cached_write = atomic_load_acquire(&hdr.write_pos)
            self._cons_ctxs[i].hdr = hdr
            self._cons_ctxs[i].data = data
            self._cons_ctxs[i].capacity = self._per_ring_capacity
            self._cons_ctxs[i].mask = self._per_ring_capacity - 1
            self._cons_ctxs[i].spin_wait = self._spin_wait

    cdef void _map_create(self, bytes path, u64 capacity, u64 nr, bint unlink_on_close, int spin_wait):
        cdef u64 per_ring = pow2_at_least(max(capacity // nr, <u64>8))
        cdef size_t total
        cdef int fd
        cdef void* base
        self._layout(nr, per_ring)
        total = _MPSC_GLOBAL_HEADER_SIZE + <size_t>nr * self._ring_stride
        fd = open(path, O_CREAT | O_RDWR | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0o600)
        if fd < 0:
            raise OSError(errno, "open failed for MPSC shared ring")
        if ftruncate(fd, <long>total) != 0:
            close(fd)
            os.unlink(path)
            raise OSError(errno, "ftruncate failed for MPSC shared ring")
        base = mmap(NULL, total, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)
        if <long>base == -1:
            close(fd)
            os.unlink(path)
            raise OSError(errno, "mmap failed for MPSC shared ring")
        self._fd = fd
        self._base = base
        self._map_len = total
        self._owner = True
        self._unlink_on_close = unlink_on_close
        self._spin_wait = spin_wait if spin_wait > 0 else 1024
        self._ghdr = <ShmMpscGlobalHeader*>base
        self._ghdr.num_rings = nr
        self._ghdr.ring_capacity = per_ring
        self._ghdr.next_producer_ring = 0
        self._contexts(True)
        atomic_store_release(&self._ghdr.magic, _MPSC_MAGIC)

    cdef void _map_attach(self, bytes path, int spin_wait):
        cdef int fd
        cdef void* base
        cdef size_t total
        cdef ShmMpscGlobalHeader* hdr
        fd = open(path, O_RDWR | O_NOFOLLOW | O_CLOEXEC, 0o600)
        if fd < 0:
            raise OSError(errno, "open failed for MPSC shared ring")
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode):
                raise RuntimeError("MPSC shared ring backing file is not a regular file")
            if stat.S_IMODE(st.st_mode) != 0o600:
                raise RuntimeError("MPSC shared ring backing file has incorrect permissions")
            if st.st_size < _MPSC_GLOBAL_HEADER_SIZE:
                raise RuntimeError("MPSC shared ring backing file too small for global header")
        except Exception:
            close(fd)
            raise
        base = mmap(NULL, _MPSC_GLOBAL_HEADER_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)
        if <long>base == -1:
            close(fd)
            raise OSError(errno, "mmap global header failed for MPSC shared ring")
        hdr = <ShmMpscGlobalHeader*>base
        try:
            if atomic_load_acquire(&hdr.magic) != _MPSC_MAGIC:
                raise RuntimeError("MPSC shared ring header mismatch")
            self._layout(hdr.num_rings, hdr.ring_capacity)
            total = _MPSC_GLOBAL_HEADER_SIZE + <size_t>self._num_rings * self._ring_stride
            if os.fstat(fd).st_size < total:
                raise RuntimeError("MPSC shared ring backing file too small for capacity")
        except Exception:
            munmap(base, _MPSC_GLOBAL_HEADER_SIZE)
            close(fd)
            raise
        munmap(base, _MPSC_GLOBAL_HEADER_SIZE)
        base = mmap(NULL, total, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)
        if <long>base == -1:
            close(fd)
            raise OSError(errno, "mmap failed for MPSC shared ring")
        self._fd = fd
        self._base = base
        self._map_len = total
        self._ghdr = <ShmMpscGlobalHeader*>base
        self._spin_wait = spin_wait if spin_wait > 0 else 1024
        self._contexts(False)

    def __dealloc__(self):
        if self._sub_hdrs != NULL:
            free(self._sub_hdrs)
        if self._sub_datas != NULL:
            free(self._sub_datas)
        if self._prod_ctxs != NULL:
            free(self._prod_ctxs)
        if self._cons_ctxs != NULL:
            free(self._cons_ctxs)

    def __len__(self) -> int:
        cdef u64 count = 0, i
        self._ensure_open()
        for i in range(self._num_rings):
            count += _count(self._sub_hdrs[i], self._sub_datas[i], self._per_ring_capacity)
        return <Py_ssize_t>count

    @property
    def num_rings(self) -> int:
        return self._num_rings

    @property
    def dropped_count(self) -> int:
        cdef u64 count = 0, i
        self._ensure_open()
        for i in range(self._num_rings):
            count += atomic_load_acquire(&self._sub_hdrs[i].dropped_count)
        return count

    @property
    def latest_insert_time_ns(self) -> int:
        cdef u64 result = 0, i, value
        self._ensure_open()
        for i in range(self._num_rings):
            value = atomic_load_acquire(&self._sub_hdrs[i].latest_insert_time_ns)
            if value > result:
                result = value
        return result

    @property
    def latest_consume_time_ns(self) -> int:
        cdef u64 result = 0, i, value
        self._ensure_open()
        for i in range(self._num_rings):
            value = atomic_load_acquire(&self._sub_hdrs[i].latest_consume_time_ns)
            if value > result:
                result = value
        return result

    cpdef bint is_empty(self):
        return len(self) == 0

    cpdef bint is_full(self):
        cdef u64 i
        self._ensure_open()
        for i in range(self._num_rings):
            if not _full(self._sub_hdrs[i], self._per_ring_capacity):
                return False
        return True


cdef class ShmMpscProducer(_MpscRing):
    """One leased shard per producer, including the creator.

    At most num_rings producer instances may be active. Exhaustion raises
    RuntimeError; close releases the shard without dropping queued records.
    Claims are deliberately not stolen from crashed processes: recreate the
    mapping after all users have stopped. Use one thread/process per instance.
    """
    cdef u64 _ring_idx
    cdef u64 _lease_owner
    cdef u64 _last_insert_dropped
    cdef u64 _failed_inserts

    def __cinit__(self, path: str, int capacity_bytes, int num_rings=0, *,
                  bint create=True, bint unlink_on_close=False, int spin_wait=1024) -> None:
        cdef u64 i, owner, start, offset
        if capacity_bytes < 0:
            raise ValueError("capacity_bytes must be >= 0")
        if num_rings < 0:
            raise ValueError("num_rings must be >= 0")
        self._path_py = path
        if create:
            self._map_create(path.encode(), capacity_bytes,
                             num_rings or (os.cpu_count() or 4), unlink_on_close, spin_wait)
        else:
            self._map_attach(path.encode(), spin_wait)
        owner = <u64>os.getpid()
        start = (atomic_add(&self._ghdr.next_producer_ring, 1) - 1) % self._num_rings
        for offset in range(self._num_rings):
            i = (start + offset) % self._num_rings
            if shm_producer_claim(self._sub_hdrs[i], owner):
                self._ring_idx = i
                self._lease_owner = owner
                # Reload after acquiring the previous writer's release lease.
                self._prod_ctxs[i].cached_read = atomic_load_acquire(&self._sub_hdrs[i].read_pos)
                self._prod_ctxs[i].cached_write = atomic_load_acquire(&self._sub_hdrs[i].write_pos)
                return
        raise RuntimeError("MPSC producer capacity exhausted: all shards are claimed")

    cdef void _release_lease(self) noexcept:
        if self._base != NULL and self._lease_owner:
            if self._lease_owner == <u64>os.getpid():
                shm_producer_release(self._prod_ctxs[self._ring_idx].hdr, self._lease_owner)
            self._lease_owner = 0

    def __dealloc__(self):
        self._release_lease()

    cdef inline void _ensure_writer(self):
        self._ensure_open()
        if self._lease_owner != <u64>os.getpid():
            raise RuntimeError("Inherited producer cannot write; attach a new producer")

    cpdef bint insert(self, bytes item):
        cdef int success
        self._ensure_writer()
        success = shm_producer_insert(&self._prod_ctxs[self._ring_idx],
            <const unsigned char*>item, len(item), &self._last_insert_dropped)
        if not success:
            self._failed_inserts += 1
        return success == 1

    cdef bint insert_char(self, const char* data, size_t n):
        cdef int success
        self._ensure_writer()
        success = shm_producer_insert(&self._prod_ctxs[self._ring_idx],
            <const unsigned char*>data, n, &self._last_insert_dropped)
        if not success:
            self._failed_inserts += 1
        return success == 1

    cpdef bint insert_batch(self, list[bytes] items):
        cdef bint success
        self._ensure_writer()
        success = _insert_batch(&self._prod_ctxs[self._ring_idx], items, &self._last_insert_dropped)
        if not success:
            self._failed_inserts += 1
        return success

    @property
    def last_insert_dropped(self) -> int:
        return self._last_insert_dropped

    @property
    def failed_inserts(self) -> int:
        return self._failed_inserts


cdef class ShmMpscConsumer(_MpscRing):
    """Poll the shards fairly and expose only validated copies."""
    cdef u64 _next_ring
    cdef bytearray _scratch

    def __cinit__(self, path: str, *, int spin_wait=1024) -> None:
        self._path_py = path
        self._map_attach(path.encode(), spin_wait)
        self._scratch = bytearray()

    cdef object _try_any(self, Py_ssize_t limit=-1):
        cdef u64 i, ring
        cdef object item
        self._ensure_open()
        for i in range(self._num_rings):
            ring = (self._next_ring + i) % self._num_rings
            item = _try_consume(&self._cons_ctxs[ring], limit)
            if item is not None:
                self._next_ring = (ring + 1) % self._num_rings
                return item
        return None

    cpdef bytes consume(self):
        cdef object item
        cdef int spins = 0
        while True:
            item = self._try_any()
            if item is not None:
                return item
            spins += 1
            if spins >= self._spin_wait:
                _sched_yield()
                spins = 0

    cdef int _try_any_into(self, bytearray dst):
        cdef u64 i, ring
        cdef int copied
        self._ensure_open()
        for i in range(self._num_rings):
            ring = (self._next_ring + i) % self._num_rings
            copied = _try_consume_into(&self._cons_ctxs[ring], dst, self._scratch)
            if copied >= 0:
                self._next_ring = (ring + 1) % self._num_rings
                return copied
        return -1

    cpdef int consume_into(self, bytearray dst):
        cdef int copied, spins = 0
        while True:
            copied = self._try_any_into(dst)
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
            copied = self._try_any_into(dst)
            if copied < 0:
                break
            total += 1
        return total

    cpdef list consume_all(self):
        cdef list items = []
        cdef object item
        while True:
            item = self._try_any()
            if item is None:
                return items
            items.append(item)

    cpdef list unwrapped(self):
        """Validated snapshot per shard; shards have independent order."""
        cdef list result = []
        cdef u64 i, ring
        self._ensure_open()
        for i in range(self._num_rings):
            ring = (self._next_ring + i) % self._num_rings
            result.extend(_snapshot(&self._cons_ctxs[ring]))
        return result

    cpdef object peekleft(self):
        cdef list items = self.unwrapped()
        return items[0] if items else None

    cpdef object peekright(self):
        cdef u64 i, ring
        cdef list items
        self._ensure_open()
        for i in range(self._num_rings):
            ring = (self._next_ring + i) % self._num_rings
            items = _snapshot(&self._cons_ctxs[ring])
            if items:
                return items[len(items) - 1]
        return None

    cpdef bint contains(self, bytes item):
        return item in self.unwrapped()

    def __contains__(self, bytes item):
        return self.contains(item)

    def consume_iterable(self):
        while True:
            yield self.consume()

    async def aconsume(self):
        import asyncio
        cdef object item
        while True:
            item = self._try_any()
            if item is not None:
                return item
            await asyncio.sleep(0)

    async def aconsume_iterable(self):
        while True:
            yield await self.aconsume()

    cpdef void clear(self):
        self.consume_all()
