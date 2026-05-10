# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

"""MPSC shared-memory bytes ring buffers."""

import ctypes
import os
import stat
from libc.stdint cimport uint64_t as u64
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

from ._shm cimport _ShmRingBase, _MPSC_MAGIC, _MPSC_GLOBAL_HEADER_SIZE, _SUB_RING_HEADER_SIZE, _HEADER_ALIGN
from .atomics cimport atomic_add, atomic_load_acquire, atomic_store_release, atomic_sub
from .header cimport ShmMpscGlobalHeader, ShmSubRingHeader
from .memory cimport (
    align_up,
    copy_from_ring,
    copy_into_ring,
    pow2_at_least,
    read_u64_le,
    write_u64_le,
)


cdef class ShmMpscProducer(_ShmRingBase):
    """Shared-memory MPSC producer for bytes payloads.

    Uses a sharded sub-ring architecture where each sub-ring is a complete
    SPSC queue.  Producers pick a sub-ring round-robin via an atomic counter.
    """

    cdef ShmMpscGlobalHeader* _ghdr
    cdef u64 _num_rings
    cdef u64 _per_ring_capacity
    cdef u64 _per_ring_mask
    cdef size_t _ring_stride
    cdef ShmSubRingHeader** _sub_hdrs
    cdef unsigned char** _sub_datas
    cdef ShmProducerContext* _prod_ctxs
    cdef u64 _ring_idx

    cdef void _map_create(self, bytes path_b, u64 capacity_bytes, u64 num_rings, bint unlink_on_close, int spin_wait):
        """Create and initialize MPSC shared ring."""
        cdef:
            size_t per_ring = pow2_at_least(capacity_bytes // num_rings)
            size_t total_len = _MPSC_GLOBAL_HEADER_SIZE + num_rings * self._ring_stride
            int fd
            void* base
            u64 i
            unsigned char* sub_data
            ShmSubRingHeader* sub_hdr

        if per_ring < 1:
            per_ring = 1
        self._per_ring_capacity = <u64>per_ring
        self._per_ring_mask = <u64>(per_ring - 1)
        self._ring_stride = align_up(_SUB_RING_HEADER_SIZE + per_ring, _HEADER_ALIGN)

        total_len = _MPSC_GLOBAL_HEADER_SIZE + num_rings * self._ring_stride

        self._sub_hdrs = <ShmSubRingHeader**>malloc(num_rings * sizeof(ShmSubRingHeader*))
        self._sub_datas = <unsigned char**>malloc(num_rings * sizeof(unsigned char*))
        self._prod_ctxs = <ShmProducerContext*>malloc(num_rings * sizeof(ShmProducerContext))
        if self._sub_hdrs == NULL or self._sub_datas == NULL or self._prod_ctxs == NULL:
            raise MemoryError("Failed to allocate MPSC sub-ring arrays")

        fd = open(path_b, O_CREAT | O_RDWR | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0o600)
        if fd < 0:
            raise OSError(errno, "open failed for MPSC shared ring")
        if ftruncate(fd, <long long>total_len) != 0:
            close(fd)
            raise OSError(errno, "ftruncate failed for MPSC shared ring")
        os.fsync(fd)
        base = mmap(NULL, total_len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)
        if <long>base == -1:
            close(fd)
            try:
                os.unlink(path_b)
            except Exception:
                pass
            finally:
                raise OSError(errno, "mmap failed for MPSC shared ring")

        self._fd = fd
        self._map_len = total_len
        self._base = base
        self._ghdr = <ShmMpscGlobalHeader*>base
        self._owner = True
        self._unlink_on_close = unlink_on_close
        self._spin_wait = spin_wait if spin_wait > 0 else 1024

        self._ghdr.magic = _MPSC_MAGIC
        self._ghdr.num_rings = num_rings
        self._ghdr.ring_capacity = <u64>per_ring
        self._ghdr.next_producer_ring = 0
        for i in range(num_rings):
            sub_hdr = <ShmSubRingHeader*>((<unsigned char*>base) + _MPSC_GLOBAL_HEADER_SIZE + i * self._ring_stride)
            sub_data = <unsigned char*>sub_hdr + _SUB_RING_HEADER_SIZE
            self._sub_hdrs[i] = sub_hdr
            self._sub_datas[i] = sub_data

            sub_hdr.write_pos = 0
            sub_hdr.read_pos = 0
            sub_hdr.msg_count = 0
            sub_hdr.latest_insert_time_ns = 0
            sub_hdr.latest_consume_time_ns = 0

            self._prod_ctxs[i].hdr = <ShmHeader*>sub_hdr
            self._prod_ctxs[i].data = sub_data
            self._prod_ctxs[i].capacity = self._per_ring_capacity
            self._prod_ctxs[i].mask = self._per_ring_mask
            self._prod_ctxs[i].cached_read = 0
            self._prod_ctxs[i].cached_write = 0

    cdef void _map_attach(self, bytes path_b, int spin_wait):
        """Attach to an existing MPSC shared ring."""
        cdef:
            int fd
            void* base
            size_t total_len
            u64 nr
            size_t per_ring
            object backing_len
            ShmMpscGlobalHeader* tmp_ghdr
            u64 i
            unsigned char* sub_data
            ShmSubRingHeader* sub_hdr

        fd = open(path_b, O_RDWR | O_NOFOLLOW | O_CLOEXEC, 0o600)
        if fd < 0:
            raise OSError(errno, "open failed for MPSC shared ring")
        try:
            st = os.fstat(fd)
            backing_len = st.st_size
        except Exception:
            close(fd)
            raise
        if not stat.S_ISREG(st.st_mode):
            close(fd)
            raise RuntimeError("MPSC shared ring backing file is not a regular file")
        if stat.S_IMODE(st.st_mode) != 0o600:
            close(fd)
            raise RuntimeError("MPSC shared ring backing file has incorrect permissions")
        if backing_len < _MPSC_GLOBAL_HEADER_SIZE:
            close(fd)
            raise RuntimeError("MPSC shared ring backing file too small for global header")

        base = mmap(NULL, _MPSC_GLOBAL_HEADER_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)
        if <long>base == -1:
            close(fd)
            raise OSError(errno, "mmap global header failed for MPSC shared ring")
        tmp_ghdr = <ShmMpscGlobalHeader*>base
        if tmp_ghdr.magic != _MPSC_MAGIC:
            munmap(base, _MPSC_GLOBAL_HEADER_SIZE)
            close(fd)
            raise RuntimeError("MPSC shared ring header mismatch")
        nr = tmp_ghdr.num_rings
        per_ring = <size_t>tmp_ghdr.ring_capacity
        if nr == 0 or per_ring == 0 or (per_ring & (per_ring - 1)) != 0:
            munmap(base, _MPSC_GLOBAL_HEADER_SIZE)
            close(fd)
            raise RuntimeError("MPSC shared ring header invalid")
        self._num_rings = nr
        self._per_ring_capacity = <u64>per_ring
        self._per_ring_mask = <u64>(per_ring - 1)
        self._ring_stride = align_up(_SUB_RING_HEADER_SIZE + per_ring, _HEADER_ALIGN)
        munmap(base, _MPSC_GLOBAL_HEADER_SIZE)

        total_len = _MPSC_GLOBAL_HEADER_SIZE + nr * self._ring_stride
        try:
            backing_len = os.fstat(fd).st_size
        except Exception:
            close(fd)
            raise
        if backing_len < total_len:
            close(fd)
            raise RuntimeError("MPSC shared ring backing file too small for capacity")

        base = mmap(NULL, total_len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)
        if <long>base == -1:
            close(fd)
            raise OSError(errno, "mmap failed for MPSC shared ring")

        self._fd = fd
        self._map_len = total_len
        self._base = base
        self._ghdr = <ShmMpscGlobalHeader*>base
        self._owner = False
        self._unlink_on_close = False
        self._spin_wait = spin_wait if spin_wait > 0 else 1024

        self._sub_hdrs = <ShmSubRingHeader**>malloc(nr * sizeof(ShmSubRingHeader*))
        self._sub_datas = <unsigned char**>malloc(nr * sizeof(unsigned char*))
        self._prod_ctxs = <ShmProducerContext*>malloc(nr * sizeof(ShmProducerContext))
        if self._sub_hdrs == NULL or self._sub_datas == NULL or self._prod_ctxs == NULL:
            raise MemoryError("Failed to allocate MPSC sub-ring arrays")

        for i in range(nr):
            sub_hdr = <ShmSubRingHeader*>((<unsigned char*>base) + _MPSC_GLOBAL_HEADER_SIZE + i * self._ring_stride)
            sub_data = <unsigned char*>sub_hdr + _SUB_RING_HEADER_SIZE
            self._sub_hdrs[i] = sub_hdr
            self._sub_datas[i] = sub_data

            self._prod_ctxs[i].hdr = <ShmHeader*>sub_hdr
            self._prod_ctxs[i].data = sub_data
            self._prod_ctxs[i].capacity = self._per_ring_capacity
            self._prod_ctxs[i].mask = self._per_ring_mask
            self._prod_ctxs[i].cached_read = sub_hdr.read_pos
            self._prod_ctxs[i].cached_write = sub_hdr.write_pos

    def __cinit__(
        self,
        path: str,
        int capacity_bytes,
        int num_rings=0,
        *,
        bint create=True,
        bint unlink_on_close=False,
        int spin_wait=1024,
    ) -> None:
        cdef u64 nr

        self._path_py = path
        path_b = (<str>path).encode()
        self._path = path_b
        self._ghdr = NULL
        self._num_rings = 0
        self._per_ring_capacity = 0
        self._per_ring_mask = 0
        self._ring_stride = 0
        self._sub_hdrs = NULL
        self._sub_datas = NULL
        self._prod_ctxs = NULL

        if num_rings < 0:
            raise ValueError("num_rings must be >= 0")
        nr = <u64>num_rings
        if nr == 0:
            import os as _os
            nr = <u64>(_os.cpu_count() or 4)
        self._num_rings = nr

        if create:
            self._map_create(path_b, <u64>capacity_bytes, nr, unlink_on_close, spin_wait)
        else:
            self._map_attach(path_b, spin_wait)

        self._ring_idx = self._pick_ring()

    def __dealloc__(self):
        if self._sub_hdrs != NULL:
            free(self._sub_hdrs)
            self._sub_hdrs = NULL
        if self._sub_datas != NULL:
            free(self._sub_datas)
            self._sub_datas = NULL
        if self._prod_ctxs != NULL:
            free(self._prod_ctxs)
            self._prod_ctxs = NULL

    cdef inline u64 _pick_ring(self) nogil:
        cdef u64 idx = atomic_add(&self._ghdr.next_producer_ring, 1)
        return idx % self._num_rings

    cpdef bint insert(self, bytes item):
        """Insert a single item into a round-robin sub-ring.

        Args:
            item: Bytes payload to insert.

        Returns:
            True if insert succeeded, False if message too large.
        """
        cdef:
            Py_ssize_t payload_len = len(item)
            const unsigned char* payload_ptr = <const unsigned char*>item
            u64 dropped = 0
            int success

        if payload_len < 0:
            return False

        with nogil:
            success = shm_producer_insert(&self._prod_ctxs[self._ring_idx], payload_ptr,
                                         <size_t>payload_len, &dropped)
        return success == 1

    cdef bint insert_char(self, const char* data, size_t n):
        """Insert a raw buffer of length n from a char* pointer.

        Args:
            data: Pointer to raw buffer.
            n: Length of buffer in bytes.

        Returns:
            True if insert succeeded, False if message too large.
        """
        cdef u64 dropped = 0
        cdef int success

        if n == 0:
            return True

        with nogil:
            success = shm_producer_insert(&self._prod_ctxs[self._ring_idx], <const unsigned char*>data,
                                         n, &dropped)
        return success == 1

    cpdef bint insert_batch(self, list[bytes] items):
        """Insert multiple items into a single sub-ring with one commit.

        Args:
            items: List of bytes payloads to insert.

        Returns:
            True if all items inserted, False if total size exceeds capacity.
        """
        cdef:
            u64 ring_idx = self._ring_idx
            Py_ssize_t i, n = len(items)
            u64 capacity = self._per_ring_capacity
            u64 mask = self._per_ring_mask
            u64 write_pos = self._prod_ctxs[ring_idx].cached_write
            size_t total = 0
            bytes it
            u64 msg_len
            u64 dropped = 0
            u64 now_ns
            ShmSubRingHeader* sub_hdr = self._sub_hdrs[ring_idx]
            unsigned char* sub_data = self._sub_datas[ring_idx]
            u64 read_pos
            u64 free_bytes
            u64 dropped_pos

        if n == 0:
            return True
        for i in range(n):
            it = items[i]
            msg_len = <u64>len(it)
            if <u64>(8) + msg_len > capacity:
                return False
            total += <size_t>(8 + msg_len)
        if <u64>total > capacity:
            return False

        with nogil:
            read_pos = atomic_load_acquire(&sub_hdr.read_pos)
        self._prod_ctxs[ring_idx].cached_read = read_pos
        free_bytes = capacity - (write_pos - read_pos)
        if free_bytes < <u64>total:
            dropped_pos = read_pos
            while True:
                msg_len = read_u64_le(sub_data, dropped_pos & mask, mask)
                if msg_len > capacity or (8 + msg_len) > capacity:
                    return False
                dropped_pos += 8 + msg_len
                dropped += 1
                free_bytes = capacity - (write_pos - dropped_pos)
                if free_bytes >= <u64>total:
                    with nogil:
                        atomic_store_release(&sub_hdr.read_pos, dropped_pos)
                        if dropped:
                            atomic_sub(&sub_hdr.msg_count, dropped)
                    self._prod_ctxs[ring_idx].cached_read = dropped_pos
                    break

        for i in range(n):
            it = items[i]
            msg_len = <u64>len(it)
            write_u64_le(sub_data, write_pos & mask, mask, msg_len)
            copy_into_ring(sub_data, write_pos + 8, mask, <const unsigned char*>it, <size_t>msg_len, capacity)
            write_pos += 8 + msg_len
        with nogil:
            now_ns = <u64>c_time_monotonic_ns()
            atomic_store_release(&sub_hdr.write_pos, write_pos)
            atomic_add(&sub_hdr.msg_count, <u64>n)
            atomic_store_release(&sub_hdr.latest_insert_time_ns, now_ns)
        self._prod_ctxs[ring_idx].cached_write = write_pos
        return True

    def __len__(self) -> int:
        cdef u64 count = 0
        cdef u64 i
        with nogil:
            for i in range(self._num_rings):
                count += atomic_load_acquire(&self._sub_hdrs[i].msg_count)
        return <Py_ssize_t>count

    @property
    def latest_insert_time_ns(self) -> int:
        cdef u64 ts = 0
        cdef u64 t
        cdef u64 i
        with nogil:
            for i in range(self._num_rings):
                t = atomic_load_acquire(&self._sub_hdrs[i].latest_insert_time_ns)
                if t > ts:
                    ts = t
        return ts

    @property
    def latest_consume_time_ns(self) -> int:
        cdef u64 ts = 0
        cdef u64 t
        cdef u64 i
        with nogil:
            for i in range(self._num_rings):
                t = atomic_load_acquire(&self._sub_hdrs[i].latest_consume_time_ns)
                if t > ts:
                    ts = t
        return ts

    @property
    def num_rings(self) -> int:
        return <Py_ssize_t>self._num_rings

    cpdef bint is_empty(self):
        """Check if the buffer is empty.

        Returns:
            True if all sub-rings are empty.
        """
        return len(self) == 0

    cpdef bint is_full(self):
        """Check if the buffer is full.

        Returns:
            True if no sub-ring has space for even a 0-byte message.
        """
        cdef u64 read_pos
        cdef u64 i
        cdef u64 free_bytes
        for i in range(self._num_rings):
            with nogil:
                read_pos = atomic_load_acquire(&self._sub_hdrs[i].read_pos)
            self._prod_ctxs[i].cached_read = read_pos
            free_bytes = self._per_ring_capacity - (self._prod_ctxs[i].cached_write - read_pos)
            if free_bytes >= 8:
                return False
        return True


cdef class ShmMpscConsumer(_ShmRingBase):
    """Shared-memory MPSC consumer for bytes payloads.

    Round-robin polls all sub-rings starting from a local index to ensure
    fairness and prevent ring starvation.
    """

    cdef ShmMpscGlobalHeader* _ghdr
    cdef u64 _num_rings
    cdef u64 _per_ring_capacity
    cdef u64 _per_ring_mask
    cdef size_t _ring_stride
    cdef ShmSubRingHeader** _sub_hdrs
    cdef unsigned char** _sub_datas
    cdef ShmConsumerContext* _cons_ctxs
    cdef u64 _next_ring

    cdef void _map_attach(self, bytes path_b, int spin_wait):
        """Attach to an existing MPSC shared ring."""
        cdef:
            u64 nr
            size_t per_ring
            size_t total_len
            int fd
            void* base
            u64 i
            unsigned char* sub_data
            ShmSubRingHeader* sub_hdr
            object backing_len
            ShmMpscGlobalHeader* tmp_ghdr

        fd = open(path_b, O_RDWR, 0o600)
        if fd < 0:
            raise OSError(errno, "open failed for MPSC shared ring")
        try:
            backing_len = os.fstat(fd).st_size
        except Exception:
            close(fd)
            raise
        if backing_len < _MPSC_GLOBAL_HEADER_SIZE:
            close(fd)
            raise RuntimeError("MPSC shared ring backing file too small for global header")

        base = mmap(NULL, _MPSC_GLOBAL_HEADER_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)
        if <long>base == -1:
            close(fd)
            raise OSError(errno, "mmap global header failed for MPSC shared ring")
        tmp_ghdr = <ShmMpscGlobalHeader*>base
        if tmp_ghdr.magic != _MPSC_MAGIC:
            munmap(base, _MPSC_GLOBAL_HEADER_SIZE)
            close(fd)
            raise RuntimeError("MPSC shared ring header mismatch")
        nr = tmp_ghdr.num_rings
        per_ring = <size_t>tmp_ghdr.ring_capacity
        if nr == 0 or per_ring == 0 or (per_ring & (per_ring - 1)) != 0:
            munmap(base, _MPSC_GLOBAL_HEADER_SIZE)
            close(fd)
            raise RuntimeError("MPSC shared ring header invalid")
        self._num_rings = nr
        self._per_ring_capacity = <u64>per_ring
        self._per_ring_mask = <u64>(per_ring - 1)
        self._ring_stride = align_up(_SUB_RING_HEADER_SIZE + per_ring, _HEADER_ALIGN)
        munmap(base, _MPSC_GLOBAL_HEADER_SIZE)

        total_len = _MPSC_GLOBAL_HEADER_SIZE + nr * self._ring_stride
        try:
            backing_len = os.fstat(fd).st_size
        except Exception:
            close(fd)
            raise
        if backing_len < total_len:
            close(fd)
            raise RuntimeError("MPSC shared ring backing file too small for capacity")

        base = mmap(NULL, total_len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)
        if <long>base == -1:
            close(fd)
            raise OSError(errno, "mmap failed for MPSC shared ring")

        self._fd = fd
        self._map_len = total_len
        self._base = base
        self._ghdr = <ShmMpscGlobalHeader*>base
        self._owner = False
        self._unlink_on_close = False
        self._spin_wait = spin_wait if spin_wait > 0 else 1024

        self._sub_hdrs = <ShmSubRingHeader**>malloc(nr * sizeof(ShmSubRingHeader*))
        self._sub_datas = <unsigned char**>malloc(nr * sizeof(unsigned char*))
        self._cons_ctxs = <ShmConsumerContext*>malloc(nr * sizeof(ShmConsumerContext))
        if self._sub_hdrs == NULL or self._sub_datas == NULL or self._cons_ctxs == NULL:
            raise MemoryError("Failed to allocate MPSC sub-ring arrays")

        for i in range(nr):
            sub_hdr = <ShmSubRingHeader*>((<unsigned char*>base) + _MPSC_GLOBAL_HEADER_SIZE + i * self._ring_stride)
            sub_data = <unsigned char*>sub_hdr + _SUB_RING_HEADER_SIZE
            self._sub_hdrs[i] = sub_hdr
            self._sub_datas[i] = sub_data

            self._cons_ctxs[i].hdr = <ShmHeader*>sub_hdr
            self._cons_ctxs[i].data = sub_data
            self._cons_ctxs[i].capacity = self._per_ring_capacity
            self._cons_ctxs[i].mask = self._per_ring_mask
            self._cons_ctxs[i].spin_wait = self._spin_wait

    def __cinit__(self, path: str, *, int spin_wait=1024) -> None:
        self._path_py = path
        path_b = (<str>path).encode()
        self._path = path_b
        self._next_ring = 0
        self._ghdr = NULL
        self._num_rings = 0
        self._per_ring_capacity = 0
        self._per_ring_mask = 0
        self._ring_stride = 0
        self._sub_hdrs = NULL
        self._sub_datas = NULL
        self._cons_ctxs = NULL

        self._map_attach(path_b, spin_wait)

    def __dealloc__(self):
        if self._sub_hdrs != NULL:
            free(self._sub_hdrs)
            self._sub_hdrs = NULL
        if self._sub_datas != NULL:
            free(self._sub_datas)
            self._sub_datas = NULL
        if self._cons_ctxs != NULL:
            free(self._cons_ctxs)
            self._cons_ctxs = NULL

    cpdef bytes consume(self):
        """Consume a single item, blocking until available.

        Returns:
            The next available bytes payload.
        """
        cdef:
            u64 msg_len = 0
            u64 read_pos = 0
            u64 read_pos_check = 0
            int spin_count = 0
            int available = 0
            bytearray buf
            unsigned char* buf_ptr
            u64 ring_idx
            u64 start_ring
            u64 i

        while True:
            start_ring = self._next_ring
            for i in range(self._num_rings):
                ring_idx = (start_ring + i) % self._num_rings
                with nogil:
                    available = shm_consumer_peek_available(&self._cons_ctxs[ring_idx], &msg_len, &read_pos)
                if available:
                    with nogil:
                        read_pos_check = atomic_load_acquire(&self._sub_hdrs[ring_idx].read_pos)
                    if read_pos_check == read_pos:
                        buf = bytearray(<Py_ssize_t>msg_len)
                        buf_ptr = <unsigned char*>buf
                        with nogil:
                            shm_consumer_consume(&self._cons_ctxs[ring_idx], buf_ptr, msg_len, read_pos)
                        self._next_ring = (ring_idx + 1) % self._num_rings
                        return bytes(buf)
            spin_count += 1
            if spin_count < self._spin_wait:
                continue
            _sched_yield()
            spin_count = 0

    cpdef object peekleft(self):
        """Peek at the next item without consuming; returns None if empty.

        Returns:
            The next bytes payload or None if buffer is empty.
        """
        cdef u64 msg_len = 0
        cdef u64 read_pos = 0
        cdef u64 read_pos_check = 0
        cdef int available = 0
        cdef u64 ring_idx
        cdef u64 i
        cdef u64 mask
        cdef u64 cap
        cdef bytes out

        for i in range(self._num_rings):
            ring_idx = (self._next_ring + i) % self._num_rings
            mask = self._per_ring_mask
            cap = self._per_ring_capacity
            with nogil:
                available = shm_consumer_peek_available(&self._cons_ctxs[ring_idx], &msg_len, &read_pos)
                if available:
                    read_pos_check = atomic_load_acquire(&self._sub_hdrs[ring_idx].read_pos)
            if not available:
                continue
            if read_pos_check != read_pos:
                continue
            out = bytes(<Py_ssize_t>msg_len)
            shm_copy_from_ring(<unsigned char*>out, self._sub_datas[ring_idx], read_pos + 8, mask, <size_t>msg_len, cap)
            return out
        return None

    cpdef object peekright(self):
        """Peek at the most recently inserted item; returns None if empty."""
        cdef u64 w
        cdef u64 r
        cdef u64 msg_len
        cdef u64 pos
        cdef u64 next_pos
        cdef u64 mask
        cdef u64 cap
        cdef bytes out
        cdef u64 ring_idx
        cdef u64 i

        for i in range(self._num_rings):
            ring_idx = (self._next_ring + i) % self._num_rings
            mask = self._per_ring_mask
            cap = self._per_ring_capacity
            with nogil:
                w = atomic_load_acquire(&self._sub_hdrs[ring_idx].write_pos)
                r = atomic_load_acquire(&self._sub_hdrs[ring_idx].read_pos)
            if w - r < 8:
                continue
            pos = r
            while pos < w:
                msg_len = read_u64_le(self._sub_datas[ring_idx], pos & mask, mask)
                if msg_len > cap or 8 + msg_len > cap:
                    return None
                next_pos = pos + 8 + msg_len
                if next_pos > w:
                    return None
                if next_pos == w:
                    out = bytes(<Py_ssize_t>msg_len)
                    copy_from_ring(<unsigned char*>out, self._sub_datas[ring_idx], pos + 8, mask, <size_t>msg_len, cap)
                    return out
                pos = next_pos
        return None

    cpdef list consume_all(self):
        """Drain all items currently available without blocking.

        Returns:
            List of all available bytes payloads (may be empty).
        """
        cdef list res = []
        cdef u64 msg_len = 0
        cdef u64 read_pos = 0
        cdef u64 read_pos_check = 0
        cdef int available = 0
        cdef bytearray buf
        cdef unsigned char* buf_ptr
        cdef u64 ring_idx
        cdef bint found = True
        cdef u64 i

        while found:
            found = False
            for i in range(self._num_rings):
                ring_idx = (self._next_ring + i) % self._num_rings
                with nogil:
                    available = shm_consumer_peek_available(&self._cons_ctxs[ring_idx], &msg_len, &read_pos)
                    if available:
                        read_pos_check = atomic_load_acquire(&self._sub_hdrs[ring_idx].read_pos)
                if available and read_pos_check == read_pos:
                    buf = bytearray(<Py_ssize_t>msg_len)
                    buf_ptr = <unsigned char*>buf
                    with nogil:
                        shm_consumer_consume(&self._cons_ctxs[ring_idx], buf_ptr, msg_len, read_pos)
                    res.append(bytes(buf))
                    self._next_ring = (ring_idx + 1) % self._num_rings
                    found = True
                    break
        return res

    cpdef int consume_into(self, bytearray dst):
        """Consume a single item into the provided bytearray.

        Blocks until a message is available.

        Args:
            dst: Pre-allocated bytearray to copy the message into.

        Returns:
            Number of bytes copied.

        Raises:
            ValueError: If dst is too small to hold the message.
        """
        cdef:
            u64 msg_len = 0
            u64 read_pos = 0
            u64 read_pos_check = 0
            int spin_count = 0
            int available = 0
            unsigned char* buf_ptr
            u64 ring_idx
            u64 start_ring
            u64 i

        while True:
            start_ring = self._next_ring
            for i in range(self._num_rings):
                ring_idx = (start_ring + i) % self._num_rings
                with nogil:
                    available = shm_consumer_peek_available(
                        &self._cons_ctxs[ring_idx], &msg_len, &read_pos
                    )
                if available:
                    with nogil:
                        read_pos_check = atomic_load_acquire(
                            &self._sub_hdrs[ring_idx].read_pos
                        )
                    if read_pos_check == read_pos:
                        if msg_len > <u64>len(dst):
                            raise ValueError(
                                f"Message size {msg_len} exceeds buffer size {len(dst)}"
                            )
                        buf_ptr = <unsigned char*>dst
                        with nogil:
                            shm_consumer_consume(
                                &self._cons_ctxs[ring_idx], buf_ptr, msg_len, read_pos
                            )
                        self._next_ring = (ring_idx + 1) % self._num_rings
                        return <int>msg_len
            spin_count += 1
            if spin_count < self._spin_wait:
                continue
            _sched_yield()
            spin_count = 0

    cpdef int consume_all_into(self, list[bytearray] buffers):
        """Consume all available items into the provided bytearrays.

        Does not block. Copies as many available messages as there are
        buffers provided.

        Args:
            buffers: List of pre-allocated bytearrays.

        Returns:
            Number of messages copied.

        Raises:
            ValueError: If any buffer is too small for its message.
        """
        cdef:
            int total_copied = 0
            int n = len(buffers)
            int i_buf
            u64 msg_len = 0
            u64 read_pos = 0
            u64 read_pos_check = 0
            int available = 0
            bytearray buf
            unsigned char* buf_ptr
            u64 ring_idx
            bint found = True
            u64 i

        for i_buf in range(n):
            found = False
            for i in range(self._num_rings):
                ring_idx = (self._next_ring + i) % self._num_rings
                with nogil:
                    available = shm_consumer_peek_available(
                        &self._cons_ctxs[ring_idx], &msg_len, &read_pos
                    )
                    if available:
                        read_pos_check = atomic_load_acquire(
                            &self._sub_hdrs[ring_idx].read_pos
                        )
                if not available:
                    continue
                if read_pos_check != read_pos:
                    continue
                buf = buffers[i_buf]
                if msg_len > <u64>len(buf):
                    raise ValueError(
                        f"Message size {msg_len} exceeds buffer size {len(buf)}"
                    )
                buf_ptr = <unsigned char*>buf
                with nogil:
                    shm_consumer_consume(
                        &self._cons_ctxs[ring_idx], buf_ptr, msg_len, read_pos
                    )
                self._next_ring = (ring_idx + 1) % self._num_rings
                total_copied += 1
                found = True
                break
            if not found:
                break
        return total_copied

    def __len__(self) -> int:
        cdef u64 count = 0
        cdef u64 i
        with nogil:
            for i in range(self._num_rings):
                count += atomic_load_acquire(&self._sub_hdrs[i].msg_count)
        return <Py_ssize_t>count

    @property
    def latest_insert_time_ns(self) -> int:
        cdef u64 ts = 0
        cdef u64 t
        cdef u64 i
        with nogil:
            for i in range(self._num_rings):
                t = atomic_load_acquire(&self._sub_hdrs[i].latest_insert_time_ns)
                if t > ts:
                    ts = t
        return ts

    @property
    def latest_consume_time_ns(self) -> int:
        cdef u64 ts = 0
        cdef u64 t
        cdef u64 i
        with nogil:
            for i in range(self._num_rings):
                t = atomic_load_acquire(&self._sub_hdrs[i].latest_consume_time_ns)
                if t > ts:
                    ts = t
        return ts

    @property
    def num_rings(self) -> int:
        return <Py_ssize_t>self._num_rings

    def consume_iterable(self):
        """Iterate over items, blocking until each is available."""
        while True:
            yield self.consume()

    async def aconsume(self):
        """Async consume a single item."""
        import asyncio
        return self.consume()

    async def aconsume_iterable(self):
        """Async iterator over consumed items."""
        while True:
            yield await self.aconsume()

    cpdef list unwrapped(self):
        """Return a list of all logical contents without consuming."""
        cdef list res = []
        cdef u64 msg_len = 0
        cdef u64 pos = 0
        cdef u64 next_pos = 0
        cdef u64 w = 0
        cdef u64 r = 0
        cdef u64 mask = self._per_ring_mask
        cdef u64 cap = self._per_ring_capacity
        cdef bytes out
        cdef u64 ring_idx
        cdef u64 i

        for i in range(self._num_rings):
            ring_idx = (self._next_ring + i) % self._num_rings
            with nogil:
                w = atomic_load_acquire(&self._sub_hdrs[ring_idx].write_pos)
                r = atomic_load_acquire(&self._sub_hdrs[ring_idx].read_pos)

            pos = r
            while pos < w:
                msg_len = read_u64_le(self._sub_datas[ring_idx], pos & mask, mask)
                if msg_len > cap or 8 + msg_len > cap:
                    break
                next_pos = pos + 8 + msg_len
                if next_pos > w:
                    break
                out = bytes(<Py_ssize_t>msg_len)
                copy_from_ring(<unsigned char*>out, self._sub_datas[ring_idx], pos + 8, mask, <size_t>msg_len, cap)
                res.append(out)
                pos = next_pos
        return res

    cpdef bint contains(self, bytes item):
        """Check if item is present in the buffer."""
        cdef bytes b
        for b in self.unwrapped():
            if b == item:
                return True
        return False

    def __contains__(self, bytes item):
        """Delegate to contains()."""
        return self.contains(item)

    cpdef bint is_empty(self):
        """Check if the buffer is empty."""
        return len(self) == 0

    cpdef bint is_full(self):
        """Check if the buffer is full (no space for even a 0-byte message in any sub-ring)."""
        cdef u64 read_pos
        cdef u64 write_pos
        cdef u64 i
        for i in range(self._num_rings):
            with nogil:
                read_pos = atomic_load_acquire(&self._sub_hdrs[i].read_pos)
                write_pos = atomic_load_acquire(&self._sub_hdrs[i].write_pos)
            if (self._per_ring_capacity - (write_pos - read_pos)) >= 8:
                return False
        return True

    cpdef void clear(self):
        """Drain all available items from the buffer."""
        self.consume_all()
