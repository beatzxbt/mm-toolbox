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

from ._shm cimport _ShmRingBase, _MAGIC, _HEADER_SIZE, _HEADER_ALIGN
from .atomics cimport atomic_add, atomic_load_acquire, atomic_store_release, atomic_sub
from .memory cimport (
    align_up,
    copy_from_ring,
    copy_into_ring,
    pow2_at_least,
    read_u64_le,
    write_u64_le,
)


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
            size_t cap = <size_t>pow2_at_least(capacity_bytes if capacity_bytes > 0 else 1)
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

        self._hdr.magic = _MAGIC
        self._hdr.capacity = <u64>cap
        self._hdr.mask = <u64>(cap - 1)
        self._hdr.write_pos = 0
        self._hdr.read_pos = 0
        self._hdr.msg_count = 0
        self._hdr.latest_insert_time_ns = 0
        self._hdr.latest_consume_time_ns = 0

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
        if tmp_hdr.magic != _MAGIC:
            munmap(base, _HEADER_SIZE)
            close(fd)
            raise RuntimeError("Shared ring header mismatch")
        capacity = tmp_hdr.capacity
        mask = tmp_hdr.mask
        if capacity == 0 or (capacity & (capacity - 1)) != 0 or mask != capacity - 1:
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
        self._cached_read = self._hdr.read_pos
        self._cached_write = self._hdr.write_pos

    def __len__(self) -> int:
        cdef u64 count
        with nogil:
            count = atomic_load_acquire(&self._hdr.msg_count)
        return <Py_ssize_t>count

    @property
    def latest_insert_time_ns(self) -> int:
        cdef u64 ts
        with nogil:
            ts = atomic_load_acquire(&self._hdr.latest_insert_time_ns)
        return ts

    @property
    def latest_consume_time_ns(self) -> int:
        cdef u64 ts
        with nogil:
            ts = atomic_load_acquire(&self._hdr.latest_consume_time_ns)
        return ts


cdef class ShmSpscProducer(_SharedBytesRing):
    """Shared-memory SPSC producer for bytes payloads.

    Inserts variable-length byte messages into a shared-memory ring buffer
    with atomic coordination via the header.
    """

    cdef ShmProducerContext _prod_ctx

    cdef inline void _init_prod_ctx(self):
        """Initialize producer context for C functions."""
        self._prod_ctx.hdr = self._hdr
        self._prod_ctx.data = self._data
        self._prod_ctx.capacity = self._capacity
        self._prod_ctx.mask = self._mask
        self._prod_ctx.cached_read = self._cached_read
        self._prod_ctx.cached_write = self._cached_write

    def __cinit__(
        self,
        path: str,
        int capacity_bytes,
        *,
        bint create=True,
        bint unlink_on_close=False,
        int spin_wait=1024,
    ) -> None:
        self._path_py = path
        path_b = (<str>path).encode()
        self._path = path_b
        if create:
            self._map_create(path_b, <u64>capacity_bytes, unlink_on_close, spin_wait)
        else:
            self._map_attach(path_b, spin_wait)
        self._init_prod_ctx()

    cdef inline bint _reserve(self, size_t need, u64* dropped_msgs):
        """Ensure space; drop oldest messages if required."""
        cdef u64 capacity = self._capacity
        cdef u64 mask = self._mask
        cdef u64 write_pos = self._cached_write
        cdef u64 read_pos
        cdef u64 free_bytes
        cdef u64 msg_len
        cdef u64 dropped_pos = 0
        dropped_msgs[0] = 0
        with nogil:
            read_pos = atomic_load_acquire(&self._hdr.read_pos)
        self._cached_read = read_pos
        free_bytes = capacity - (write_pos - read_pos)
        if free_bytes >= <u64>need:
            return True
        dropped_pos = read_pos
        while True:
            msg_len = read_u64_le(self._data, dropped_pos & mask, mask)
            if msg_len > capacity or (8 + msg_len) > capacity:
                return False
            dropped_pos += 8 + msg_len
            dropped_msgs[0] += 1
            free_bytes = capacity - (write_pos - dropped_pos)
            if free_bytes >= <u64>need:
                with nogil:
                    atomic_store_release(&self._hdr.read_pos, dropped_pos)
                    if dropped_msgs[0]:
                        atomic_sub(&self._hdr.msg_count, dropped_msgs[0])
                self._cached_read = dropped_pos
                return True

    cdef inline bint _can_reserve_once(self, size_t need):
        """Fast check that space is available without dropping."""
        cdef u64 capacity = self._capacity
        cdef u64 write_pos = self._cached_write
        cdef u64 read_pos
        with nogil:
            read_pos = atomic_load_acquire(&self._hdr.read_pos)
        self._cached_read = read_pos
        return (capacity - (write_pos - read_pos)) >= <u64>need

    cpdef bint insert(self, bytes item):
        """Insert a single item.

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
            success = shm_producer_insert(&self._prod_ctx, payload_ptr,
                                         <size_t>payload_len, &dropped)

        self._cached_read = self._prod_ctx.cached_read
        self._cached_write = self._prod_ctx.cached_write
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
            success = shm_producer_insert(&self._prod_ctx, <const unsigned char*>data,
                                         n, &dropped)

        self._cached_read = self._prod_ctx.cached_read
        self._cached_write = self._prod_ctx.cached_write
        return success == 1

    cpdef bint insert_batch(self, list[bytes] items):
        """Insert multiple items with a single commit of write_pos.

        Args:
            items: List of bytes payloads to insert.

        Returns:
            True if all items inserted, False if total size exceeds capacity.
        """
        cdef:
            Py_ssize_t i, n = len(items)
            u64 capacity = self._capacity
            u64 mask = self._mask
            u64 write_pos = self._cached_write
            size_t total = 0
            bytes it
            u64 msg_len
            u64 dropped = 0
            u64 now_ns

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
        if not self._reserve(total, &dropped):
            return False
        for i in range(n):
            it = items[i]
            msg_len = <u64>len(it)
            write_u64_le(self._data, write_pos & mask, mask, msg_len)
            copy_into_ring(self._data, write_pos + 8, mask, <const unsigned char*>it, <size_t>msg_len, capacity)
            write_pos += 8 + msg_len
        with nogil:
            now_ns = <u64>c_time_monotonic_ns()
            atomic_store_release(&self._hdr.write_pos, write_pos)
            atomic_add(&self._hdr.msg_count, <u64>n)
            atomic_store_release(&self._hdr.latest_insert_time_ns, now_ns)
        self._cached_write = write_pos
        self._prod_ctx.cached_read = self._cached_read
        self._prod_ctx.cached_write = write_pos
        return True

    cpdef bint is_empty(self):
        """Check if the buffer is empty.

        Returns:
            True if no messages are available.
        """
        return len(self) == 0

    cpdef bint is_full(self):
        """Check if the buffer is full.

        Returns:
            True if no space for even a 0-byte message.
        """
        cdef u64 read_pos
        with nogil:
            read_pos = atomic_load_acquire(&self._hdr.read_pos)
        self._cached_read = read_pos
        return (self._capacity - (self._cached_write - read_pos)) < 8

cdef class ShmSpscConsumer(_SharedBytesRing):
    """Shared-memory SPSC consumer for bytes payloads.

    Consumes variable-length byte messages from a shared-memory ring buffer
    with atomic coordination via the header.
    """

    cdef ShmConsumerContext _cons_ctx

    cdef inline void _init_cons_ctx(self):
        """Initialize consumer context for C functions."""
        self._cons_ctx.hdr = self._hdr
        self._cons_ctx.data = self._data
        self._cons_ctx.capacity = self._capacity
        self._cons_ctx.mask = self._mask
        self._cons_ctx.spin_wait = self._spin_wait

    def __cinit__(self, path: str, *, int spin_wait=1024) -> None:
        self._path_py = path
        path_b = (<str>path).encode()
        self._path = path_b
        self._map_attach(path_b, spin_wait)
        self._init_cons_ctx()

    cdef inline bint _peek_available(self, u64* out_len, u64* out_read_pos) nogil:
        """Check availability of a complete message without advancing."""
        cdef u64 read_pos = atomic_load_acquire(&self._hdr.read_pos)
        cdef u64 write_pos = atomic_load_acquire(&self._hdr.write_pos)
        cdef u64 avail = write_pos - read_pos
        if avail < 8:
            return False
        cdef u64 msg_len = read_u64_le(self._data, read_pos & self._mask, self._mask)
        if avail < 8 + msg_len:
            return False
        out_len[0] = <u64>msg_len
        out_read_pos[0] = read_pos
        return True

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

        while True:
            with nogil:
                available = shm_consumer_peek_available(&self._cons_ctx, &msg_len, &read_pos)
            if not available:
                spin_count += 1
                if spin_count < self._spin_wait:
                    continue
                _sched_yield()
                spin_count = 0
                continue
            with nogil:
                read_pos_check = atomic_load_acquire(&self._hdr.read_pos)
            if read_pos_check != read_pos:
                continue
            break

        buf = bytearray(<Py_ssize_t>msg_len)
        buf_ptr = <unsigned char*>buf
        with nogil:
            shm_consumer_consume(&self._cons_ctx, buf_ptr, msg_len, read_pos)

        self._cached_read = read_pos + 8 + msg_len
        return bytes(buf)

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

        while True:
            with nogil:
                available = shm_consumer_peek_available(&self._cons_ctx, &msg_len, &read_pos)
            if not available:
                spin_count += 1
                if spin_count < self._spin_wait:
                    continue
                _sched_yield()
                spin_count = 0
                continue
            with nogil:
                read_pos_check = atomic_load_acquire(&self._hdr.read_pos)
            if read_pos_check != read_pos:
                continue
            break

        if msg_len > <u64>len(dst):
            raise ValueError(
                f"Message size {msg_len} exceeds buffer size {len(dst)}"
            )

        buf_ptr = <unsigned char*>dst
        with nogil:
            shm_consumer_consume(&self._cons_ctx, buf_ptr, msg_len, read_pos)

        self._cached_read = read_pos + 8 + msg_len
        return <int>msg_len

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
            int i
            u64 msg_len = 0
            u64 read_pos = 0
            u64 read_pos_check = 0
            int available = 0
            bytearray buf
            unsigned char* buf_ptr

        for i in range(n):
            with nogil:
                available = shm_consumer_peek_available(&self._cons_ctx, &msg_len, &read_pos)
                if available:
                    read_pos_check = atomic_load_acquire(&self._hdr.read_pos)
            if not available:
                break
            if read_pos_check != read_pos:
                continue

            buf = buffers[i]
            if msg_len > <u64>len(buf):
                raise ValueError(
                    f"Message size {msg_len} exceeds buffer size {len(buf)}"
                )

            buf_ptr = <unsigned char*>buf
            with nogil:
                shm_consumer_consume(&self._cons_ctx, buf_ptr, msg_len, read_pos)

            total_copied += 1
            self._cached_read = read_pos + 8 + msg_len

        return total_copied

    cpdef object peekleft(self):
        """Peek at the next item without consuming; returns None if empty.

        Returns:
            The next bytes payload or None if buffer is empty.
        """
        cdef u64 msg_len = 0
        cdef u64 read_pos = 0
        cdef u64 read_pos_check = 0
        cdef u64 mask = self._mask
        cdef u64 cap = self._capacity
        cdef int available = 0

        with nogil:
            available = shm_consumer_peek_available(&self._cons_ctx, &msg_len, &read_pos)
            if available:
                read_pos_check = atomic_load_acquire(&self._hdr.read_pos)
        if not available:
            return None
        if read_pos_check != read_pos:
            return None
        cdef bytes out = bytes(<Py_ssize_t>msg_len)
        shm_copy_from_ring(<unsigned char*>out, self._data, read_pos + 8, mask, <size_t>msg_len, cap)
        return out

    cpdef object peekright(self):
        """Peek at the most recently inserted item; returns None if empty."""
        cdef u64 w
        cdef u64 r
        cdef u64 msg_len
        cdef u64 pos
        cdef u64 next_pos
        cdef u64 mask = self._mask
        cdef u64 cap = self._capacity
        cdef bytes out
        with nogil:
            w = atomic_load_acquire(&self._hdr.write_pos)
            r = atomic_load_acquire(&self._hdr.read_pos)
        if w - r < 8:
            return None
        pos = r
        while pos < w:
            msg_len = read_u64_le(self._data, pos & mask, mask)
            if msg_len > cap or 8 + msg_len > cap:
                return None
            next_pos = pos + 8 + msg_len
            if next_pos > w:
                return None
            if next_pos == w:
                out = bytes(<Py_ssize_t>msg_len)
                copy_from_ring(<unsigned char*>out, self._data, pos + 8, mask, <size_t>msg_len, cap)
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

        while True:
            with nogil:
                available = shm_consumer_peek_available(&self._cons_ctx, &msg_len, &read_pos)
                if available:
                    read_pos_check = atomic_load_acquire(&self._hdr.read_pos)
            if not available:
                break
            if read_pos_check != read_pos:
                continue
            buf = bytearray(<Py_ssize_t>msg_len)
            buf_ptr = <unsigned char*>buf
            with nogil:
                shm_consumer_consume(&self._cons_ctx, buf_ptr, msg_len, read_pos)
            res.append(bytes(buf))
            self._cached_read = read_pos + 8 + msg_len
        return res

    def consume_iterable(self):
        """Iterate over items, blocking until each is available."""
        while True:
            yield self.consume()

    async def aconsume(self):
        """Async consume a single item."""
        import asyncio
        cdef u64 msg_len = 0
        cdef u64 read_pos = 0
        cdef int spin_count = 0
        cdef int available = 0
        cdef bytearray buf
        cdef unsigned char* buf_ptr

        while True:
            with nogil:
                available = shm_consumer_peek_available(&self._cons_ctx, &msg_len, &read_pos)
            if available:
                break
            spin_count += 1
            if spin_count >= self._spin_wait:
                await asyncio.sleep(0)
                spin_count = 0

        buf = bytearray(<Py_ssize_t>msg_len)
        buf_ptr = <unsigned char*>buf
        with nogil:
            shm_consumer_consume(&self._cons_ctx, buf_ptr, msg_len, read_pos)

        self._cached_read = read_pos + 8 + msg_len
        return bytes(buf)

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
        cdef u64 mask = self._mask
        cdef u64 cap = self._capacity
        cdef bytes out

        with nogil:
            w = atomic_load_acquire(&self._hdr.write_pos)
            r = atomic_load_acquire(&self._hdr.read_pos)

        pos = r
        while pos < w:
            msg_len = read_u64_le(self._data, pos & mask, mask)
            if msg_len > cap or 8 + msg_len > cap:
                break
            next_pos = pos + 8 + msg_len
            if next_pos > w:
                break
            out = bytes(<Py_ssize_t>msg_len)
            copy_from_ring(<unsigned char*>out, self._data, pos + 8, mask, <size_t>msg_len, cap)
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
        """Check if the buffer is full (no space for even a 0-byte message)."""
        cdef u64 read_pos
        cdef u64 write_pos
        with nogil:
            read_pos = atomic_load_acquire(&self._hdr.read_pos)
            write_pos = atomic_load_acquire(&self._hdr.write_pos)
        return (self._capacity - (write_pos - read_pos)) < 8

    cpdef void clear(self):
        """Drain all available items from the buffer."""
        self.consume_all()
