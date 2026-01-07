# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

"""SPSC shared-memory bytes ring buffer.

Design
-------
- Single-producer/single-consumer over a file-backed mmap.
- Messages are length-prefixed (u64, little-endian) allowing arbitrary byte payloads.
- Producer advances write_pos with release semantics; consumer reads write_pos with acquire semantics.
- Consumer advances read_pos with release semantics; producer reads read_pos with acquire semantics.
- Capacity is rounded to the next power of two; indices use a mask to wrap.

Notes
-----
- Inserts are non-blocking and overwrite the oldest messages as needed.
- Consumer operations may block while waiting for new data.
- This is intentionally ONLY SPSC (single-producer/single-consumer) rings. A wrapper for SPMC will be added in the future,
  wrapping multiple SPSC rings within the same API.
"""

import os
import time
from libc.stdint cimport uint64_t as u64
from libc.string cimport memcpy
from libc.stddef cimport size_t
from libc.errno cimport errno

cdef extern from "sys/mman.h":
    void* mmap(void* addr, size_t length, int prot, int flags, int fd, long offset)
    int munmap(void* addr, size_t length)
    int PROT_READ
    int PROT_WRITE
    int MAP_SHARED

cdef extern from "fcntl.h":
    int open(const char* path, int oflag, ...)
    int O_RDONLY
    int O_RDWR
    int O_CREAT
    int O_EXCL

cdef extern from "unistd.h":
    int ftruncate(int fd, long length)
    int close(int fd)

# Use compiler builtins for atomics to avoid requiring stdatomic flags
cdef extern from *:
    unsigned long long __atomic_load_n(unsigned long long* ptr, int memorder)
    void __atomic_store_n(unsigned long long* ptr, unsigned long long val, int memorder)

cdef int _ATOMIC_RELAXED = 0
cdef int _ATOMIC_ACQUIRE = 2
cdef int _ATOMIC_RELEASE = 3

from .shm cimport ShmHeader, _SharedBytesRing

cdef inline u64 _atomic_load_acquire(u64* p) nogil:
    """Load value from pointer with acquire semantics."""
    return <u64>__atomic_load_n(<unsigned long long*>p, _ATOMIC_ACQUIRE)

cdef inline void _atomic_store_release(u64* p, u64 v) nogil:
    """Store value to pointer with release semantics."""
    __atomic_store_n(<unsigned long long*>p, <unsigned long long>v, _ATOMIC_RELEASE)

cdef inline size_t _align_up(size_t x, size_t a) nogil:
    """Align value up to the next multiple of alignment."""
    return (x + (a - 1)) & ~(a - 1)

cdef inline u64 _pow2_at_least(u64 v) nogil:
    """Return the smallest power of two greater than or equal to value."""
    cdef u64 p = 1
    if v <= 1:
        return 1
    while p < v:
        p <<= 1
    return p

cdef inline void _write_u64_le(unsigned char* base, u64 idx, u64 mask, u64 val) nogil:
    """Write 64-bit value in little-endian format with wrap-around."""
    cdef u64 pos = idx & mask
    cdef u64 capacity = mask + 1
    if pos + 8 <= capacity:
        # Fast path: no wrap-around, use direct write if aligned
        if pos & 7 == 0:
            # Aligned: direct 64-bit write
            (<u64*>(base + pos))[0] = val
        else:
            # Unaligned: byte-by-byte
            base[pos + 0] = <unsigned char>(val & 0xFF)
            base[pos + 1] = <unsigned char>((val >> 8) & 0xFF)
            base[pos + 2] = <unsigned char>((val >> 16) & 0xFF)
            base[pos + 3] = <unsigned char>((val >> 24) & 0xFF)
            base[pos + 4] = <unsigned char>((val >> 32) & 0xFF)
            base[pos + 5] = <unsigned char>((val >> 40) & 0xFF)
            base[pos + 6] = <unsigned char>((val >> 48) & 0xFF)
            base[pos + 7] = <unsigned char>((val >> 56) & 0xFF)
    else:
        # Slow path: wrap-around
        base[(idx + 0) & mask] = <unsigned char>(val & 0xFF)
        base[(idx + 1) & mask] = <unsigned char>((val >> 8) & 0xFF)
        base[(idx + 2) & mask] = <unsigned char>((val >> 16) & 0xFF)
        base[(idx + 3) & mask] = <unsigned char>((val >> 24) & 0xFF)
        base[(idx + 4) & mask] = <unsigned char>((val >> 32) & 0xFF)
        base[(idx + 5) & mask] = <unsigned char>((val >> 40) & 0xFF)
        base[(idx + 6) & mask] = <unsigned char>((val >> 48) & 0xFF)
        base[(idx + 7) & mask] = <unsigned char>((val >> 56) & 0xFF)

cdef inline u64 _read_u64_le(const unsigned char* base, u64 idx, u64 mask) nogil:
    """Read 64-bit value in little-endian format with wrap-around."""
    cdef u64 pos = idx & mask
    cdef u64 capacity = mask + 1
    if pos + 8 <= capacity:
        # Fast path: no wrap-around
        if pos & 7 == 0:
            # Aligned: direct 64-bit read
            return (<u64*>(base + pos))[0]
        else:
            # Unaligned: byte-by-byte
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
        # Slow path: wrap-around
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

cdef inline void _copy_in(unsigned char* dst, const unsigned char* src, size_t n) nogil:
    """Copy bytes from source to destination."""
    memcpy(dst, src, n)

cdef inline void _copy_into_ring(unsigned char* ring, u64 start, u64 mask, const unsigned char* src, size_t n, u64 capacity) nogil:
    """Copy bytes into ring buffer with wrap-around."""
    if n == 0:
        return
    cdef u64 idx = start & mask
    cdef u64 endspace = capacity - idx
    if n <= endspace:
        # Fast path: no wrap-around
        memcpy(ring + idx, src, n)
    else:
        # Slow path: wrap-around
        memcpy(ring + idx, src, <size_t>endspace)
        memcpy(ring, src + <size_t>endspace, n - <size_t>endspace)

cdef inline void _copy_from_ring(unsigned char* dst, const unsigned char* ring, u64 start, u64 mask, size_t n, u64 capacity) nogil:
    """Copy bytes from ring buffer with wrap-around."""
    if n == 0:
        return
    cdef u64 idx = start & mask
    cdef u64 endspace = capacity - idx
    if n <= endspace:
        # Fast path: no wrap-around
        memcpy(dst, ring + idx, n)
    else:
        # Slow path: wrap-around
        memcpy(dst, ring + idx, <size_t>endspace)
        memcpy(dst + <size_t>endspace, ring, n - <size_t>endspace)


cdef u64 _MAGIC = 0x53484252  # 'SHBR' (Shared Bytes Ring)
cdef u64 _VERSION = 1
cdef size_t _HEADER_ALIGN = 64
cdef size_t _HEADER_SIZE = _align_up(sizeof(ShmHeader), _HEADER_ALIGN)


cdef class _SharedBytesRing:
    """Common mapping and lifecycle for shared ring buffers.

    Attributes
    ----------
    _hdr
        Pointer to shared header containing cursors and metadata.
    _data
        Pointer to ring data region (capacity bytes).
    _capacity
        Ring capacity in bytes, power-of-two.
    _mask
        Capacity - 1 for fast wrapping.
    _spin_wait
        Spin iterations before yielding inside blocking ops.
    """
    def __cinit__(self) -> None:
        """Initialize the shared bytes ring buffer."""
        if os.name != "posix":
            raise OSError("Shared memory ringbuffer is only supported on POSIX platforms")
        self._hdr = NULL
        self._data = NULL
        self._map_len = 0
        self._fd = -1
        self._owner = False
        self._unlink_on_close = False
        self._capacity = 0
        self._mask = 0
        self._cached_read = 0
        self._cached_write = 0
        self._spin_wait = 1024
        self._path_py = None
        self._path = NULL

    cdef void _map_create(self, bytes path_b, u64 capacity_bytes, bint unlink_on_close, int spin_wait):
        """Create and initialize a new shared ring buffer."""
        cdef:
            int fd
            size_t cap = <size_t>_pow2_at_least(capacity_bytes if capacity_bytes > 0 else 1)
            size_t total_len = _HEADER_SIZE + cap
            void* base
        fd = open(path_b, O_CREAT | O_RDWR, 0o600)
        if fd < 0:
            raise OSError(errno, "open failed for shared ring")
        if ftruncate(fd, <long>total_len) != 0:
            close(fd)
            raise OSError(errno, "ftruncate failed for shared ring")
        base = mmap(NULL, total_len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)
        if <long>base == -1:
            close(fd)
            # Best-effort cleanup: try to unlink file if it was just created
            try:
                import os
                os.unlink(path_b)
            except Exception:
                pass
            raise OSError(errno, "mmap failed for shared ring")

        self._fd = fd
        self._map_len = total_len
        self._hdr = <ShmHeader*>base
        self._data = <unsigned char*>base + _HEADER_SIZE
        self._owner = True
        self._unlink_on_close = unlink_on_close
        self._spin_wait = spin_wait if spin_wait > 0 else 1024
        self._capacity = <u64>cap
        self._mask = <u64>(cap - 1)
        self._cached_read = 0
        self._cached_write = 0

        # Initialize header
        self._hdr.magic = _MAGIC
        self._hdr.version = _VERSION
        self._hdr.capacity = <u64>cap
        self._hdr.mask = <u64>(cap - 1)
        self._hdr.read_pos = 0
        self._hdr.write_pos = 0

    cdef void _map_attach(self, bytes path_b, int spin_wait):
        """Attach to an existing shared ring buffer."""
        cdef:
            int fd
            void* base
            size_t total_len
            u64 capacity
        fd = open(path_b, O_RDWR, 0o600)
        if fd < 0:
            raise OSError(errno, "open failed for shared ring")
        # Map header only to read capacity
        base = mmap(NULL, _HEADER_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)
        if <long>base == -1:
            close(fd)
            raise OSError(errno, "mmap header failed for shared ring")
        self._hdr = <ShmHeader*>base
        if self._hdr.magic != _MAGIC or self._hdr.version != _VERSION:
            munmap(base, _HEADER_SIZE)
            close(fd)
            raise RuntimeError("Shared ring header mismatch")
        capacity = self._hdr.capacity
        munmap(base, _HEADER_SIZE)

        total_len = _HEADER_SIZE + <size_t>capacity
        base = mmap(NULL, total_len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)
        if <long>base == -1:
            close(fd)
            raise OSError(errno, "mmap failed for shared ring")

        self._fd = fd
        self._map_len = total_len
        self._hdr = <ShmHeader*>base
        self._data = <unsigned char*>base + _HEADER_SIZE
        self._owner = False
        self._unlink_on_close = False
        self._spin_wait = spin_wait if spin_wait > 0 else 1024
        self._capacity = capacity
        self._mask = self._hdr.mask
        self._cached_read = self._hdr.read_pos
        self._cached_write = self._hdr.write_pos

    cdef inline void _close_map(self):
        """Unmap memory and close file descriptor."""
        if self._hdr != NULL:
            munmap(<void*>self._hdr, self._map_len)
            self._hdr = NULL
            self._data = NULL
            self._map_len = 0
        if self._fd >= 0:
            close(self._fd)
            self._fd = -1

    def __dealloc__(self):
        """Deallocate resources on object destruction."""
        try:
            self.close()
        except Exception:
            pass

    cpdef void close(self):
        """Close all mappings and optionally unlink the backing file."""
        cdef object path_obj
        cdef bytes path_b
        self._close_map()
        if self._owner and self._unlink_on_close and self._path_py is not None:
            # Best-effort unlink via Python os to avoid cross-platform libc differences
            try:
                import os
                os.unlink(self._path_py)
            except Exception:
                pass
        self._owner = False

    def __enter__(self):
        """Enter context manager, returning self.
        
        Returns:
            Self for use in with statements.
        """
        return self

    def __exit__(self, exc_type, exc, tb):
        """Exit context manager, closing resources.
        
        Args:
            exc_type: Exception type if an exception occurred.
            exc: Exception instance if an exception occurred.
            tb: Traceback if an exception occurred.
        """
        self.close()


cdef class SharedBytesRingBufferProducer(_SharedBytesRing):
    """Shared-memory SPSC producer for bytes payloads.

    Parameters
    ----------
    path
        Filesystem path for the ring backing file.
    capacity_bytes
        Total ring capacity in bytes (rounded to next power of two).
    create
        If True, create/initialize the ring; otherwise attach to existing.
    unlink_on_close
        If True and owner, unlink backing file on close.
    spin_wait
        Spin iterations in blocking calls before polling again.
    """
    def __cinit__(
        self,
        path: str,
        int capacity_bytes,
        *,
        bint create=True,
        bint unlink_on_close=False,
        int spin_wait=1024,
    ) -> None:
        """Initialize the producer.
        
        Args:
            path: Filesystem path for the ring backing file.
            capacity_bytes: Total ring capacity in bytes (rounded to next power of two).
            create: If True, create/initialize the ring; otherwise attach to existing.
            unlink_on_close: If True and owner, unlink backing file on close.
            spin_wait: Spin iterations in blocking calls before polling again.
        """
        self._path_py = path
        path_b = (<str>path).encode()
        self._path = path_b
        if create:
            self._map_create(path_b, <u64>capacity_bytes, unlink_on_close, spin_wait)
        else:
            self._map_attach(path_b, spin_wait)

    cdef inline bint _reserve(self, size_t need):
        """Ensure space for required bytes by dropping oldest messages as needed."""
        cdef u64 cap = self._capacity
        cdef u64 mask = self._mask
        cdef u64 w = self._cached_write
        cdef u64 r
        cdef u64 free_bytes
        cdef u64 L
        cdef u64 dropped_pos = 0
        with nogil:
            r = _atomic_load_acquire(&self._hdr.read_pos)
        self._cached_read = r
        free_bytes = cap - (w - r)
        if free_bytes >= <u64>need:
            return True
        # Not enough space: drop oldest messages until we have enough
        dropped_pos = r
        while True:
            L = _read_u64_le(self._data, dropped_pos & mask, mask)
            # Overflow check: ensure L doesn't exceed capacity
            if L > cap or (dropped_pos + 8 + L) < dropped_pos:
                return False
            dropped_pos += 8 + L
            free_bytes = cap - (w - dropped_pos)
            if free_bytes >= <u64>need:
                # Batch update: single atomic store for all dropped messages
                with nogil:
                    _atomic_store_release(&self._hdr.read_pos, dropped_pos)
                self._cached_read = dropped_pos
                return True

    cdef inline bint _can_reserve_once(self, size_t need):
        """Check once if space for required bytes is available."""
        cdef u64 cap = self._capacity
        cdef u64 w = self._cached_write
        cdef u64 r
        with nogil:
            r = _atomic_load_acquire(&self._hdr.read_pos)
        self._cached_read = r
        return (cap - (w - r)) >= <u64>need

    cpdef bint insert(self, bytes item):
        """Insert a single item into the ring buffer."""
        cdef:
            Py_ssize_t n = len(item)
            size_t total = <size_t>(8 + n)
            u64 cap = self._capacity
            u64 mask = self._mask
            u64 w = self._cached_write
        if n < 0:
            return False
        if total > cap:
            return False
        self._reserve(total)
        # Write header and payload
        _write_u64_le(self._data, w & mask, mask, <u64>n)
        _copy_into_ring(self._data, w + 8, mask, <const unsigned char*>item, <size_t>n, cap)
        w += total
        with nogil:
            _atomic_store_release(&self._hdr.write_pos, w)
        self._cached_write = w
        return True

    cpdef bint insert_batch(self, list[bytes] items):
        """Insert multiple items in order using a single reserve and commit."""
        cdef:
            Py_ssize_t i, n = len(items)
            u64 cap = self._capacity
            u64 mask = self._mask
            u64 w = self._cached_write
            u64 r
            u64 free_bytes
            u64 L64
            size_t total = 0
            bytes it
        if n == 0:
            return True
        # Validate and compute total size
        for i in range(n):
            it = items[i]
            L64 = <u64>len(it)
            if <u64>(8) + L64 > cap:
                return False
            total += <size_t>(8 + L64)
        # Drop oldest messages until enough contiguous capacity is free (byte-accurate)
        with nogil:
            r = _atomic_load_acquire(&self._hdr.read_pos)
        self._cached_read = r
        while (cap - (w - r)) < <u64>total:
            L64 = _read_u64_le(self._data, r & mask, mask)
            # Overflow check: ensure L64 doesn't exceed capacity
            if L64 > cap or (r + 8 + L64) < r:
                return False
            r += 8 + L64
            with nogil:
                _atomic_store_release(&self._hdr.read_pos, r)
            self._cached_read = r
        # Write all items
        for i in range(n):
            it = items[i]
            L64 = <u64>len(it)
            _write_u64_le(self._data, w & mask, mask, L64)
            _copy_into_ring(self._data, w + 8, mask, <const unsigned char*>it, <size_t>L64, cap)
            w += 8 + L64
        with nogil:
            _atomic_store_release(&self._hdr.write_pos, w)
        self._cached_write = w
        return True

    cpdef bint insert_packed(self, list[bytes] items):
        """Insert items as one packed message."""
        cdef Py_ssize_t i, n = len(items)
        cdef Py_ssize_t total = 0
        cdef bytes it
        if n == 0:
            return True
        for i in range(n):
            it = items[i]
            total += 4 + len(it)
        if total <= 0:
            return True
        if <u64>(8 + total) > self._capacity:
            return False
        cdef bytearray buf = bytearray(<int>total)
        cdef unsigned char* p = <unsigned char*>buf
        cdef size_t off = 0
        cdef Py_ssize_t L
        for i in range(n):
            it = items[i]
            L = len(it)
            # write 4-byte little-endian length into contiguous buffer
            p[off + 0] = <unsigned char>(L & 0xFF)
            p[off + 1] = <unsigned char>((L >> 8) & 0xFF)
            p[off + 2] = <unsigned char>((L >> 16) & 0xFF)
            p[off + 3] = <unsigned char>((L >> 24) & 0xFF)
            off += 4
            memcpy(p + off, <const unsigned char*>it, <size_t>L)
            off += <size_t>L
        # Now insert single packed message
        return self.insert(bytes(buf))


cdef class SharedBytesRingBufferConsumer(_SharedBytesRing):
    """Shared-memory SPSC consumer for bytes payloads.

    Parameters
    ----------
    path
        Filesystem path of an existing ring backing file to attach to.
    spin_wait
        Spin iterations in blocking calls before polling again.
    """
    def __cinit__(self, path: str, *, int spin_wait=1024) -> None:
        """Initialize the consumer.
        
        Args:
            path: Filesystem path of an existing ring backing file to attach to.
            spin_wait: Spin iterations in blocking calls before polling again.
        """
        self._path_py = path
        path_b = (<str>path).encode()
        self._path = path_b
        self._map_attach(path_b, spin_wait)

    cdef inline bint _peek_available(self, u64* out_len, u64* out_read_pos) nogil:
        """Check if a complete message is available without committing read."""
        cdef u64 r = _atomic_load_acquire(&self._hdr.read_pos)
        cdef u64 w = _atomic_load_acquire(&self._hdr.write_pos)
        cdef u64 avail = w - r
        if avail < 8:
            return False
        cdef u64 L = _read_u64_le(self._data, r & self._mask, self._mask)
        if avail < 8 + L:
            return False
        out_len[0] = <u64>L
        out_read_pos[0] = r
        return True

    cpdef bytes consume(self):
        """Consume a single item (blocking)."""
        cdef:
            u64 L64 = 0
            u64 r = 0
            u64 r_check = 0
            u64 mask = self._mask
            u64 cap = self._capacity
            int spin_count = 0
            bint available = False
        while True:
            with nogil:
                available = self._peek_available(&L64, &r)
            if not available:
                spin_count += 1
                if spin_count < self._spin_wait:
                    continue
                # Yield to other threads after spin_wait iterations
                time.sleep(0.0001)
                spin_count = 0
                continue
            # Re-check read_pos hasn't changed (producer might have dropped messages)
            with nogil:
                r_check = _atomic_load_acquire(&self._hdr.read_pos)
            if r_check != r:
                # Producer advanced read_pos, re-check availability
                continue
            break
        cdef u64 L = L64
        cdef bytes out = bytes(<Py_ssize_t>L)
        _copy_from_ring(<unsigned char*>out, self._data, r + 8, mask, <size_t>L, cap)
        r += 8 + L
        with nogil:
            _atomic_store_release(&self._hdr.read_pos, r)
        self._cached_read = r
        return out

    cpdef object try_consume(self):
        """Try to consume a single item (non-blocking)."""
        cdef u64 L64 = 0
        cdef u64 r = 0
        cdef u64 r_check = 0
        cdef u64 mask = self._mask
        cdef u64 cap = self._capacity
        with nogil:
            if not self._peek_available(&L64, &r):
                return None
            r_check = _atomic_load_acquire(&self._hdr.read_pos)
        if r_check != r:
            return None
        cdef u64 L = L64
        cdef bytes out = bytes(<Py_ssize_t>L)
        _copy_from_ring(<unsigned char*>out, self._data, r + 8, mask, <size_t>L, cap)
        r += 8 + L
        with nogil:
            _atomic_store_release(&self._hdr.read_pos, r)
        self._cached_read = r
        return out

    cpdef list consume_all_nowait(self):
        """Drain all currently available items without blocking."""
        cdef list res = []
        cdef object it
        while True:
            it = self.try_consume()
            if it is None:
                break
            res.append(it)
        return res

    cpdef list consume_packed(self):
        """Consume a packed message and unpack into a list of items."""
        cdef bytes buf = self.consume()
        cdef memoryview mv = memoryview(buf)
        cdef Py_ssize_t n = mv.shape[0]
        cdef Py_ssize_t off = 0
        cdef list items = []
        cdef u64 L
        while off + 4 <= n:
            L = (
                (<u64>mv[off])
                | (<u64>mv[off + 1] << 8)
                | (<u64>mv[off + 2] << 16)
                | (<u64>mv[off + 3] << 24)
            )
            off += 4
            if off + L > n:
                raise ValueError("Corrupted packed message")
            items.append(bytes(mv[off : off + L]))
            off += L
        return items

