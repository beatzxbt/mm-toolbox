# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

import asyncio
from typing import Iterator, AsyncIterator

from libc.stdint cimport uint64_t as u64
from libc.string cimport memcpy, memcmp
from cpython.bytes cimport PyBytes_FromStringAndSize
from cpython.mem cimport PyMem_Malloc, PyMem_Free

from mm_toolbox.time.time cimport time_monotonic_ns

cdef class BytesRingBuffer:
    """A fixed-size ring buffer for bytes objects.

    Provides FIFO semantics with optional async waiting, batch operations,
    and uniqueness constraints. When full, new inserts overwrite the oldest
    elements.

    Attributes:
        _max_capacity: Ring capacity (rounded up to power of two).
        _disable_async: If True, async operations are disabled.
        _only_insert_unique: If True, duplicate inserts are skipped.
    """

    def __cinit__(self, int max_capacity, bint disable_async=False, bint only_insert_unique=False) -> None:
        """Initialize a new BytesRingBuffer.

        Args:
            max_capacity: Maximum number of elements (rounded up to power of two).
            disable_async: If True, disable asyncio.Event for performance.
            only_insert_unique: If True, skip duplicate inserts.

        Raises:
            ValueError: If max_capacity is not positive.
        """
        if max_capacity <= 0:
            raise ValueError(f"Capacity cannot be negative; expected >0 but got {max_capacity}")
        self._max_capacity = <u64>(1 << (max_capacity - 1).bit_length() if max_capacity > 1 else 1)
        self._mask = self._max_capacity - 1
        self._tail = 0
        self._head = 0
        self._size = 0
        self._latest_insert_time_ns = 0
        self._latest_consume_time_ns = 0
        self._buffer: list = [b""] * self._max_capacity
        self._buffer_not_empty_event = asyncio.Event()
        self._disable_async = disable_async
        self._only_insert_unique = only_insert_unique

    cpdef list unwrapped(self):
        """Return a list of the buffer's contents in logical (oldest to newest) order.

        Returns:
            List of bytes objects in FIFO order.
        """
        cdef:
            u64     size = self._size
            u64     tail = self._tail
            u64     capacity = self._max_capacity
            u64     mask = self._mask
            list    buf = self._buffer

        if size == 0:
            return []
        if tail + size <= capacity:
            return buf[tail:tail + size]
        return buf[tail:] + buf[:(tail + size) & mask]

    cpdef bint insert(self, bytes item):
        """Add a new element to the end of the buffer.

        If the buffer is full, the oldest element is overwritten.

        Args:
            item: Bytes object to insert.

        Returns:
            True if the insert succeeded (or was skipped due to uniqueness).
        """
        if self._only_insert_unique and self.contains(item):
            return True

        cdef:
            u64     head = self._head
            u64     tail = self._tail
            u64     mask = self._mask
            bint    is_full = self.is_full()
            list    buf = self._buffer

        if is_full:
            tail = (tail + 1) & mask
        buf[head] = item
        if not is_full:
            self._size += 1
        self._head = (head + 1) & mask
        self._tail = tail
        if not self._disable_async and self._size == 1:
            self._buffer_not_empty_event.set()
        self._latest_insert_time_ns = <u64>time_monotonic_ns()
        return True

    cpdef bint insert_char(self, const char* data, Py_ssize_t n):
        """Add a new element directly from char* to avoid byte conversion overhead.

        Args:
            data: Pointer to raw character data.
            n: Length of the data in bytes.

        Returns:
            True if the insert succeeded.
        """
        cdef bytes item = data[:n]
        return self.insert(item)

    cpdef int consume_into(self, bytearray dst):
        """Consume one item and copy it into the provided bytearray.

        Args:
            dst: Pre-allocated bytearray to copy the item into.

        Returns:
            Number of bytes copied.

        Raises:
            IndexError: If the buffer is empty.
            ValueError: If dst is too small to hold the item.
        """
        self.__enforce_ringbuffer_not_empty()
        cdef:
            u64 tail = self._tail
            bytes item = self._buffer[tail]
            Py_ssize_t item_len = len(item)
            Py_ssize_t dst_len = len(dst)
        if dst_len < item_len:
            raise ValueError(f"Destination buffer too small: {dst_len} < {item_len}")
        memcpy(<char*>dst, <const char*>item, item_len)
        self._tail = (tail + 1) & self._mask
        self._size -= 1
        if not self._disable_async and self.is_empty():
            self._buffer_not_empty_event.clear()
        self._latest_consume_time_ns = <u64>time_monotonic_ns()
        return item_len

    cpdef int consume_all_into(self, list buffers):
        """Consume all available items and copy them into the provided bytearrays.

        Args:
            buffers: List of pre-allocated bytearrays.

        Returns:
            Number of messages copied.

        Raises:
            ValueError: If any destination buffer is too small.
        """
        cdef:
            u64 n = min(<u64>len(buffers), self._size)
            u64 i
            u64 tail = self._tail
            u64 mask = self._mask
            bytes item
            bytearray dst
            Py_ssize_t item_len
            Py_ssize_t dst_len
        if n == 0:
            return 0
        for i in range(n):
            item = self._buffer[tail]
            dst = buffers[i]
            item_len = len(item)
            dst_len = len(dst)
            if dst_len < item_len:
                raise ValueError(f"Destination buffer too small at index {i}: {dst_len} < {item_len}")
            memcpy(<char*>dst, <const char*>item, item_len)
            tail = (tail + 1) & mask
        self._tail = tail
        self._size -= n
        if not self._disable_async and self.is_empty():
            self._buffer_not_empty_event.clear()
        self._latest_consume_time_ns = <u64>time_monotonic_ns()
        return n

    cpdef bint insert_batch(self, list[bytes] items):
        """Add a batch of elements to the end of the buffer.

        Args:
            items: List of bytes objects to insert.

        Returns:
            True if the batch insert succeeded.
        """
        cdef:
            bytes item
            u64 i, n = len(items)
            u64 old_size = self._size
            u64 max_capacity = self._max_capacity
            u64 head = self._head
            u64 tail = self._tail
            u64 mask = self._mask
            u64 new_size, overwrite_count
            list buf = self._buffer
            bint unique = self._only_insert_unique

        if n == 0:
            return True

        if not unique:
            if n >= max_capacity:
                items = items[-max_capacity:]
                n = max_capacity

            if old_size + n > max_capacity:
                overwrite_count = old_size + n - max_capacity
                tail = (tail + overwrite_count) & mask
                new_size = max_capacity
            else:
                new_size = old_size + n

            for i in range(n):
                buf[head] = items[i]
                head = (head + 1) & mask

            self._head = head
            self._tail = tail
            self._size = new_size
        else:
            for i in range(n):
                item = items[i]
                if self.contains(item):
                    continue
                if self._size == max_capacity:
                    tail = (tail + 1) & mask
                buf[head] = item
                if self._size < max_capacity:
                    self._size += 1
                head = (head + 1) & mask

            self._head = head
            self._tail = tail

        if not self._disable_async and old_size == 0 and self._size > 0:
            self._buffer_not_empty_event.set()
        self._latest_insert_time_ns = <u64>time_monotonic_ns()
        return True

    cpdef bint contains(self, bytes item):
        """Checks if the item exists in the buffer, searching from newest to oldest.

        Args:
            item: Bytes object to search for.

        Returns:
            True if the item is found in the buffer.
        """
        if self.is_empty():
            return False

        cdef:
            u64     idx = (self._head - 1) & self._mask
            u64     remaining = self._size
            list    buf = self._buffer
            bytes   item_at_idx

        while remaining:
            item_at_idx = buf[idx]
            if item_at_idx == item:
                return True
            idx = (idx - 1) & self._mask
            remaining -= 1
        return False

    cpdef bytes consume(self):
        """Remove and return the first (oldest) element from the buffer.

        Returns:
            The oldest bytes object in the buffer.

        Raises:
            IndexError: If the buffer is empty.
        """
        self.__enforce_ringbuffer_not_empty()
        cdef u64 tail = self._tail
        cdef bytes item = self._buffer[tail]
        self._tail = (tail + 1) & self._mask
        self._size -= 1
        if not self._disable_async and self.is_empty():
            self._buffer_not_empty_event.clear()
        self._latest_consume_time_ns = <u64>time_monotonic_ns()
        return item

    cpdef list consume_all(self):
        """Remove and return all elements from the buffer.

        Returns:
            List of all bytes objects in FIFO order.

        Raises:
            IndexError: If the buffer is empty.
        """
        self.__enforce_ringbuffer_not_empty()
        cdef list result = self.unwrapped()
        self.clear()
        return result

    def consume_iterable(self) -> Iterator[bytes]:
        """Iterate over the elements in the buffer in order from oldest to newest.

        Yields:
            Bytes objects in FIFO order.
        """
        while self._size > 0:
            yield self.consume()

    async def aconsume(self):
        """Remove and return the first (oldest) element from the buffer (async).

        Returns:
            The oldest bytes object in the buffer.

        Raises:
            RuntimeError: If async operations are disabled.
        """
        self.__enforce_async_not_disabled()
        if self._size > 0:
            return self.consume()
        await self._buffer_not_empty_event.wait()
        return self.consume()

    async def aconsume_iterable(self) -> AsyncIterator[bytes]:
        """Yield and remove elements from the buffer in FIFO order (async).

        Yields:
            Bytes objects in FIFO order.

        Raises:
            RuntimeError: If async operations are disabled.
        """
        self.__enforce_async_not_disabled()
        while True:
            if self._size > 0:
                yield self.consume()
                continue
            await self._buffer_not_empty_event.wait()

    cpdef bytes peekright(self):
        """Return the last element from the buffer without removing it.

        Returns:
            The newest bytes object in the buffer.

        Raises:
            IndexError: If the buffer is empty.
        """
        self.__enforce_ringbuffer_not_empty()
        return self._buffer[(self._head - 1) & self._mask]

    cpdef bytes peekleft(self):
        """Return the first element from the buffer without removing it.

        Returns:
            The oldest bytes object in the buffer.

        Raises:
            IndexError: If the buffer is empty.
        """
        self.__enforce_ringbuffer_not_empty()
        return self._buffer[self._tail]

    cpdef void clear(self):
        """Clear the buffer and reset it to its initial state.

        All elements are discarded and the buffer is empty after this call.
        """
        self._tail = 0
        self._head = 0
        self._size = 0
        if not self._disable_async:
            self._buffer_not_empty_event.clear()

    cpdef bint is_empty(self):
        """Check if the buffer is empty.

        Returns:
            True if the buffer contains no elements.
        """
        return self._size == 0

    cpdef bint is_full(self):
        """Check if the buffer is full.

        Returns:
            True if the buffer has reached its maximum capacity.
        """
        return self._size == self._max_capacity

    def __contains__(self, bytes item):
        """Check if a value is present in the buffer."""
        return self.contains(item)

    def __len__(self):
        """Get the number of elements currently in the buffer."""
        return self._size

    @property
    def latest_insert_time_ns(self):
        """Return the timestamp (ns) of the latest successful insert."""
        return self._latest_insert_time_ns

    @property
    def latest_consume_time_ns(self):
        """Return the timestamp (ns) of the latest successful consume."""
        return self._latest_consume_time_ns

    cdef inline bint __enforce_ringbuffer_not_empty(self):
        if self.is_empty():
            raise IndexError("Cannot pop from an empty RingBuffer;")

    cdef inline bint __enforce_async_not_disabled(self):
        if self._disable_async:
            raise RuntimeError("Async operations are disabled for this buffer; use `disable_async=False` to enable.")


# BytesRingBufferFast - High-performance version with pre-allocated memory slots

cdef class BytesRingBufferFast:
    """A high-performance fixed-size ring buffer using pre-allocated memory slots.

    Stores byte strings in fixed-size memory slots allocated via PyMem_Malloc,
    avoiding Python object overhead for better cache locality and throughput.
    """

    def __cinit__(self, int max_capacity, bint disable_async=False, bint only_insert_unique=False, int expected_item_size=128, double buffer_percent=25.0) -> None:
        if max_capacity <= 0:
            raise ValueError(f"Capacity cannot be negative; expected >0 but got {max_capacity}")
        if expected_item_size <= 0:
            raise ValueError(f"Expected item size cannot be negative; expected >0 but got {expected_item_size}")
        if buffer_percent < 0.0:
            raise ValueError(f"Buffer percent cannot be negative; got {buffer_percent}")
        import math
        if not math.isfinite(buffer_percent):
            raise ValueError(f"buffer_percent must be finite; got {buffer_percent}")

        self._max_capacity = 1 << (max_capacity - 1).bit_length() if max_capacity > 1 else 1
        self._mask = self._max_capacity - 1
        
        cdef u64 slot_size_base = <u64>(expected_item_size * (1.0 + buffer_percent / 100.0))
        self._slot_size = self._next_power_of_2(slot_size_base)
        self._slot_size_log2 = (<u64>self._slot_size - 1).bit_length()

        # Validate slot_size to prevent overflow in total_bytes calculation
        if self._slot_size == 0:
            raise ValueError("Calculated slot size cannot be zero")
        cdef size_t max_alloc = <size_t>-1
        if self._slot_size > <u64>(max_alloc // self._max_capacity):
            raise ValueError(
                f"Buffer allocation would overflow: capacity={self._max_capacity}, "
                f"slot_size={self._slot_size}"
            )
        
        self._tail = 0
        self._head = 0
        self._size = 0
        self._latest_insert_time_ns = 0
        self._latest_consume_time_ns = 0
        
        cdef u64 total_bytes = self._max_capacity * self._slot_size
        self._buffer = <char*>PyMem_Malloc(total_bytes)
        if not self._buffer:
            raise MemoryError("Failed to allocate buffer")
        
        self._lengths = <u64*>PyMem_Malloc(self._max_capacity * sizeof(u64))
        if not self._lengths:
            PyMem_Free(self._buffer)
            raise MemoryError("Failed to allocate lengths")
        
        self._buffer_not_empty_event = asyncio.Event()
        self._disable_async = disable_async
        self._only_insert_unique = only_insert_unique

    def __dealloc__(self):
        if self._buffer:
            PyMem_Free(self._buffer)
        if self._lengths:
            PyMem_Free(self._lengths)

    cdef inline char* _get_slot_ptr(self, u64 idx) nogil:
        """Get pointer to slot at index using bit shift for performance."""
        return self._buffer + (idx << self._slot_size_log2)

    cdef inline bytes _make_bytes(self, u64 idx):
        """Create bytes object from slot at index."""
        cdef char* ptr = self._get_slot_ptr(idx)
        return PyBytes_FromStringAndSize(ptr, self._lengths[idx])

    cdef inline u64 _next_power_of_2(self, u64 n) nogil:
        """Calculate the next power of 2 greater than or equal to n."""
        if n == 0:
            return 1
        if n & (n - 1) == 0:
            return n
        n -= 1
        n |= n >> 1
        n |= n >> 2
        n |= n >> 4
        n |= n >> 8
        n |= n >> 16
        n |= n >> 32
        return n + 1

    cpdef list unwrapped(self):
        """Return a list of the buffer's contents in logical (oldest to newest) order."""
        cdef:
            u64     size = self._size
            u64     tail = self._tail
            u64     mask = self._mask
            list    result = []

        if size == 0:
            return []
        cdef u64 i, idx
        for i in range(size):
            idx = (tail + i) & mask
            result.append(self._make_bytes(idx))
        return result

    cpdef bint insert(self, bytes item):
        """Add a new element to the end of the buffer."""
        if self._only_insert_unique and self.contains(item):
            return True

        cdef:
            u64     head = self._head
            u64     tail = self._tail
            u64     mask = self._mask
            bint    is_full = self._size == self._max_capacity
            char*   dest = self._get_slot_ptr(head)
            Py_ssize_t item_len = len(item)
            Py_ssize_t copy_len

        if item_len > self._slot_size:
            raise ValueError(f"Item length {item_len} exceeds slot size {self._slot_size}")
        copy_len = item_len

        memcpy(dest, <const char*>item, copy_len)
        self._lengths[head] = copy_len
        
        if is_full:
            tail = (tail + 1) & mask
        else:
            self._size += 1
        
        self._head = (head + 1) & mask
        self._tail = tail
        
        if not self._disable_async and self._size == 1:
            self._buffer_not_empty_event.set()
        self._latest_insert_time_ns = <u64>time_monotonic_ns()
        return True

    cpdef bint insert_char(self, const char* item, Py_ssize_t item_len):
        """Add a new element directly from char* to avoid byte conversion overhead."""
        cdef:
            u64     head = self._head
            u64     tail = self._tail
            u64     mask = self._mask
            bint    is_full = self._size == self._max_capacity
            char*   dest = self._get_slot_ptr(head)
            Py_ssize_t copy_len

        if item_len > self._slot_size:
            raise ValueError(f"Item length {item_len} exceeds slot size {self._slot_size}")
        copy_len = item_len

        memcpy(dest, item, copy_len)
        self._lengths[head] = copy_len
        
        if is_full:
            tail = (tail + 1) & mask
        else:
            self._size += 1
        
        self._head = (head + 1) & mask
        self._tail = tail
        
        if not self._disable_async and self._size == 1:
            self._buffer_not_empty_event.set()
        self._latest_insert_time_ns = <u64>time_monotonic_ns()
        return True

    cpdef int consume_into(self, bytearray dst):
        """Consume one item and copy it into the provided bytearray."""
        if self._size == 0:
            raise IndexError("Cannot pop from an empty RingBuffer;")
        cdef:
            u64 tail = self._tail
            char* src = self._get_slot_ptr(tail)
            u64 item_len = self._lengths[tail]
            Py_ssize_t dst_len = len(dst)
        if dst_len < <Py_ssize_t>item_len:
            raise ValueError(f"Destination buffer too small: {dst_len} < {item_len}")
        memcpy(<char*>dst, src, item_len)
        self._tail = (tail + 1) & self._mask
        self._size -= 1
        if not self._disable_async and self._size == 0:
            self._buffer_not_empty_event.clear()
        self._latest_consume_time_ns = <u64>time_monotonic_ns()
        return <int>item_len

    cpdef int consume_all_into(self, list buffers):
        """Consume all available items and copy them into the provided bytearrays.

        Returns the number of messages copied.
        """
        cdef:
            u64 n = min(<u64>len(buffers), self._size)
            u64 i
            u64 tail = self._tail
            u64 mask = self._mask
            char* src
            bytearray dst
            u64 item_len
            Py_ssize_t dst_len
        if n == 0:
            return 0
        for i in range(n):
            src = self._get_slot_ptr(tail)
            item_len = self._lengths[tail]
            dst = buffers[i]
            dst_len = len(dst)
            if dst_len < <Py_ssize_t>item_len:
                raise ValueError(f"Destination buffer too small at index {i}: {dst_len} < {item_len}")
            memcpy(<char*>dst, src, item_len)
            tail = (tail + 1) & mask
        self._tail = tail
        self._size -= n
        if not self._disable_async and self._size == 0:
            self._buffer_not_empty_event.clear()
        self._latest_consume_time_ns = <u64>time_monotonic_ns()
        return <int>n

    cpdef bint insert_batch(self, list[bytes] items):
        """Add a batch of elements to the end of the buffer."""
        cdef:
            bytes item
            u64 i, n = len(items)
            u64 old_size = self._size
            u64 max_capacity = self._max_capacity
            u64 head = self._head
            u64 tail = self._tail
            u64 mask = self._mask
            u64 new_size, overwrite_count
            Py_ssize_t item_len, copy_len
            char* dest
            bint unique = self._only_insert_unique

        if n == 0:
            return True

        if not unique:
            if n >= max_capacity:
                items = items[-max_capacity:]
                n = max_capacity

            if old_size + n > max_capacity:
                overwrite_count = old_size + n - max_capacity
                tail = (tail + overwrite_count) & mask
                new_size = max_capacity
            else:
                new_size = old_size + n

            for i in range(n):
                item = items[i]
                dest = self._get_slot_ptr(head)
                item_len = len(item)
                if item_len > self._slot_size:
                    raise ValueError(f"Item length {item_len} exceeds slot size {self._slot_size}")
                copy_len = item_len
                memcpy(dest, <const char*>item, copy_len)
                self._lengths[head] = copy_len
                head = (head + 1) & mask

            self._head = head
            self._tail = tail
            self._size = new_size
        else:
            for i in range(n):
                item = items[i]
                if self.contains(item):
                    continue
                dest = self._get_slot_ptr(head)
                item_len = len(item)
                if item_len > self._slot_size:
                    raise ValueError(f"Item length {item_len} exceeds slot size {self._slot_size}")
                copy_len = item_len
                memcpy(dest, <const char*>item, copy_len)
                self._lengths[head] = copy_len
                if self._size == max_capacity:
                    tail = (tail + 1) & mask
                else:
                    self._size += 1
                head = (head + 1) & mask

            self._head = head
            self._tail = tail

        if not self._disable_async and old_size == 0 and self._size > 0:
            self._buffer_not_empty_event.set()
        self._latest_insert_time_ns = <u64>time_monotonic_ns()
        return True

    cpdef bint contains(self, bytes item):
        """Checks if the item exists in the buffer, searching from newest to oldest.

        Args:
            item: Bytes object to search for.

        Returns:
            True if the item is found in the buffer.
        """
        if self._size == 0:
            return False

        cdef:
            Py_ssize_t item_len = len(item)
            const char* item_ptr = <const char*>item
            u64     idx = (self._head - 1) & self._mask
            u64     remaining = self._size
            char*   buf_ptr

        while remaining:
            if self._lengths[idx] == item_len:
                buf_ptr = self._get_slot_ptr(idx)
                if memcmp(buf_ptr, item_ptr, item_len) == 0:
                    return True
            idx = (idx - 1) & self._mask
            remaining -= 1
        return False

    cpdef bytes consume(self):
        """Remove and return the first (oldest) element from the buffer.

        Returns:
            The oldest bytes object in the buffer.

        Raises:
            IndexError: If the buffer is empty.
        """
        if self._size == 0:
            raise IndexError("Cannot pop from an empty RingBuffer;")
        
        cdef u64 tail = self._tail
        cdef bytes item = self._make_bytes(tail)
        self._tail = (tail + 1) & self._mask
        self._size -= 1
        if not self._disable_async and self._size == 0:
            self._buffer_not_empty_event.clear()
        self._latest_consume_time_ns = <u64>time_monotonic_ns()
        return item

    cpdef list consume_all(self):
        """Remove and return all elements from the buffer.

        Returns:
            List of all bytes objects in FIFO order.

        Raises:
            IndexError: If the buffer is empty.
        """
        if self._size == 0:
            raise IndexError("Cannot pop from an empty RingBuffer;")
        cdef list result = self.unwrapped()
        self.clear()
        return result

    def consume_iterable(self) -> Iterator[bytes]:
        """Iterate over the elements in the buffer in order from oldest to newest.

        Yields:
            Bytes objects in FIFO order.
        """
        while self._size > 0:
            yield self.consume()

    async def aconsume(self):
        """Remove and return the first (oldest) element from the buffer (async).

        Returns:
            The oldest bytes object in the buffer.

        Raises:
            RuntimeError: If async operations are disabled.
        """
        if self._disable_async:
            raise RuntimeError("Async operations are disabled for this buffer; use `disable_async=False` to enable.")
        if self._size > 0:
            return self.consume()
        await self._buffer_not_empty_event.wait()
        return self.consume()

    async def aconsume_iterable(self) -> AsyncIterator[bytes]:
        """Yield and remove elements from the buffer in FIFO order (async).

        Yields:
            Bytes objects in FIFO order.

        Raises:
            RuntimeError: If async operations are disabled.
        """
        if self._disable_async:
            raise RuntimeError("Async operations are disabled for this buffer; use `disable_async=False` to enable.")
        while True:
            if self._size > 0:
                yield self.consume()
                continue
            await self._buffer_not_empty_event.wait()

    cpdef bytes peekright(self):
        """Return the last element from the buffer without removing it.

        Returns:
            The newest bytes object in the buffer.

        Raises:
            IndexError: If the buffer is empty.
        """
        if self._size == 0:
            raise IndexError("Cannot pop from an empty RingBuffer;")
        return self._make_bytes((self._head - 1) & self._mask)

    cpdef bytes peekleft(self):
        """Return the first element from the buffer without removing it.

        Returns:
            The oldest bytes object in the buffer.

        Raises:
            IndexError: If the buffer is empty.
        """
        if self._size == 0:
            raise IndexError("Cannot pop from an empty RingBuffer;")
        return self._make_bytes(self._tail)

    cpdef void clear(self):
        """Clear the buffer and reset it to its initial state.

        All elements are discarded and the buffer is empty after this call.
        """
        self._tail = 0
        self._head = 0
        self._size = 0
        if not self._disable_async:
            self._buffer_not_empty_event.clear()

    cpdef bint is_empty(self):
        """Check if the buffer is empty.

        Returns:
            True if the buffer contains no elements.
        """
        return self._size == 0

    cpdef bint is_full(self):
        """Check if the buffer is full.

        Returns:
            True if the buffer has reached its maximum capacity.
        """
        return self._size == self._max_capacity

    def __contains__(self, bytes item):
        """Check if a value is present in the buffer."""
        return self.contains(item)

    def __len__(self):
        """Get the number of elements currently in the buffer."""
        return self._size

    @property
    def latest_insert_time_ns(self):
        """Return the timestamp (ns) of the latest successful insert."""
        return self._latest_insert_time_ns

    @property
    def latest_consume_time_ns(self):
        """Return the timestamp (ns) of the latest successful consume."""
        return self._latest_consume_time_ns
