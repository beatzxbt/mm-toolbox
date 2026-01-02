# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

import asyncio
from typing import Iterator, AsyncIterator

from libc.stdint cimport uint64_t as u64

cdef class BytesRingBuffer:
    """A fixed-size ring buffer for bytes objects."""

    def __cinit__(self, int max_capacity, bint disable_async=False, bint only_insert_unique=False) -> None:
        if max_capacity <= 0:
            raise ValueError(f"Capacity cannot be negative; expected >0 but got {max_capacity}")
        self._max_capacity = <u64>(1 << (max_capacity - 1).bit_length() if max_capacity > 1 else 1)
        self._mask = self._max_capacity - 1
        self._tail = 0
        self._head = 0
        self._size = 0
        self._buffer: list = [b""] * self._max_capacity
        self._buffer_not_empty_event = asyncio.Event()
        self._disable_async = disable_async
        self._only_insert_unique = only_insert_unique

    cpdef list raw(self, bint copy=True):
        """Return a copy of the internal buffer array."""
        return self._buffer.copy() if copy else self._buffer

    cpdef list unwrapped(self):
        """Return a list of the buffer's contents in logical (oldest to newest) order."""
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

    cpdef void overwrite_latest(self, bytes item, bint increment_count=False):
        """Overwrite the latest element in the buffer. Optionally increment count."""
        cdef u64 idx = (self._head - 1) & self._mask
        if increment_count:
            self.insert(item)
        else:
            self._buffer[idx] = item

    cpdef void insert(self, bytes item):
        """Add a new element to the end of the buffer."""
        if self._only_insert_unique and self.contains(item):
            return

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

    cpdef void insert_batch(self, list[bytes] items):
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
            list buf = self._buffer
            bint unique = self._only_insert_unique

        if n == 0:
            return

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

    cpdef bint contains(self, bytes item):
        """Checks if the item exists in the buffer, searching from newest to oldest."""
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
        """Remove and return the last element from the buffer."""
        self.__enforce_ringbuffer_not_empty()
        self._head = (self._head - 1) & self._mask
        cdef bytes item = self._buffer[self._head]
        self._size -= 1
        if not self._disable_async and self.is_empty():
            self._buffer_not_empty_event.clear()
        return item

    cpdef list consume_all(self):
        """Remove and return all elements from the buffer."""
        self.__enforce_ringbuffer_not_empty()
        cdef list result = self.unwrapped()
        self.clear()
        return result

    def consume_iterable(self) -> Iterator[bytes]:
        """Iterate over the elements in the buffer in order from oldest to newest."""
        while self._size > 0:
            yield self.consume()

    async def aconsume(self):
        """Remove and return the last element from the buffer."""
        self.__enforce_async_not_disabled()
        if self._size > 0:
            return self.consume()
        await self._buffer_not_empty_event.wait()
        return self.consume()

    async def aconsume_iterable(self) -> AsyncIterator[bytes]:
        """Remove and return the last element from the buffer."""
        self.__enforce_async_not_disabled()
        while True:
            if self._size > 0:
                yield self.consume()
                continue
            await self._buffer_not_empty_event.wait()

    cpdef bytes peekright(self):
        """Return the last element from the buffer without removing it."""
        self.__enforce_ringbuffer_not_empty()
        return self._buffer[(self._head - 1) & self._mask]

    cpdef bytes peekleft(self):
        """Return the first element from the buffer without removing it."""
        self.__enforce_ringbuffer_not_empty()
        return self._buffer[self._tail]

    cpdef void clear(self):
        """Clear the buffer and reset it to its initial state."""
        self._tail = 0
        self._head = 0
        self._size = 0
        if not self._disable_async:
            self._buffer_not_empty_event.clear()

    cpdef bint is_empty(self):
        """Check if the buffer is empty."""
        return self._size == 0

    cpdef bint is_full(self):
        """Check if the buffer is full."""
        return self._size == self._max_capacity

    def __contains__(self, bytes item):
        """Check if a value is present in the buffer."""
        return self.contains(item)

    def __len__(self):
        """Get the number of elements currently in the buffer."""
        return self._size

    def __getitem__(self, int idx):
        """Get the element at the given index."""
        cdef:
            u64     size = self._size
            u64     tail = self._tail
            u64     mask = self._mask
            list    buffer = self._buffer

        if idx < 0:
            idx += size
        if idx < 0 or <u64>idx >= size: 
            raise IndexError(f"Index out of range; expected within ({-size} <> {size}) but got {idx}")

        cdef u64 fixed_idx = (tail + <u64>idx) & mask
        return buffer[fixed_idx]

    cdef inline bint __enforce_ringbuffer_not_empty(self):
        if self.is_empty():
            raise IndexError("Cannot pop from an empty RingBuffer;")

    cdef inline bint __enforce_async_not_disabled(self):
        if self._disable_async:
            raise RuntimeError("Async operations are disabled for this buffer; use `disable_async=False` to enable.")


# BytesRingBufferFast - High-performance version with pre-allocated memory slots

import asyncio
from typing import Iterator, AsyncIterator

from libc.stdint cimport uint64_t as u64
from libc.string cimport memcpy, memcmp
from cpython.bytes cimport PyBytes_FromStringAndSize
from cpython.mem cimport PyMem_Malloc, PyMem_Free

cdef class BytesRingBufferFast:
    """A high-performance fixed-size ring buffer using pre-allocated memory slots."""

    def __cinit__(self, int max_capacity, bint disable_async=False, bint only_insert_unique=False, int expected_item_size=128, double buffer_percent=25.0, bint silent_overflow=False) -> None:
        if max_capacity <= 0:
            raise ValueError(f"Capacity cannot be negative; expected >0 but got {max_capacity}")
        if expected_item_size <= 0:
            raise ValueError(f"Expected item size cannot be negative; expected >0 but got {expected_item_size}")
        if buffer_percent < 0.0:
            raise ValueError(f"Buffer percent cannot be negative; got {buffer_percent}")

        self._max_capacity = 1 << (max_capacity - 1).bit_length() if max_capacity > 1 else 1
        self._mask = self._max_capacity - 1
        
        cdef u64 slot_size_base = <u64>(expected_item_size * (1.0 + buffer_percent / 100.0))
        self._slot_size = self._next_power_of_2(slot_size_base)
        self._slot_size_log2 = (<u64>self._slot_size - 1).bit_length()
        
        self._tail = 0
        self._head = 0
        self._size = 0
        
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
        self._silent_overflow = silent_overflow

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

    cpdef list raw(self, bint copy=True):
        """Return a copy of the internal buffer array."""
        cdef list result = []
        cdef u64 i
        for i in range(self._max_capacity):
            result.append(self._make_bytes(i))
        return result

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

    cpdef void overwrite_latest(self, bytes item, bint increment_count=False):
        """Overwrite the latest element in the buffer. Optionally increment count."""
        if increment_count:
            self.insert(item)
            return

        cdef:
            u64 idx = (self._head - 1) & self._mask
            char* dest = self._get_slot_ptr(idx)
            Py_ssize_t item_len = len(item)
            Py_ssize_t copy_len

        if item_len <= self._slot_size:
            copy_len = item_len
        elif self._silent_overflow:
            copy_len = self._slot_size
        else:
            raise ValueError(f"Item length {item_len} exceeds slot size {self._slot_size}")
        
        memcpy(dest, <const char*>item, copy_len)
        self._lengths[idx] = copy_len

    cpdef void insert(self, bytes item):
        """Add a new element to the end of the buffer."""
        if self._only_insert_unique and self.contains(item):
            return

        cdef:
            u64     head = self._head
            u64     tail = self._tail
            u64     mask = self._mask
            bint    is_full = self._size == self._max_capacity
            char*   dest = self._get_slot_ptr(head)
            Py_ssize_t item_len = len(item)
            Py_ssize_t copy_len

        if item_len <= self._slot_size:
            copy_len = item_len
        elif self._silent_overflow:
            copy_len = self._slot_size
        else:
            raise ValueError(f"Item length {item_len} exceeds slot size {self._slot_size}")

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

    cpdef void insert_char(self, const char* item, Py_ssize_t item_len):
        """Add a new element directly from char* to avoid byte conversion overhead."""
        cdef:
            u64     head = self._head
            u64     tail = self._tail
            u64     mask = self._mask
            bint    is_full = self._size == self._max_capacity
            char*   dest = self._get_slot_ptr(head)
            Py_ssize_t copy_len

        if item_len <= self._slot_size:
            copy_len = item_len
        elif self._silent_overflow:
            copy_len = self._slot_size
        else:
            raise ValueError(f"Item length {item_len} exceeds slot size {self._slot_size}")

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

    cpdef void insert_batch(self, list[bytes] items):
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
            return

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
                if item_len <= self._slot_size:
                    copy_len = item_len
                elif self._silent_overflow:
                    copy_len = self._slot_size
                else:
                    raise ValueError(f"Item length {item_len} exceeds slot size {self._slot_size}")
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
                if item_len <= self._slot_size:
                    copy_len = item_len
                elif self._silent_overflow:
                    copy_len = self._slot_size
                else:
                    raise ValueError(f"Item length {item_len} exceeds slot size {self._slot_size}")
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

    cpdef bint contains(self, bytes item):
        """Checks if the item exists in the buffer, searching from newest to oldest."""
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
        """Remove and return the last element from the buffer."""
        if self._size == 0:
            raise IndexError("Cannot pop from an empty RingBuffer;")
        
        self._head = (self._head - 1) & self._mask
        cdef bytes item = self._make_bytes(self._head)
        self._size -= 1
        if not self._disable_async and self._size == 0:
            self._buffer_not_empty_event.clear()
        return item

    cpdef list consume_all(self):
        """Remove and return all elements from the buffer."""
        if self._size == 0:
            raise IndexError("Cannot pop from an empty RingBuffer;")
        cdef list result = self.unwrapped()
        self.clear()
        return result

    def consume_iterable(self) -> Iterator[bytes]:
        """Iterate over the elements in the buffer in order from oldest to newest."""
        while self._size > 0:
            yield self.consume()

    async def aconsume(self):
        """Remove and return the last element from the buffer."""
        if self._disable_async:
            raise RuntimeError("Async operations are disabled for this buffer; use `disable_async=False` to enable.")
        if self._size > 0:
            return self.consume()
        await self._buffer_not_empty_event.wait()
        return self.consume()

    async def aconsume_iterable(self) -> AsyncIterator[bytes]:
        """Remove and return the last element from the buffer."""
        if self._disable_async:
            raise RuntimeError("Async operations are disabled for this buffer; use `disable_async=False` to enable.")
        while True:
            if self._size > 0:
                yield self.consume()
                continue
            await self._buffer_not_empty_event.wait()

    cpdef bytes peekright(self):
        """Return the last element from the buffer without removing it."""
        if self._size == 0:
            raise IndexError("Cannot pop from an empty RingBuffer;")
        return self._make_bytes((self._head - 1) & self._mask)

    cpdef bytes peekleft(self):
        """Return the first element from the buffer without removing it."""
        if self._size == 0:
            raise IndexError("Cannot pop from an empty RingBuffer;")
        return self._make_bytes(self._tail)

    cpdef void clear(self):
        """Clear the buffer and reset it to its initial state."""
        self._tail = 0
        self._head = 0
        self._size = 0
        if not self._disable_async:
            self._buffer_not_empty_event.clear()

    cpdef bint is_empty(self):
        """Check if the buffer is empty."""
        return self._size == 0

    cpdef bint is_full(self):
        """Check if the buffer is full."""
        return self._size == self._max_capacity

    def __contains__(self, bytes item):
        """Check if a value is present in the buffer."""
        return self.contains(item)

    def __len__(self):
        """Get the number of elements currently in the buffer."""
        return self._size

    def __getitem__(self, int idx):
        """Get the element at the given index."""
        cdef:
            u64     size = self._size
            u64     tail = self._tail
            u64     mask = self._mask

        if idx < 0:
            idx += size
        if idx < 0 or <u64>idx >= size: 
            raise IndexError(f"Index out of range; expected within ({-size} <> {size}) but got {idx}")

        cdef u64 fixed_idx = (tail + <u64>idx) & mask
        return self._make_bytes(fixed_idx)