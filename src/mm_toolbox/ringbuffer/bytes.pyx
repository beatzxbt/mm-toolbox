# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

import asyncio
from typing import Iterator, AsyncIterator

from libc.stdint cimport uint64_t as u64
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy, memcmp
from cpython.bytes cimport PyBytes_FromStringAndSize

cdef class BytesRingBuffer:
    """A fixed-size ring buffer for bytes objects."""

    def __cinit__(self, int max_capacity, bint disable_async=False, bint only_insert_unique=False) -> None:
        if max_capacity <= 0:
            raise ValueError(f"Capacity cannot be negative; expected >0 but got {max_capacity}")
        self._max_capacity = 1 << (max_capacity - 1).bit_length() if max_capacity > 1 else 1
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
            list    buf = self._buffer

        if size == 0:
            return []
        if tail + size <= capacity:
            return buf[tail:tail + size]
        return buf[tail:] + buf[:(tail + size) % capacity]

    cpdef void overwrite_latest(self, bytes item, bint increment_count=False):
        """Overwrite the latest element in the buffer. Optionally increment count."""
        cdef u64 idx
        if increment_count:
            self.insert(item)
        else:
            idx = (self._head - 1) & self._mask
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

        buf[head] = item
        if is_full:
            self._tail = (tail + 1) & mask
        else:
            self._size += 1
        self._head = (head + 1) & mask
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
            u64 new_size, overwrite_count = 0

        if n == 0:
            return

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
            if self._only_insert_unique and self.contains(item):
                continue
            self._buffer[head] = item
            head = (head + 1) & mask

        self._head = head
        self._tail = tail
        self._size = new_size

        if not self._disable_async and old_size == 0:
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
            if item_at_idx is item or item_at_idx == item:
                return True
            idx = (idx - 1) & self._mask
            remaining -= 1
        return False

    cpdef bytes consume(self):
        """Remove and return the last element from the buffer."""
        self.__enforce_ringbuffer_not_empty()
        self._head = (self._head - 1) & self._mask
        self._size -= 1
        if not self._disable_async and self.is_empty():
            self._buffer_not_empty_event.clear()
        return self._buffer[self._head]

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
            await self._buffer_not_empty_event.wait()

    cpdef bytes peekright(self):
        """Return the last element from the buffer without removing it."""
        return self._buffer[(self._head - 1 + self._max_capacity) % self._max_capacity]

    cpdef bytes peekleft(self):
        """Return the first element from the buffer without removing it."""
        return self._buffer[(self._tail + self._max_capacity) % self._max_capacity]

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
            u64     capacity = self._max_capacity
            list    buffer = self._buffer

        if idx < 0:
            idx += size
        if idx < 0 or <u64>idx >= size: 
            raise IndexError(f"Index out of range; expected within ({-size} <> {size}) but got {idx}")

        fixed_idx = (tail + <u64>idx) % capacity
        return buffer[fixed_idx]

    cdef inline bint __enforce_ringbuffer_not_empty(self):
        if self.is_empty():
            raise IndexError("Cannot pop from an empty RingBuffer;")

    cdef inline bint __enforce_async_not_disabled(self):
        if self._disable_async:
            raise RuntimeError("Async operations are disabled for this buffer; use `disable_async=False` to enable.")


# cdef class BytesRingBufferFast:
#     """Fast fixed-size ring buffer for bytes using C char* slots."""

#     cdef:
#         u64     _max_capacity
#         u64     _mask
#         u64     _tail
#         u64     _head
#         u64     _size
#         u64     _max_msg_size
#         char*   _data
#         u64*    _lengths
#         bint    _only_insert_unique
#         bint    _silent_size_cap

#     def __cinit__(self, int max_capacity, int max_msg_size, bint only_insert_unique=False, bint silent_size_cap=False) -> None:
#         if max_capacity <= 0:
#             raise ValueError(f"Capacity cannot be negative; expected >0 but got {max_capacity}")
#         if max_msg_size <= 0:
#             raise ValueError(f"Message size must be >0; got {max_msg_size}")

#         self._max_capacity = 1 << (max_capacity - 1).bit_length() if max_capacity > 1 else 1
#         self._mask = self._max_capacity - 1
#         self._max_msg_size = 1 << (max_msg_size - 1).bit_length()
#         self._tail = 0
#         self._head = 0
#         self._size = 0
#         self._only_insert_unique = only_insert_unique
#         self._silent_size_cap = silent_size_cap

#         self._data = <char*> malloc(self._max_capacity * self._max_msg_size)
#         if self._data is NULL:
#             raise MemoryError("Failed to allocate buffer data")
#         self._lengths = <u64*> malloc(self._max_capacity * sizeof(u64))
#         if self._lengths is NULL:
#             free(self._data)
#             self._data = NULL
#             raise MemoryError("Failed to allocate buffer lengths")

#     def __dealloc__(self):
#         if self._data is not NULL:
#             free(self._data)
#             self._data = NULL
#         if self._lengths is not NULL:
#             free(self._lengths)
#             self._lengths = NULL

#     cdef inline char* _slot_ptr(self, u64 idx) nogil:
#         return self._data + (idx * self._max_msg_size)

#     cdef bytes _get_at_index(self, u64 pos):
#         cdef u64 n = self._lengths[pos]
#         return <bytes> PyBytes_FromStringAndSize(self._slot_ptr(pos), n)

#     cpdef bint is_empty(self):
#         """Check if the buffer is empty."""
#         return self._size == 0

#     cpdef bint is_full(self):
#         """Check if the buffer is full."""
#         return self._size == self._max_capacity

#     cpdef void clear(self):
#         """Reset indices and size to empty state."""
#         self._tail = 0
#         self._head = 0
#         self._size = 0

#     cpdef void insert(self, bytes item):
#         """Append an item, overwriting the oldest when full."""
#         cdef:
#             u64 n, pos
#             char* internal_buffer
#             const char* source = <const char*> item

#         if self._only_insert_unique and self.contains(item):
#             return

#         n = <u64> len(item)
#         if n > self._max_msg_size:
#             n = self._max_msg_size
#             if self._silent_size_cap:
#                 return
#             else:
#                 raise ValueError(f"Message size must be <= {self._max_msg_size}; got {n}")
        
#         pos = self._head
#         internal_buffer = self._slot_ptr(pos)
#         memcpy(internal_buffer, source, n)
#         self._lengths[pos] = n

#         if self.is_full():
#             self._tail = (self._tail + 1) & self._mask
#         else:
#             self._size += 1
#         self._head = (self._head + 1) & self._mask

#     cpdef void insert_batch(self, list[bytes] items):
#         """Append a batch of items, truncating each to max_msg_size."""
#         cdef:
#             u64 i, n = <u64> len(items)
#             bytes item
#         if n == 0:
#             return
#         for i in range(n):
#             item = items[i]
#             self.insert(item)

#     cpdef bytes consume(self):
#         """Pop and return the newest element."""
#         if self.is_empty():
#             raise IndexError("Cannot pop from an empty RingBuffer;")
#         self._head = (self._head - 1) & self._mask
#         self._size -= 1
#         return self._get_at_index(self._head)

#     cpdef bytes peekright(self):
#         """Return the newest element without removing it."""
#         if self.is_empty():
#             raise IndexError("Cannot peek from an empty RingBuffer;")
#         return self._get_at_index((self._head - 1) & self._mask)

#     cpdef bytes peekleft(self):
#         """Return the oldest element without removing it."""
#         if self.is_empty():
#             raise IndexError("Cannot peek from an empty RingBuffer;")
#         return self._get_at_index(self._tail)

#     cpdef bint contains(self, bytes item):
#         """Check if the exact bytes object exists (by value)."""
#         if self.is_empty():
#             return False
#         cdef:
#             u64 remaining = self._size
#             u64 idx = (self._head - 1) & self._mask
#             const char* needle = <const char*> item
#             u64 nlen = <u64> len(item)
#             u64 slen
#             char* slot
#         while remaining:
#             slen = self._lengths[idx]
#             if slen == nlen:
#                 slot = self._slot_ptr(idx)
#                 if memcmp(<const void*>slot, <const void*>needle, nlen) == 0:
#                     return True
#             idx = (idx - 1) & self._mask
#             remaining -= 1
#         return False

#     def consume_iterable(self) -> Iterator[bytes]:
#         """Iterate from oldest to newest, consuming elements."""
#         while self._size > 0:
#             yield self.consume()

#     def __len__(self):
#         """Return current number of elements."""
#         return self._size

#     def __getitem__(self, int idx):
#         """Get the element at logical index (oldest=0)."""
#         cdef:
#             u64 size = self._size
#             u64 tail = self._tail
#         if idx < 0:
#             idx += size
#         if idx < 0 or <u64>idx >= size:
#             raise IndexError(f"Index out of range; expected within ({-size} <> {size}) but got {idx}")
#         cdef u64 fixed_idx = (tail + <u64>idx) & self._mask
#         return self._get_at_index(fixed_idx)

#     def __contains__(self, bytes item):
#         """Return True if value is present in the buffer."""
#         return self.contains(item)
