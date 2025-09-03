# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

import asyncio
from typing import AsyncIterator, Iterator

from libc.stdint cimport uint64_t as u64

"""
Performance on my machine (MacBook Air M2, 16GB RAM).

Benchmark inserts a byte string of 128 bytes, checks existance then consumes the message.

With `disable_async=False`:
Insert: ~313 ns/msg
Contains: ~160 ns/check
Consume: ~234 ns/msg

With `disable_async=True`:
Insert: ~97 ns/msg
Contains: ~103 ns/check
Consume: ~105 ns/msg
"""

cdef class GenericRingBuffer:
    """A fixed-size ring buffer for objects."""

    def __cinit__(self, int max_capacity, bint disable_async=False) -> None:
        """
        Parameters:
            capacity (int): The maximum number of elements the buffer can hold.
            disable_async (bool): If True, the buffer will disable use of asyncio.Event for extra performance.
                All async methods will raise an exception.
        """
        if max_capacity <= 0:
            raise ValueError(f"Capacity cannot be negative; expected >0 but got {max_capacity}")

        # Find the smallest power of 2 greater than or equal to max_capacity.
        self._max_capacity = 1 << (max_capacity - 1).bit_length() if max_capacity > 1 else 1
        self._mask = self._max_capacity - 1
        self._tail = 0
        self._head = 0
        self._size = 0
        self._buffer: list = [None] * self._max_capacity
        self._buffer_not_empty_event = asyncio.Event() 
        self._disable_async = disable_async

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
    
    cpdef void overwrite_latest(self, object item, bint increment_count=False):
        """Overwrite the latest element in the buffer. Optionally increment count."""
        cdef u64 idx
        if increment_count:
            self.insert(item)
        else:
            idx = (self._head - 1) & self._mask
            self._buffer[idx] = item

    cpdef void insert(self, object item):
        """Add a new element to the end of the buffer."""
        cdef:
            u64     head = self._head
            u64     tail = self._tail
            u64     mask = self._mask
            bint    is_full = self._size == self._max_capacity
            list    buf = self._buffer

        buf[head] = item
        if is_full:
            self._tail = (tail + 1) & mask
        else:
            self._size += 1
        self._head = (head + 1) & mask
        if not self._disable_async and self._size == 1:
            self._buffer_not_empty_event.set()

    cpdef void insert_batch(self, list[object] items):
        """Add a batch of elements to the end of the buffer."""
        cdef: 
            u64     i, n = len(items)
            u64     old_size = self._size
            bint    was_empty = self._size == 0
            u64     max_capacity = self._max_capacity
            u64     head = self._head
            u64     tail = self._tail
            u64     mask = self._mask
            u64     new_size, overwrite_count = 0
            list    buf = self._buffer

        if n == 0:
            return

        # If batch is larger than capacity, only keep the last max_capacity items
        if n >= max_capacity:
            items = items[-max_capacity:]
            n = max_capacity

        # Calculate how many will be overwritten
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

        if not self._disable_async and was_empty:
            self._buffer_not_empty_event.set()
    
    cpdef bint contains(self, object item):
        """Checks if the item exists in the buffer, searching from newest to oldest."""
        cdef:
            u64     idx = (self._head - 1) & self._mask
            u64     remaining = self._size
            list    buf = self._buffer
            object  item_at_idx
            bint    eq_result 

        while remaining:
            item_at_idx = buf[idx]
            if item_at_idx is item:
                return True

            try:
                # Safeguard against objects which dont return
                # a bool when compared to another object.
                if eq_result := bool(item_at_idx == item): 
                    return eq_result
            except Exception:
                pass

            idx = (idx - 1) & self._mask
            remaining -= 1
        return False

    cpdef object consume(self):
        """Remove and return the last element from the buffer."""
        self.__enforce_ringbuffer_not_empty()
        cdef:
            u64     mask = self._mask
            u64     new_head = (self._head - 1) & mask
            list    buf = self._buffer

        self._head = new_head
        self._size -= 1
        if not self._disable_async and self.is_empty():
            self._buffer_not_empty_event.clear()

        return buf[new_head]

    cpdef list consume_all(self):
        """Remove and return all elements from the buffer."""
        self.__enforce_ringbuffer_not_empty()
        cdef list result = self.unwrapped()
        self.clear()
        return result

    def consume_iterable(self) -> Iterator[object]:
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

    async def aconsume_iterable(self) -> AsyncIterator[object]:
        """Remove and return the last element from the buffer."""
        self.__enforce_async_not_disabled()
        while True:
            if self._size > 0:
                yield self.consume()
            await self._buffer_not_empty_event.wait()
    
    cpdef object peekright(self):
        """Return the last element from the buffer without removing it."""
        cdef:
            u64     head = self._head   
            u64     max_capacity = self._max_capacity
            list    buf = self._buffer

        return buf[(head - 1 + max_capacity) % max_capacity]
    
    cpdef object peekleft(self):
        """Return the first element from the buffer without removing it."""
        cdef:
            u64     tail = self._tail
            u64     max_capacity = self._max_capacity
            list    buf = self._buffer

        return buf[(tail + max_capacity) % max_capacity]

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

    def __contains__(self, object item):
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
            list    buf = self._buffer

        if idx < 0:
            idx += size
        if idx < 0 or <u64>idx >= size: 
            raise IndexError(f"Index out of range; expected within ({-size} <> {size}) but got {idx}")

        fixed_idx = (tail + <u64>idx) % capacity
        return buf[fixed_idx]

    cdef inline bint __enforce_ringbuffer_not_empty(self):
        if self.is_empty():
            raise IndexError("Cannot pop from an empty RingBuffer;")

    cdef inline bint __enforce_async_not_disabled(self):
        if self._disable_async:
            raise RuntimeError("Async operations are disabled for this buffer; use `disable_async=False` to enable.")