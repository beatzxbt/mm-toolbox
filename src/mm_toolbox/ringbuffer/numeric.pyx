# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

import asyncio
import numpy as np
from typing import Iterator

cimport numpy as cnp
from libc.stdint cimport uint64_t as u64

from .numeric cimport numeric_t


cpdef object resolve_numeric_dtype(object dtype):
    """Resolve and validate numeric dtype to a NumPy dtype supported by numeric_t."""
    cdef object dt = np.dtype(dtype)
    cdef str kind_str = dt.kind
    cdef int sz = <int> dt.itemsize

    if kind_str == 'i':  # signed integer
        if sz in (1, 2, 4, 8):
            return dt
    elif kind_str == 'u':  # unsigned integer
        if sz in (1, 2, 4, 8):
            return dt
    elif kind_str == 'f':  # floating point
        if sz in (4, 8):
            return dt
    raise ValueError(f"Unsupported dtype for numeric_t: {dt} (kind={kind_str}, itemsize={sz})")


cdef class NumericRingBuffer:
    """A fixed-size ring buffer for numeric types."""

    def __cinit__(self, int max_capacity, object dtype, bint disable_async=False):
        """
        Parameters:
            max_capacity (int): The maximum number of elements the buffer can hold.
            dtype (numpy.dtype | type): The data type of the buffer.
            disable_async (bool): If True, the buffer will disable use of asyncio.Event for extra performance.
                All async methods will raise an exception.
        """
        if max_capacity <= 0:
            raise ValueError(f"Capacity cannot be negative; expected >0 but got {max_capacity}")
        self._max_capacity = 1 << (max_capacity - 1).bit_length() if max_capacity > 1 else 1
        self._mask = self._max_capacity - 1
        self._tail = 0
        self._head = 0
        self._size = 0
        self._dtype = resolve_numeric_dtype(dtype)
        self._buffer = np.empty(self._max_capacity, dtype=self._dtype)
        self._buffer_not_empty_event = asyncio.Event() 
        self._disable_async = disable_async

    cpdef cnp.ndarray raw(self, bint copy=True):
        """Return a copy of the internal buffer array."""
        return self._buffer.copy() if copy else self._buffer

    cpdef cnp.ndarray unwrapped(self):
        """Return a list of the buffer's contents in logical (oldest to newest) order."""
        cdef:
            u64 size = self._size
            u64 tail = self._tail
            u64 capacity = self._max_capacity
            cnp.ndarray buf = self._buffer

        if size == 0:
            return np.empty((0,), dtype=self._dtype)
        if tail + size <= capacity:
            return buf[tail:tail + size].copy()
        return np.concatenate((buf[tail:], buf[:(tail + size) % capacity]))

    cpdef void overwrite_latest(self, numeric_t item, bint increment_count=False):
        """Overwrite the latest element in the buffer. Optionally increment count."""
        cdef:
            u64 idx
            u64 head = self._head
            u64 tail = self._tail
            u64 mask = self._mask
            bint is_full = self._size == self._max_capacity
            numeric_t[::1] buf = self._buffer
        
        if increment_count:
            buf[head] = item
            if is_full:
                self._tail = (tail + 1) & mask
            else:
                self._size += 1
            self._head = (head + 1) & mask
            if not self._disable_async and self._size == 1:
                self._buffer_not_empty_event.set()
        else:
            idx = (head - 1) & mask
            buf[idx] = item

    cpdef void insert(self, numeric_t item):
        """Add a new element to the end of the buffer."""
        cdef:
            u64 head = self._head
            u64 tail = self._tail
            u64 mask = self._mask
            bint was_empty = self._size == 0
            bint is_full = self._size == self._max_capacity
            numeric_t[::1] buf = self._buffer

        buf[head] = item
        if is_full:
            self._tail = (tail + 1) & mask
        else:
            self._size += 1
        self._head = (head + 1) & mask
        if not self._disable_async and was_empty:
            self._buffer_not_empty_event.set()

    cpdef void insert_batch(self, numeric_t[::1] items):
        """Add a batch of elements to the end of the buffer."""
        cdef:
            u64 n = len(items)
            u64 old_size = self._size
            bint was_empty = self._size == 0
            u64 head = self._head
            u64 tail = self._tail
            u64 mask = self._mask
            u64 max_capacity = self._max_capacity
            u64 i, new_size, overwrite_count = 0
            numeric_t[::1] buf = self._buffer

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
            buf[head] = items[i]
            head = (head + 1) & mask

        self._head = head
        self._tail = tail
        self._size = new_size
        if not self._disable_async and was_empty:
            self._buffer_not_empty_event.set()

    cpdef bint contains(self, numeric_t item):
        """Checks if the item exists in the buffer, searching from newest to oldest."""
        if self.is_empty():
            return False

        cdef:
            u64 idx = (self._head - 1) & self._mask
            u64 remaining = self._size
            numeric_t[::1] buf = self._buffer

        while remaining:
            if buf[idx] == item:
                return True
            idx = (idx - 1) & self._mask
            remaining -= 1
        return False

    cpdef object consume(self):
        """Remove and return the first element from the buffer."""
        self.__enforce_ringbuffer_not_empty()
        cdef:
            u64 tail = self._tail
            u64 mask = self._mask
            u64 new_tail = (tail + 1) & mask
            cnp.ndarray buf = self._buffer

        self._tail = new_tail
        self._size -= 1
        if not self._disable_async and self.is_empty():
            self._buffer_not_empty_event.clear()
        return buf[tail]

    cpdef cnp.ndarray consume_all(self):
        """Remove and return all elements from the buffer."""
        self.__enforce_ringbuffer_not_empty()
        cdef cnp.ndarray result = self.unwrapped()
        self.clear()
        return result

    def consume_iterable(self) -> Iterator[object]:
        """Iterate over the elements in the buffer in order from oldest to newest."""
        while self._size > 0:
            yield self.consume()

    async def aconsume(self):
        """Remove and return the first element from the buffer (async)."""
        self.__enforce_async_not_disabled()
        if self._size > 0:
            return self.consume()
        await self._buffer_not_empty_event.wait()
        return self.consume()

    async def aconsume_iterable(self):
        """Yield items as they become available (async)."""
        self.__enforce_async_not_disabled()
        while True:
            if self._size > 0:
                yield self.consume()
                continue
            await self._buffer_not_empty_event.wait()

    cpdef object peekright(self):
        """Return the last element from the buffer without removing it."""
        cdef:
            u64 head = self._head
            u64 max_capacity = self._max_capacity
            cnp.ndarray buf = self._buffer

        return buf[(head - 1 + max_capacity) % max_capacity]

    cpdef object peekleft(self):
        """Return the first element from the buffer without removing it."""
        cdef:
            u64 tail = self._tail
            u64 max_capacity = self._max_capacity
            cnp.ndarray buf = self._buffer

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

    def __contains__(self, item):
        """Check if a item is present in the buffer."""
        # Code is identical to contains() but uses a different method signature
        # to avoid type checking issues. Due to this, it is guaranteed to be
        # slower than calling contains() directly.
        if self.is_empty():
            return False

        cdef:
            u64 idx = (self._head - 1) & self._mask
            u64 remaining = self._size
            cnp.ndarray buf = self._buffer

        while remaining:
            if buf[idx] == item:
                return True
            idx = (idx - 1) & self._mask
            remaining -= 1
        return False

    def __len__(self):
        """Get the number of elements currently in the buffer."""
        return self._size

    def __getitem__(self, int idx):
        """Get the element at the given index."""
        cdef:
            u64 size = self._size
            u64 tail = self._tail
            u64 capacity = self._max_capacity
            cnp.ndarray buf = self._buffer

        if idx < 0:
            idx += size
        if idx < 0 or <u64>idx >= size:
            raise IndexError(f"Index out of range; expected within ({-size} <= {idx} <= {size - 1}) but got {idx}")

        fixed_idx = (tail + <u64>idx) % capacity
        return buf[fixed_idx]

    cdef inline bint __enforce_ringbuffer_not_empty(self):
        if self.is_empty():
            raise IndexError("Cannot pop from an empty RingBuffer;")

    cdef inline bint __enforce_async_not_disabled(self):
        if self._disable_async:
            raise RuntimeError("Async operations are disabled for this buffer; use `disable_async=False` to enable.")
