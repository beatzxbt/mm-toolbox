import numpy as np
cimport numpy as cnp

from libc.stdint cimport uint32_t as u32

cdef class RingBufferOneDim:
    """
    A 1-dimensional fixed-size circular buffer for floats/doubles.
    """

    def __cinit__(self, int capacity):
        """
        Parameters:
            capacity (int): The maximum number of elements the buffer can hold.
        """
        if capacity <= 0:
            raise ValueError(f"Capacity cannot be negative; expected >0 but got {capacity}")

        self._capacity = capacity
        self._left_index = 0
        self._right_index = 0
        self._size = 0
        self._buffer = np.empty(capacity, dtype=np.double)

    cpdef cnp.ndarray raw(self):
        """
        Return a copy of the internal buffer array.

        Returns:
            np.ndarray: A copy of the internal 1D NumPy array representing the buffer's contents.

        Note:
            The returned array includes all allocated space, not just the filled elements.
        """
        return np.asarray(self._buffer).copy()
    
    cpdef cnp.ndarray unwrapped(self):
        """
        Return a copy of the buffer's contents in the correct (unwrapped) order.

        Returns:
            np.ndarray: A 1D NumPy array containing the buffer's data in order from oldest to newest.

        Tip:
            If you intend to iterate over the buffer in order for read only purposes,
            use `iter(self)` instead of `self.unwrapped()` for better performance.
        """
        if self.is_empty():
            return np.empty_like(self._buffer)
            
        if self.is_full():
            return np.concatenate((
                self._buffer[self._left_index:], 
                self._buffer[:self._right_index]
            ))
        elif self._left_index < self._right_index:
            return np.asarray(self._buffer[self._left_index:self._right_index]).copy()
        else:
            return np.concatenate((
                self._buffer[self._left_index:], 
                self._buffer[:self._right_index]
            ))

    cpdef void append(self, double value):
        """
        Add a new element to the end of the buffer.

        Parameters:
            value (float): The float value to be added to the buffer.
        """
        self._buffer[self._right_index] = value

        if self.is_full():
            self._left_index = (self._left_index + 1) % self._capacity
        else:
            self._size += 1
        
        self._right_index = (self._right_index + 1) % self._capacity
        
    cpdef double popright(self):
        """
        Remove and return the last element from the buffer.

        Returns:
            float: The last value in the buffer.

        Raises:
            IndexError: If the buffer is empty.
        """
        if self._size == 0:
            raise IndexError("Cannot pop from an empty RingBuffer")

        self._size -= 1
        self._right_index = (self._right_index - 1 + self._capacity) % self._capacity
        return self._buffer[self._right_index]

    cpdef double popleft(self):
        """
        Remove and return the first element from the buffer.

        Returns:
            float: The first value in the buffer.

        Raises:
            IndexError: If the buffer is empty.
        """
        if self._size == 0:
            raise IndexError("Cannot pop from an empty RingBuffer")

        cdef double value = self._buffer[self._left_index]
        self._left_index = (self._left_index + 1) % self._capacity
        self._size -= 1
        return value

    cpdef double peekright(self):
        """
        Return the last element from the buffer without removing it.
        """
        if self._size == 0:
            raise IndexError("Cannot peek into an empty RingBuffer")
        return self._buffer[(self._right_index - 1 + self._capacity) % self._capacity]
    
    cpdef double peekleft(self):
        """
        Return the first element from the buffer without removing it.
        """
        if self._size == 0:
            raise IndexError("Cannot peek into an empty RingBuffer")
        return self._buffer[(self._left_index + self._capacity) % self._capacity]
    
    cpdef cnp.ndarray reset(self):
        """
        Clear the buffer and reset it to its initial state.

        Returns:
            np.ndarray: A copy of the buffer's contents before resetting.

        Note:
            This method returns the data that was in the buffer before the reset.
        """
        result = self.unwrapped()
        self.fast_reset()
        return result

    cpdef void fast_reset(self):
        """
        Quickly reset the buffer to its initial state without returning data.

        Note:
            This method clears the buffer's contents and resets indices.
            It does not return the previous data.
        """
        self._left_index = 0
        self._right_index = 0
        self._size = 0

    cpdef bint is_full(self):
        """
        Check if the buffer is full.

        Returns:
            bool: True if the buffer is full, False otherwise.
        """
        return self._size == self._capacity

    cpdef bint is_empty(self):
        """
        Check if the buffer is empty.

        Returns:
            bool: True if the buffer is empty, False otherwise.
        """
        return self._size == 0

    def __contains__(self, double value):
        """
        Check if a value is present in the buffer. 

        Parameters:
            value (float): The value to search for.

        Returns:
            bool: True if the value is in the buffer, False otherwise.
        """
        if self.is_empty():
            return False
            
        cdef u32 i, idx
        for i in range(self._size - 1, -1, -1):
            idx = (self._left_index + i) % self._capacity
            if self._buffer[idx] == value:
                return True
        return False

    def __iter__(self):
        """
        Iterate over the elements in the buffer in order from oldest to newest.

        Yields:
            float: Each value in the buffer.
        """
        cdef u32 idx = self._left_index
        for _ in range(self._size):
            yield self._buffer[idx]
            idx = (idx + 1) % self._capacity

    def __len__(self):
        """
        Get the number of elements currently in the buffer.

        Returns:
            int: The current size of the buffer.
        """
        return self._size

    def __getitem__(self, int idx):
        """
        Get the element at the given index.

        Parameters:
            idx (int): The index of the element to retrieve.

        Returns:
            float: The value at the specified index.

        Raises:
            IndexError: If the index is out of range.
        """
        # Save a few nanos by locally accessing size rather
        # than repeatadly calling the attribute.
        cdef u32 _size = self._size
        if idx < 0:
            idx += _size
        if idx < 0 or idx >= _size:
            raise IndexError(f"Index out of range; expected within ({-_size} <> {_size}) but got {idx}")

        cdef u32 fixed_idx = (self._left_index + idx) % self._capacity
        return self._buffer[fixed_idx]