import numpy as np

cimport numpy as np
from libc.stdint cimport uint32_t, int32_t


cdef class RingBufferOneDim:
    """
    A 1-dimensional fixed-size circular buffer for floats/doubles.
    """

    def __init__(self, uint32_t capacity):
        """
        Initialize a new RingBufferOneDim with the given capacity.

        Parameters:
            capacity (int): The maximum number of elements the buffer can hold.
        """
        self._capacity = capacity
        self._left_index = 0
        self._right_index = 0
        self._size = 0
        self._buffer = np.empty(capacity, dtype=np.double)

    cpdef np.ndarray raw(self):
        """
        Return a copy of the array.

        Returns
        -------
        np.ndarray: A copy of the internal 1D NumPy array representing the buffer's contents.

        Note:
            The returned array includes all allocated space, not just the filled elements.
        """
        return np.asarray(self._buffer).copy()
    
    cpdef np.ndarray unsafe_raw(self):
        """
        Return a view of the array's underlying memory without copying.

        Returns
        -------
        np.ndarray: A NumPy array view of the buffer's internal data.

        Warning:
            Modifying the returned array may affect the buffer's internal state.
            Use with caution, as no copy is made.
        """
        return np.asarray(self._buffer)

    cpdef np.ndarray unwrapped(self):
        """
        Return a copy of the buffer's contents in the correct (unwrapped) order.

        Returns
        -------
        np.ndarray: A 1D NumPy array containing the buffer's data in order from oldest to newest.
        """
        if self.is_full():
            return np.concatenate((
                self._buffer[self._left_index:], 
                self._buffer[:self._right_index]
            ))
        else:
            return np.asarray(self._buffer[:self._right_index]).copy()

    cpdef void unsafe_write(self, double value):
        """
        Directly write a value to the buffer at the current right index without updating indices.

        Parameters
        ----------
        value : float 
            The float value to be added to the buffer.

        Warning
        -------
        This method does not check if the buffer is full and does not update buffer indices.
        It is intended for use in conjunction with `unsafe_push`. Use with caution to avoid data corruption.
        """
        self._buffer[self._right_index] = value

    cpdef void unsafe_push(self):
        """
        Advance the buffer indices after writing a value.

        Warning:
            This method assumes that a value has already been written to the buffer at the current right index.
            
        Note:
            This method is intended for use in conjunction with `unsafe_write` for performance
            optimization when you are certain that the buffer management is correct.
        """
        if self.is_full():
            self._left_index = (self._left_index + 1) % self._capacity
        else:
            self._size += 1
        
        self._right_index = (self._right_index + 1) % self._capacity
        
    cpdef void append(self, double value):
        """
        Add a new element to the end of the buffer.

        Parameters:
            value (float): The float value to be added to the buffer.
        """
        if self.is_full():
            self._left_index = (self._left_index + 1) % self._capacity
        else:
            self._size += 1

        self._buffer[self._right_index] = value
        self._right_index = (self._right_index + 1) % self._capacity
        
    cpdef double popright(self):
        """
        Remove and return the last element from the buffer.

        Returns
        -------
        float: The last value in the buffer.

        Raises:
            IndexError: If the buffer is empty.
        """
        if self._size == 0:
            raise IndexError("Cannot pop from an empty RingBuffer.")

        self._size -= 1
        self._right_index = (self._right_index - 1 + self._capacity) % self._capacity
        return self._buffer[self._right_index]

    cpdef double popleft(self):
        """
        Remove and return the first element from the buffer.

        Returns
        -------
        float: The first value in the buffer.

        Raises:
            IndexError: If the buffer is empty.
        """
        if self._size == 0:
            raise IndexError("Cannot pop from an empty RingBuffer.")

        cdef double value = self._buffer[self._left_index]
        self._left_index = (self._left_index + 1) % self._capacity
        self._size -= 1
        return value

    cpdef np.ndarray reset(self):
        """
        Clear the buffer and reset it to its initial state.

        Returns
        -------
        np.ndarray: A copy of the buffer's contents before resetting.

        Note:
            This method returns the data that was in the buffer before the reset.
        """
        result = self.unwrapped()
        self._buffer[:] = 0.0
        self._left_index = 0
        self._right_index = 0
        self._size = 0
        return result

    cpdef void fast_reset(self):
        """
        Quickly reset the buffer to its initial state without returning data.

        Note:
            This method clears the buffer's contents and resets indices.
            It does not return the previous data.
        """
        self._buffer[:] = 0.0
        self._left_index = 0
        self._right_index = 0
        self._size = 0

    cpdef bint is_full(self):
        """
        Check if the buffer is full.

        Returns
        -------
        bool: True if the buffer is full, False otherwise.
        """
        return self._size == self._capacity

    cpdef bint is_empty(self):
        """
        Check if the buffer is empty.

        Returns
        -------
        bool: True if the buffer is empty, False otherwise.
        """
        return self._size == 0

    def __contains__(self, double value):
        """
        Check if a value is present in the buffer.

        Parameters:
            value (float): The value to search for.

        Returns
        -------
        bool: True if the value is in the buffer, False otherwise.
        """
        cdef uint32_t i 

        for i in range(self._size):
            if self._buffer[i] == value:
                return True
            
        return False
    
    def __iter__(self):
        """
        Iterate over the elements in the buffer in order from oldest to newest.

        Yields:
            float: Each value in the buffer.
        """
        cdef uint32_t idx = self._left_index
        for _ in range(self._size):
            yield self._buffer[idx]
            idx = (idx + 1) % self._capacity

    def __len__(self):
        """
        Get the number of elements currently in the buffer.

        Returns
        -------
        int: The current size of the buffer.
        """
        return self._size 

    def __getitem__(self, int idx):
        cdef int32_t _size = self._size
        if idx < 0:
            idx += _size
        if idx < 0 or idx >= _size:
            raise IndexError("Index out of range.")
        
        cdef int32_t fixed_idx = (self._left_index + idx) % self._capacity
        return self._buffer[fixed_idx]
