import numpy as np
cimport numpy as cnp

cdef class RingBufferTwoDim:
    """
    A two-dimensional circular buffer for storing sub-arrays of floats.
    """

    def __init__(self, Py_ssize_t capacity, Py_ssize_t sub_array_len):
        """
        Initialize the 2D ring buffer.

        Args:
            capacity (int): Maximum number of rows the buffer can hold.
            sub_array_len (int): Number of columns (length of each sub-array).
        """
        if capacity <= 0:
            raise ValueError(f"Negative capacity not allowed; expected >0 but got {capacity}")
        if sub_array_len <= 0:
            raise ValueError(f"Negative sub-array length not allowed; expected >0 but got {sub_array_len}")

        self._capacity = capacity
        self._sub_array_len = sub_array_len
        self._left_index = 0
        self._right_index = 0
        self._size = 0
        self._buffer = np.empty(shape=(capacity, sub_array_len), dtype=np.double)

    cpdef cnp.ndarray raw(self):
        """
        Create a copy of the entire underlying 2D buffer.

        Returns:
            np.ndarray: A 2D array copy of the entire buffer space.
        """
        return np.asarray(self._buffer).copy()
        
    cpdef cnp.ndarray unsafe_raw(self):
        """
        Return a direct view of the underlying 2D array without copying.

        Returns:
            np.ndarray: A NumPy array sharing memory with the buffer.

        Warning:
            Modifying the returned array will affect this ring buffer's state.
        """
        return np.asarray(self._buffer)

    cpdef cnp.ndarray unwrapped(self):
        """
        Return the buffer's contents in logical (unwrapped) order.

        Returns:
            np.ndarray: A 2D array in order from the oldest to the newest row.

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
    
    cpdef void unsafe_write(self, cnp.ndarray values, Py_ssize_t insert_idx=0):
        """
        Write values into the current right index row without moving the buffer indices.

        Args:
            values (np.ndarray): A 1D float array to write into the row.
            insert_idx (int): Column offset where values will be written.

        Warning:
            Does not check if the buffer is full. Does not advance indices.
            Intended to be paired with `unsafe_push`.
        """
        cdef double[:] values_view = values
        self._buffer[self._right_index, insert_idx:] = values_view

    cpdef void unsafe_push(self):
        """
        Advance indices after an `unsafe_write`, ignoring capacity checks.

        Warning:
            Overwrites oldest data if the buffer is full. Use with caution.
        """
        if self.is_full():
            self._left_index = (self._left_index + 1) % self._capacity
        else:
            self._size += 1
        
        self._right_index = (self._right_index + 1) % self._capacity

    cpdef void append(self, cnp.ndarray values):
        """
        Append a new row to the buffer.

        Args:
            values (np.ndarray): A 1D float array of length `sub_array_len`.

        Raises:
            IndexError: If `values` is not 1D or its length is incorrect.
        """
        cdef:
            Py_ssize_t values_ndim = values.ndim
            Py_ssize_t values_len = values.shape[0]
            double[:] values_view = values

        if values_ndim != 1:
            raise IndexError(f"Input array dimension mismatch; expected 1D but got {values_ndim}D")

        if values_len != self._sub_array_len:
            raise IndexError(
                f"Input array length mismatch; expected {self._sub_array_len} but got {values_len}"
            )

        if self.is_full():
            self._left_index = (self._left_index + 1) % self._capacity
        else:
            self._size += 1

        self._buffer[self._right_index, :] = values_view
        self._right_index = (self._right_index + 1) % self._capacity
            
    cpdef cnp.ndarray popright(self):
        """
        Remove and return the last (most recently added) row.

        Returns:
            np.ndarray: A 1D float array representing the row.

        Raises:
            IndexError: If the buffer is empty.
        """
        if self._size == 0:
            raise IndexError("Cannot pop from an empty RingBuffer")
        
        self._size -= 1
        self._right_index = (self._right_index - 1 + self._capacity) % self._capacity
        return np.asarray(self._buffer[self._right_index])
    
    cpdef cnp.ndarray popleft(self):
        """
        Remove and return the first (oldest) row.

        Returns:
            np.ndarray: A 1D float array representing the row.

        Raises:
            IndexError: If the buffer is empty.
        """
        if self._size == 0:
            raise IndexError("Cannot pop from an empty RingBuffer")

        values = np.asarray(self._buffer[self._left_index])
        self._left_index = (self._left_index + 1) % self._capacity
        self._size -= 1
        return values
    
    cpdef cnp.ndarray reset(self):
        """
        Reset the buffer, returning the unwrapped data prior to clearing.

        Returns:
            np.ndarray: A 2D array of the data in logical order before clearing.
        """
        result = self.unwrapped()
        self.fast_reset()
        return result
    
    cpdef void fast_reset(self):
        """
        Quickly clear the buffer without returning old data.
        """
        self._left_index = 0
        self._right_index = 0
        self._size = 0
    
    cpdef bint is_full(self):
        """
        Check if the buffer is full.

        Returns:
            bool: True if size == capacity, False otherwise.
        """
        return self._size == self._capacity
    
    cpdef bint is_empty(self):
        """
        Check if the buffer is empty.

        Returns:
            bool: True if size == 0, False otherwise.
        """
        return self._size == 0
    
    def __contains__(self, values):
        """
        Check whether a given 1D array is present in the buffer.

        Args:
            values (np.ndarray): A 1D float array to search for.

        Returns:
            bool: True if the array is found among the rows, otherwise False.
        """
        if not isinstance(values, np.ndarray):
            raise TypeError(f"Invalid input type; expected np.ndarray but got {type(values)}")
        if values.ndim != 1:
            raise ValueError(f"Invalid input dimensions; expected 1D but got {values.ndim}D")
        if values.size != self._sub_array_len:
            raise ValueError(f"Invalid input length; expected {self._sub_array_len} but got {values.size}")

        cdef:
            double[:] value
            double[:] values_view = values
            Py_ssize_t i, matching_values

        for value in self:
            matching_values = 0
            for i in range(self._sub_array_len):
                if value[i] == values_view[i]:
                    matching_values += 1
                else:
                    break

            if matching_values == self._sub_array_len:
                return True
                
        return False
    
    def __iter__(self):
        """
        Iterate over rows from oldest to newest.

        Yields:
            np.ndarray: Each row as a 1D array.
        """
        cdef Py_ssize_t idx = self._left_index
        for _ in range(self._size):
            yield self._buffer[idx]
            idx = (idx + 1) % self._capacity
    
    def __len__(self):
        """
        Number of rows currently stored in the buffer.

        Returns:
            int: The size of the buffer.
        """
        return self._size 
    
    def __getitem__(self, int idx):
        """
        Access a row by its logical index.

        Args:
            idx (int): The row index, 0-based from the oldest element. 
                Negative indices count backward from the newest element.

        Returns:
            np.ndarray: The row as a 1D float array.

        Raises:
            IndexError: If idx is out of range.
        """
        cdef Py_ssize_t _size = self._size
        if idx < 0:
            idx += _size
        if idx < 0 or idx >= _size:
            raise IndexError("Index out of range")
        cdef Py_ssize_t fixed_idx = (self._left_index + idx) % self._capacity
        return self._buffer[fixed_idx]
