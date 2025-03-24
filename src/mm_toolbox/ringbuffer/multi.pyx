import numpy as np
cimport numpy as cnp

cdef class RingBufferMulti:
    """
    A fixed-size circular buffer supporting up to 2D shapes.
    """

    def __init__(self, object shape, object dtype):
        """
        Initialize the circular buffer with a specified shape and dtype.

        Args:
            shape (int or tuple of int): The shape of the buffer. For a 1D buffer,
                this can be a single integer (the capacity). For a 2D buffer,
                use (capacity, sub_array_length). Shapes with more than 2D
                are not supported.
            dtype (numpy.dtype or type): The dtype (or coercible type) for the buffer.
                Must be numeric, string, or bytes sub-dtype in NumPy.

        Raises:
            ValueError: If shape has more than 2 dimensions or capacity is <= 0.
            TypeError: If shape is not int, tuple, or list, or if dtype is unsupported.
        """
        if isinstance(shape, int):
            if shape <= 0:
                raise ValueError(f"Capacity must be positive; expected >0 but got {shape}")
            self._ndim = 1
            self._capacity = shape
            self._sub_array_len = 0  # Ignored for 1D
        elif isinstance(shape, (tuple, list)):
            if len(shape) > 2:
                raise ValueError("Shape with more than 2 dimensions is not supported")
            if len(shape) == 0 or shape[0] <= 0:
                raise ValueError(f"Capacity must be positive; expected >0 but got {shape[0] if len(shape) > 0 else 'empty shape'}")
            self._ndim = len(shape)
            self._capacity = shape[0]
            self._sub_array_len = shape[1] if len(shape) > 1 else 0
        else:
            raise TypeError(
                f"Invalid shape type; expected (int | tuple | list) but got {type(shape)}"
            )

        # Ensure the dtype provided is a supported numpy dtype. This is also key
        # for append() and __contains__() methods which involve type checking.
        if not (
            np.issubdtype(dtype, np.number)
            or np.issubdtype(dtype, np.str_)
            or np.issubdtype(dtype, np.bytes_)
        ):
            raise TypeError(f"Unsupported dtype; expected (numeric | string | bytes) but got {dtype}")

        self._left_index = 0
        self._right_index = 0
        self._size = 0
        self._dtype = dtype
        self._buffer = np.empty(shape, dtype=dtype)

    cpdef cnp.ndarray raw(self):
        """
        Create a copy of the entire buffer array.

        Returns:
            np.ndarray: A copy of the internal buffer (including unused capacity).
        """
        return self._buffer.copy()
    
    cpdef cnp.ndarray unsafe_raw(self):
        """
        Return a direct view of the underlying buffer array without copying.

        Returns:
            np.ndarray: A NumPy array sharing memory with the buffer.

        Warning:
            Modifying the returned array also changes the buffer's internal state.
        """
        return self._buffer

    cpdef cnp.ndarray unwrapped(self):
        """
        Return the buffer contents in logical order (oldest to newest).

        Returns:
            np.ndarray: A copy of the buffer's data from oldest to newest.
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

    cpdef void unsafe_write(self, object value, Py_ssize_t insert_idx=0):
        """
        Write a value directly into the buffer at the current right index.

        Args:
            value (object): The value to write. Must be compatible with the buffer's dtype.
            insert_idx (Py_ssize_t, optional): For 2D buffers, which sub-array index to write from.

        Warning:
            This does not check buffer capacity or update indices. Use `unsafe_push` afterward.
        """
        if self._ndim == 1:
            self._buffer[self._right_index] = value
        else:
            if insert_idx >= self._sub_array_len:
                # Silently handle out-of-bounds index by capping at max length. This 
                # is risky if not known, but for the sake of speed and the fact that
                # this is an unsafe function this wont raise an IndexError. If you just
                # spent a few hours debugging this, im sorry <3
                insert_idx = self._sub_array_len - 1
            self._buffer[self._right_index, insert_idx:] = value

    cpdef void unsafe_push(self):
        """
        Advance the right index after an `unsafe_write`.

        Warning:
            Overwrites the oldest data if the buffer is full. This does not check capacity.
        """
        if self.is_full():
            self._left_index = (self._left_index + 1) % self._capacity
        else:
            self._size += 1
        
        self._right_index = (self._right_index + 1) % self._capacity

    cpdef void append(self, object value):
        """
        Append a new element (scalar or 1D array) to the buffer.

        Args:
            value (object): If the buffer is 1D, this should be a scalar. For a 2D buffer,
                this should be a 1D array-like of length `sub_array_len`.

        Raises:
            TypeError: If `value` cannot be cast to the buffer's dtype.
            ValueError: If `value` has invalid shape or length for a 2D buffer.
        """
        cdef:
            object      casted_value
            Py_ssize_t  value_ndim
            Py_ssize_t  value_len
            object      value_dtype
        
        if self._ndim == 1:
            try:
                casted_value = self._dtype(value)
            except (TypeError, ValueError):
                raise TypeError(f"Failed to cast {value} to dtype {self._dtype}")
            
            self._buffer[self._right_index] = casted_value

        else:
            if isinstance(value, (tuple, list)):
                try:
                    casted_value = np.asarray(value, dtype=self._dtype)
                except (TypeError, ValueError):
                    raise TypeError(f"Failed to cast {value} to dtype {self._dtype}")

                value_ndim = casted_value.ndim
                value_len = casted_value.shape[0]

                if value_ndim != 1:
                    raise ValueError(f"Invalid dimensions; expected 1D but got {value_ndim}D")
                if value_len != self._sub_array_len:
                    raise ValueError(f"Invalid length; expected {self._sub_array_len} but found {value_len}")

                self._buffer[self._right_index, :] = casted_value

            elif isinstance(value, np.ndarray):
                value_dtype = value.dtype
                value_ndim = value.ndim
                
                if value_ndim != 1:
                    raise ValueError(f"Invalid dimensions; expected 1D but found {value_ndim}D")
                
                value_len = value.shape[0]
                if value_len != self._sub_array_len:
                    raise ValueError(f"Invalid length; expected {self._sub_array_len} but found {value_len}")
                
                # Performance optimization: only convert dtype if needed
                if value_dtype != self._dtype:
                    try:
                        value = value.astype(self._dtype, copy=False)
                    except (TypeError, ValueError):
                        raise TypeError(f"Invalid dtype; expected {self._dtype} but got {value_dtype}")

                self._buffer[self._right_index, :] = value

            else:
                raise TypeError(
                    f"Invalid type; expected (np.ndarray | tuple | list) but got {type(value)}"
                )

        self.unsafe_push()

    cpdef object popright(self):
        """
        Remove and return the most recently appended element.

        Returns:
            object: The last stored value.

        Raises:
            IndexError: If the buffer is empty.
        """
        if self._size == 0:
            raise IndexError("Cannot pop from an empty RingBuffer")

        self._size -= 1
        self._right_index = (self._right_index - 1 + self._capacity) % self._capacity
        
        # Return a copy to avoid potential data corruption
        if self._ndim == 1:
            return self._dtype(self._buffer[self._right_index])
        else:
            return self._buffer[self._right_index].copy()

    cpdef object popleft(self):
        """
        Remove and return the oldest element in the buffer.

        Returns:
            object: The first stored value.

        Raises:
            IndexError: If the buffer is empty.
        """
        if self._size == 0:
            raise IndexError("Cannot pop from an empty RingBuffer")

        # Return a copy to avoid potential data corruption
        if self._ndim == 1:
            value = self._dtype(self._buffer[self._left_index])
        else:
            value = self._buffer[self._left_index].copy()
            
        self._left_index = (self._left_index + 1) % self._capacity
        self._size -= 1
        return value

    cpdef cnp.ndarray reset(self):
        """
        Clear the buffer and return its contents in logical order before clearing.

        Returns:
            cnp.ndarray: The data in the buffer (unwrapped) prior to reset.
        """
        cdef cnp.ndarray result = self.unwrapped()
        self._buffer = np.empty_like(self._buffer)
        self._left_index = 0
        self._right_index = 0
        self._size = 0
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
        Check if the buffer is currently full.

        Returns:
            bool: True if buffer size == capacity, False otherwise.
        """
        return self._size == self._capacity

    cpdef bint is_empty(self):
        """
        Check if the buffer is empty.

        Returns:
            bool: True if size == 0, False otherwise.
        """
        return self._size == 0

    def __contains__(self, object other):
        """
        Check whether a given value is stored in the buffer.

        Args:
            other (object): The value to search for. Must be compatible with the buffer's dtype and shape.

        Returns:
            bool: True if found, False otherwise.
        """
        if self.is_empty():
            return False

        other_type = type(other)

        if self._ndim == 1:
            if other_type in (list, tuple, np.ndarray):
                return False
            # Potentially check dtype, or do direct comparisons
            for value in self._buffer:
                if value == other:
                    return True
            return False

        else:
            if other_type not in (list, tuple, np.ndarray):
                return False

            if other_type in (list, tuple):
                try:
                    other_arr = np.asarray(other, dtype=self._dtype)
                except (TypeError, ValueError):
                    return False
            else:
                if other.dtype != self._dtype:
                    try:
                        other_arr = other.astype(self._dtype, copy=False)
                    except (TypeError, ValueError):
                        return False
                else:
                    other_arr = other

            if other_arr.ndim != 1 or other_arr.shape[0] != self._sub_array_len:
                return False

            for row in self._buffer:
                if np.array_equal(row, other_arr):
                    return True

            return False

    def __iter__(self):
        """
        Yield elements in logical order from oldest to newest.

        Yields:
            object: Each element in the buffer.
        """
        cdef Py_ssize_t idx = self._left_index
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
        Access an element by logical index.

        Args:
            idx (int): The position of the desired element, where 0 is the oldest.

        Returns:
            object: The element at the specified index.

        Raises:
            IndexError: If the index is out of range.
        """
        cdef Py_ssize_t _size = self._size
        if idx < 0:
            idx += _size
        if idx < 0 or idx >= _size:
            raise IndexError("Index out of range")

        cdef Py_ssize_t fixed_idx = (self._left_index + idx) % self._capacity
        
        # Return a copy to avoid potential data corruption
        if self._ndim == 1:
            return self._dtype(self._buffer[fixed_idx])
        else:
            return self._buffer[fixed_idx].copy()
