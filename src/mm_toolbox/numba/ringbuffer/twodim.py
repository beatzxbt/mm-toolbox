import numpy as np
from numba.types import uint64, float64
from numba.experimental import jitclass

@jitclass
class RingBufferTwoDim:
    """
    A two-dimensional circular buffer for storing sub-arrays of floats.
    """

    _capacity: uint64
    _sub_array_len: uint64
    _left_index: uint64
    _right_index: uint64
    _size: uint64
    _buffer: float64[:, :]

    def __init__(self, capacity: int, sub_array_len: int):
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
        self._buffer = np.empty(shape=(capacity, sub_array_len), dtype=np.float64)

    def raw(self) -> np.ndarray:
        """
        Create a copy of the entire underlying 2D buffer.

        Returns:
            np.ndarray: A 2D array copy of the entire buffer space.
        """
        return np.asarray(self._buffer).copy()
        
    def unsafe_raw(self) -> np.ndarray:
        """
        Return a direct view of the underlying 2D array without copying.

        Returns:
            np.ndarray: A NumPy array sharing memory with the buffer.

        Warning:
            Modifying the returned array will affect this ring buffer's state.
        """
        return np.asarray(self._buffer)

    def unwrapped(self) -> np.ndarray:
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
    
    def unsafe_write(self, values: np.ndarray, insert_idx: int=0) -> None:
        """
        Write values into the current right index row without moving the buffer indices.

        Args:
            values (np.ndarray): A 1D float array to write into the row.
            insert_idx (int): Column offset where values will be written.

        Warning:
            Does not check if the buffer is full. Does not advance indices.
            Intended to be paired with `unsafe_push`.
        """
        self._buffer[self._right_index, insert_idx:] = values

    def unsafe_push(self) -> None:
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

    def append(self, values: np.ndarray) -> None:
        """
        Append a new row to the buffer.

        Args:
            values (np.ndarray): A 1D float array of length `sub_array_len`.

        Raises:
            IndexError: If `values` is not 1D or its length is incorrect.
        """
        values_ndim = values.ndim
        values_len = values.shape[0]

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

        self._buffer[self._right_index, :] = values
        self._right_index = (self._right_index + 1) % self._capacity
            
    def popright(self) -> np.ndarray:
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
    
    def popleft(self) -> np.ndarray:
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
    
    def reset(self) -> np.ndarray:
        """
        Reset the buffer, returning the unwrapped data prior to clearing.

        Returns:
            np.ndarray: A 2D array of the data in logical order before clearing.
        """
        result = self.unwrapped()
        self.fast_reset()
        return result
    
    def fast_reset(self) -> None:
        """
        Quickly clear the buffer without returning old data.
        """
        self._left_index = 0
        self._right_index = 0
        self._size = 0
    
    def is_full(self) -> bool:
        """
        Check if the buffer is full.

        Returns:
            bool: True if size == capacity, False otherwise.
        """
        return self._size == self._capacity
    
    def is_empty(self) -> bool:
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
        # NOTE: Numba cannot support Python object input types anyways, so we just
        # trust that it is a numpy array. The isinstance check is not possible in jitclasses
        #
        # if not isinstance(values, np.ndarray):
        #     raise TypeError(f"Invalid input type; expected np.ndarray but got {type(values)}")

        if values.ndim != 1:
            raise ValueError(f"Invalid input dimensions; expected 1D but got {values.ndim}D")
        if values.size != self._sub_array_len:
            raise ValueError(f"Invalid input length; expected {self._sub_array_len} but got {values.size}")

        for i in range(self._size):
            idx = (self._left_index + i) % self._capacity
            current_row = self._buffer[idx]
            
            all_match = True
            for i in range(self._sub_array_len):
                if current_row[i] != values[i]:
                    all_match = False
                    break
                    
            if all_match:
                return True
                
        return False
    
    # NOTE: This method is unfortunately not supported within jitclasses.
    # You can work around this by doing 'for value in self.unwrapped()'
    # although with a performance hit. Hopefully this is supported soon,
    # at which point the code below can be uncommented. 
    #
    # def __iter__(self):
    #     """
    #     Iterate over rows from oldest to newest.

    #     Yields:
    #         np.ndarray: Each row as a 1D array.
    #     """
    #     idx = self._left_index
    #     for _ in range(self._size):
    #         yield self._buffer[idx]
    #         idx = (idx + 1) % self._capacity
    
    def __len__(self):
        """
        Number of rows currently stored in the buffer.

        Returns:
            int: The size of the buffer.
        """
        return self._size 
    
    def __getitem__(self, idx: int):
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
        _size = self._size
        if idx < 0:
            idx += _size
        if idx < 0 or idx >= _size:
            raise IndexError("Index out of range")
        fixed_idx = (self._left_index + idx) % self._capacity
        return self._buffer[fixed_idx]
