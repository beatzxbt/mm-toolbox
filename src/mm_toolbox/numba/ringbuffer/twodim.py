import numpy as np
from numba.types import uint64, float64
from numba.experimental import jitclass


@jitclass
class RingBufferTwoDim:
    """
    A 2-dimensional fixed-size circular buffer for floats/doubles.
    """

    _capacity: uint64
    _sub_array_len: uint64
    _left_index: uint64
    _right_index: uint64
    _size: uint64
    _buffer: float64[:, :]

    def __init__(self, capacity: int, sub_array_len: int):
        """
        Parameters:
            capacity (int): The maximum number of elements the buffer can hold.
            sub_array_len (int): The number of columns (length of each sub-array).
        """
        self._capacity = capacity
        self._sub_array_len = sub_array_len
        self._left_index = 0
        self._right_index = 0
        self._size = 0
        self._buffer = np.empty(shape=(capacity, sub_array_len), dtype=float64)

    def raw(self) -> np.ndarray:
        """
        Return a copy of the internal buffer array.

        Returns:
            np.ndarray: A copy of the internal 1D NumPy array representing the buffer's contents.

        Note:
            The returned array includes all allocated space, not just the filled elements.
        """
        return self._buffer.copy()

    def unsafe_raw(self) -> np.ndarray:
        """
        Return a view of the internal buffer array without copying.

        Returns:
            np.ndarray: A NumPy array view of the buffer's internal data.

        Warning:
            Modifying the returned array may affect the buffer's internal state.
            Use with caution, as no copy is made.
        """
        return self._buffer

    def unwrapped(self) -> np.ndarray:
        """
        Return a copy of the buffer's contents in the correct (unwrapped) order.

        Returns:
            np.ndarray: A 1D NumPy array containing the buffer's data in order from oldest to newest.
        """
        if self.is_full():
            return np.concatenate(
                (self._buffer[self._left_index :], self._buffer[: self._right_index])
            )
        else:
            return self._buffer[: self._right_index].copy()

    def unsafe_write(self, values: np.ndarray, insert_idx: int):
        """
        Directly write a value to the buffer at the current right index without updating indices.

        Parameters:
            values (np.ndarray): The values to be added to the buffer.
            insert_idx (int): The starting column index within the sub-array where values will be written.

        Warning:
            This method does not check if the buffer is full and does not update buffer indices.
            It is intended for use in conjunction with `unsafe_push`. Use with caution to avoid data corruption.
        """
        start_idx = insert_idx
        end_idx = values.size + insert_idx
        self._buffer[self._right_index, start_idx:end_idx] = values

    def unsafe_push(self):
        """
        Advance the buffer indices after writing a value, without checking for buffer fullness.

        Warning:
            This method assumes that a value has already been written to the buffer at the current right index.
            It updates the buffer indices accordingly. It does not check if the buffer is full.
            If the buffer is full, it will overwrite the oldest data. Use with caution to avoid data corruption.

        Note:
            This method is intended for use in conjunction with `unsafe_write` for performance
            optimization when you are certain that the buffer management is correct.
        """
        if self.is_full():
            self._left_index = (self._left_index + 1) % self._capacity
        else:
            self._size += 1

        self._right_index = (self._right_index + 1) % self._capacity

    def append(self, values: np.ndarray):
        """
        Add a new element to the end of the buffer.

        Parameters:
            value (float): The float value to be added to the buffer.
        """
        # Required for compilation for some reason.
        assert values.size == self._sub_array_len, f"Invalid array len; expected {self._sub_array_len} but got {values.size}"
        assert values.ndim == 1, f"Invalid array ndim; expected 1D but got {values.ndim}D"

        if self.is_full():
            self._left_index = (self._left_index + 1) % self._capacity
        else:
            self._size += 1

        self._buffer[self._right_index, :] = values
        self._right_index = (self._right_index + 1) % self._capacity

    def popright(self) -> float:
        """
        Remove and return the last element from the buffer.

        Returns:
            float: The last value in the buffer.

        Raises:
            IndexError: If the buffer is empty.
        """
        if self._size == 0:
            raise IndexError("Cannot pop from an empty RingBuffer.")

        self._size -= 1
        self._right_index = (self._right_index - 1 + self._capacity) % self._capacity
        return self._buffer[self._right_index]

    def popleft(self) -> float:
        """
        Remove and return the first element from the buffer.

        Returns:
            float: The first value in the buffer.

        Raises:
            IndexError: If the buffer is empty.
        """
        if self._size == 0:
            raise IndexError("Cannot pop from an empty RingBuffer.")

        values = self._buffer[self._left_index]
        self._left_index = (self._left_index + 1) % self._capacity
        self._size -= 1
        return values

    def reset(self) -> np.ndarray:
        """
        Clear the buffer and reset it to its initial state.

        Returns:
            np.ndarray: A copy of the buffer's contents before resetting.

        Note:
            This method returns the data that was in the buffer before the reset.
        """
        result = self.unwrapped()
        self._buffer.fill(0.0)
        self._left_index = 0
        self._right_index = 0
        self._size = 0
        return result

    def fast_reset(self):
        """
        Quickly reset the buffer to its initial state without returning data.

        Note:
            This method clears the buffer's contents and resets indices.
            It does not return the previous data.
        """
        self._buffer.fill(0.0)
        self._left_index = 0
        self._right_index = 0
        self._size = 0

    def is_full(self) -> bool:
        """
        Check if the buffer is full.

        Returns:
            bool: True if the buffer is full, False otherwise.
        """
        return self._size == self._capacity

    def is_empty(self) -> bool:
        """
        Check if the buffer is empty.

        Returns:
            bool: True if the buffer is empty, False otherwise.
        """
        return self._size == 0

    def __contains__(self, values: np.ndarray):
        """
        Check if the array is present in the buffer.

        Parameters:
            values (np.ndarray): The array to search for.

        Returns:
            bool: True if the array is in the buffer, False otherwise.
        """
        if self.is_empty():
            return False
        
        try:
            # Required for compilation for some reason.
            assert values.size == self._sub_array_len
            assert values.ndim == 1
        except Exception:
            # AssertionError is not supported as an exception 
            # handler by numba, so we must raise a general one
            # to catch it. Not ideal, improvement needed.
            return False
        
        for i in range(self._size):
            if np.array_equal(self._buffer[i], values):
                return True

        return False
    
    # def __iter__(self):
    #     """
    #     Iterate over the elements in the buffer in order from oldest to newest.

    #     Yields:
    #         float: Each value in the buffer.
    #     """
    #     raise NotImplementedError("Numba does not support '__iter__', iterate over '.unwrapped()' instead.")

    def __len__(self):
        """
        Get the number of elements currently in the buffer.

        Returns:
            int: The current size of the buffer.
        """
        return self._size
    
    def __getitem__(self, idx: int):
        """
        Get the element at the given index.

        Parameters:
            idx (int): The index of the element to retrieve.

        Returns:
            float: The value at the specified index.

        Raises:
            IndexError: If the index is out of range.
        """
        if idx < 0:
            idx += self._size
        if idx < 0 or idx >= self._size:
            raise IndexError("Index out of range.")

        fixed_idx = (self._left_index + idx) % self._capacity
        return self._buffer[fixed_idx]