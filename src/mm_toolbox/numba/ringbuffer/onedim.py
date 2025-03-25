import numpy as np
from numba.types import uint64, float64
from numba.experimental import jitclass


@jitclass
class RingBufferOneDim:
    """
    A 1-dimensional fixed-size circular buffer for floats/doubles.
    """

    _capacity: uint64
    _left_index: uint64
    _right_index: uint64
    _size: uint64
    _buffer: float64[:]

    def __init__(self, capacity: int):
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
        self._buffer = np.empty(shape=capacity, dtype=float64)

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

    def unsafe_write(self, value: float) -> None:
        """
        Directly write a value to the buffer at the current right index without updating indices.

        Parameters:
            value (float): The float value to be added to the buffer.

        Warning:
            This method does not check if the buffer is full and does not update buffer indices.
            It is intended for use in conjunction with `unsafe_push`. Use with caution to avoid data corruption.
        """
        self._buffer[self._right_index] = value

    def unsafe_push(self) -> None:
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
        
    def append(self, value: float) -> None:
        """
        Add a new element to the end of the buffer.

        Parameters:
            value (float): The float value to be added to the buffer.
        """
        self.unsafe_write(value)
        self.unsafe_push()
        
    def popright(self) -> float:
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

    def popleft(self) -> float:
        """
        Remove and return the first element from the buffer.

        Returns:
            float: The first value in the buffer.

        Raises:
            IndexError: If the buffer is empty.
        """
        if self._size == 0:
            raise IndexError("Cannot pop from an empty RingBuffer")

        value = self._buffer[self._left_index]
        self._left_index = (self._left_index + 1) % self._capacity
        self._size -= 1
        return value

    def reset(self) -> np.ndarray:
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

    def fast_reset(self) -> None:
        """
        Quickly reset the buffer to its initial state without returning data.

        Note:
            This method clears the buffer's contents and resets indices.
            It does not return the previous data.
        """
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

    def __contains__(self, value: float):
        """
        Check if a value is present in the buffer.

        Parameters:
            value (float): The value to search for.

        Returns:
            bool: True if the value is in the buffer, False otherwise.
        """
        if self.is_empty():
            return False
            
        for i in range(self._size):
            idx = (self._left_index + i) % self._capacity
            if self._buffer[idx] == value:
                return True
        return False

    # NOTE: This method is unfortunately not supported within jitclasses.
    # You can work around this by doing 'for value in self.unwrapped()'
    # although with a performance hit. Hopefully this is supported soon,
    # at which point the code below can be uncommented. 
    # def __iter__(self):
    #     """
    #     Iterate over the elements in the buffer in order from oldest to newest.

    #     Yields:
    #         float: Each value in the buffer.
    #     """
    #     idx = self._left_index
    #     for _ in range(self._size):
    #         yield self._buffer[idx]
    #         idx = (idx + 1) % self._capacity

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
        # Save a few nanos by locally accessing size rather
        # than repeatadly calling the attribute.
        _size = self._size
        if idx < 0:
            idx += _size
        if idx < 0 or idx >= _size:
            raise IndexError(f"Index out of range; expected within ({-_size} <> {_size}) but got {idx}")

        fixed_idx = (self._left_index + idx) % self._capacity
        return self._buffer[fixed_idx]