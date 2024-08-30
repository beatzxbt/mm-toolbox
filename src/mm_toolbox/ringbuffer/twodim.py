import numpy as np
from numba.types import uint32, float64, int64
from numba.experimental import jitclass
from typing import Tuple


@jitclass
class RingBufferTwoDimFloat:
    """
    A 2-dimensional fixed-size circular buffer, only supporting
    floats. Optimized for super high performance, sacrificing
    safety and ease of use. Be careful!

    Parameters
    ----------
    capacity : int
        The capacity of the ring buffer (number of 1D arrays it will hold).

    sub_array_len : int
        The length of each 1D array in the buffer.
    """

    capacity: uint32
    sub_array_len: uint32
    _left_index_: uint32
    _right_index_: uint32
    _array_: float64[:, :]

    def __init__(self, capacity: int, sub_array_len: int) -> None:
        self.capacity = capacity
        self.sub_array_len = sub_array_len
        self._left_index_ = 0
        self._right_index_ = 0
        self._array_ = np.zeros(
            shape=(self.capacity, self.sub_array_len), dtype=float64
        )

    @property
    def is_full(self) -> bool:
        return (self._right_index_ - self._left_index_) == self.capacity

    @property
    def is_empty(self) -> bool:
        return self._left_index_ == 0 and self._right_index_ == 0

    @property
    def dtype(self) -> np.dtype:
        return np.float64

    @property
    def shape(self) -> Tuple[int, int]:
        return (len(self), self.sub_array_len)

    def as_array(self) -> np.ndarray:
        """
        Copy the data from this buffer into unwrapped form.

        Returns
        -------
        np.ndarray[np.float64]
            A numpy array containing the unwrapped buffer data.
        """
        if self._right_index_ <= self.capacity:
            return self._array_[self._left_index_ : self._right_index_]

        return np.concatenate(
            (
                self._array_[self._left_index_ :],
                self._array_[: self._right_index_ % self.capacity],
            )
        )

    def _fix_indices_(self) -> None:
        """
        Enforce the invariant that 0 <= self._left_index_ < self.capacity.

        This method adjusts the indices to ensure they stay within the bounds
        of the buffer's capacity.
        """
        if self._left_index_ >= self.capacity:
            self._left_index_ -= self.capacity
            self._right_index_ -= self.capacity
        elif self._left_index_ < 0:
            self._left_index_ += self.capacity
            self._right_index_ += self.capacity

    def append(self, values: np.ndarray[np.float64]) -> None:
        """
        Add a 1D array to the end of the buffer.

        Parameters
        ----------
        values : np.ndarray
            The 1D array to be added to the buffer.
        """
        assert values.size == self.sub_array_len and values.ndim == 1

        if self.is_full:
            self._left_index_ += 1

        self._array_[self._right_index_ % self.capacity, :] = values
        self._right_index_ += 1
        self._fix_indices_()

    def pop(self) -> np.ndarray[np.float64]:
        """
        Remove and return the last value from the buffer.

        Returns
        -------
        np.ndarray
            The last value removed from the buffer.

        Raises
        ------
        ValueError
            If the buffer is empty.
        """
        assert len(self) > 0, "Cannot pop from an empty RingBuffer."

        self._right_index_ -= 1
        self._fix_indices_()
        res = self._array_[self._right_index_ % self.capacity]
        return res

    def popleft(self) -> np.ndarray[np.float64]:
        """
        Remove and return the first value from the buffer.

        Returns
        -------
        np.ndarray
            The first value removed from the buffer.

        Raises
        ------
        ValueError
            If the buffer is empty.
        """
        assert len(self) > 0, "Cannot pop from an empty RingBuffer."

        res = self._array_[self._left_index_]
        self._left_index_ += 1
        self._fix_indices_()
        return res

    def __contains__(self, sub_array: np.ndarray[np.float64]) -> bool:
        assert sub_array.size == self.sub_array_len and sub_array.ndim == 1

        if self.is_empty:
            return False

        for i in range(len(self)):
            match = True

            for j in range(self.sub_array_len):
                if self._array_[i, j] != sub_array[j]:
                    match = False
                    break

            if match:
                return True

        return False

    def __eq__(self, ringbuffer: "RingBufferTwoDimFloat") -> bool:
        if isinstance(ringbuffer, RingBufferTwoDimFloat):
            return np.array_equal(ringbuffer.as_array(), self.as_array())
        return False

    def __len__(self) -> int:
        return self._right_index_ - self._left_index_

    def __getitem__(self, item: int) -> np.ndarray:
        return self.as_array()[item]

    def __str__(self) -> str:
        return (
            f"RingBufferTwoDimFloat(capacity={self.capacity}, "
            f"dtype=float64, "
            f"current_length={len(self)}, "
            f"data={self.as_array()})"
        )


@jitclass
class RingBufferTwoDimInt:
    """
    A 2-dimensional fixed-size circular buffer, only supporting
    ints. Optimized for super high performance, sacrificing
    safety and ease of use. Be careful!

    Parameters
    ----------
    capacity : int
        The capacity of the ring buffer (number of 1D arrays it will hold).

    sub_array_len : int
        The length of each 1D array in the buffer.
    """

    capacity: uint32
    sub_array_len: uint32
    _left_index_: uint32
    _right_index_: uint32
    _array_: int64[:, :]

    def __init__(self, capacity: int, sub_array_len: int) -> None:
        self.capacity = capacity
        self.sub_array_len = sub_array_len
        self._left_index_ = 0
        self._right_index_ = 0
        self._array_ = np.zeros(shape=(self.capacity, self.sub_array_len), dtype=int64)

    @property
    def is_full(self) -> bool:
        return (self._right_index_ - self._left_index_) == self.capacity

    @property
    def is_empty(self) -> bool:
        return self._left_index_ == 0 and self._right_index_ == 0

    @property
    def dtype(self) -> np.dtype:
        return np.int64

    @property
    def shape(self) -> Tuple[int, int]:
        return (len(self), self.sub_array_len)

    def as_array(self) -> np.ndarray:
        """
        Copy the data from this buffer into unwrapped form.

        Returns
        -------
        np.ndarray[np.int64]
            A numpy array containing the unwrapped buffer data.
        """
        if self._right_index_ <= self.capacity:
            return self._array_[self._left_index_ : self._right_index_]

        return np.concatenate(
            (
                self._array_[self._left_index_ :],
                self._array_[: self._right_index_ % self.capacity],
            )
        )

    def _fix_indices_(self) -> None:
        """
        Enforce the invariant that 0 <= self._left_index_ < self.capacity.

        This method adjusts the indices to ensure they stay within the bounds
        of the buffer's capacity.
        """
        if self._left_index_ >= self.capacity:
            self._left_index_ -= self.capacity
            self._right_index_ -= self.capacity
        elif self._left_index_ < 0:
            self._left_index_ += self.capacity
            self._right_index_ += self.capacity

    def append(self, values: np.ndarray[np.int64]) -> None:
        """
        Add a 1D array to the end of the buffer.

        Parameters
        ----------
        values : np.ndarray
            The 1D array to be added to the buffer.
        """
        assert values.size == self.sub_array_len and values.ndim == 1

        if self.is_full:
            self._left_index_ += 1

        self._array_[self._right_index_ % self.capacity, :] = values
        self._right_index_ += 1
        self._fix_indices_()

    def pop(self) -> np.ndarray[np.int64]:
        """
        Remove and return the last value from the buffer.

        Returns
        -------
        np.ndarray
            The last value removed from the buffer.

        Raises
        ------
        ValueError
            If the buffer is empty.
        """
        assert len(self) > 0, "Cannot pop from an empty RingBuffer."

        self._right_index_ -= 1
        self._fix_indices_()
        res = self._array_[self._right_index_ % self.capacity]
        return res

    def popleft(self) -> np.ndarray[np.int64]:
        """
        Remove and return the first value from the buffer.

        Returns
        -------
        np.ndarray
            The first value removed from the buffer.

        Raises
        ------
        ValueError
            If the buffer is empty.
        """
        assert len(self) > 0, "Cannot pop from an empty RingBuffer."

        res = self._array_[self._left_index_]
        self._left_index_ += 1
        self._fix_indices_()
        return res

    def __contains__(self, sub_array: np.ndarray[np.int64]) -> bool:
        assert sub_array.size == self.sub_array_len and sub_array.ndim == 1

        if self.is_empty:
            return False

        for i in range(len(self)):
            match = True

            for j in range(self.sub_array_len):
                if self._array_[i, j] != sub_array[j]:
                    match = False
                    break

            if match:
                return True

        return False

    def __eq__(self, ringbuffer: "RingBufferTwoDimInt") -> bool:
        if isinstance(ringbuffer, RingBufferTwoDimInt):
            return np.array_equal(ringbuffer.as_array(), self.as_array())
        return False

    def __len__(self) -> int:
        return self._right_index_ - self._left_index_

    def __getitem__(self, item: int) -> np.ndarray:
        return self.as_array()[item]

    def __str__(self) -> str:
        return (
            f"RingBufferTwoDimInt(capacity={self.capacity}, "
            f"dtype=int64, "
            f"current_length={len(self)}, "
            f"data={self.as_array()})"
        )
