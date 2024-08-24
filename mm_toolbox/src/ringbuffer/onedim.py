import numpy as np
from numba.types import uint32, int64, float64
from numba.experimental import jitclass
from typing import Tuple


@jitclass
class RingBufferSingleDimFloat:
    """
    A 1-dimensional fixed-size circular buffer, only supporting
    floats. Optimized for super high performance, sacrificing
    safety and ease of use. Be careful!

    Parameters
    ----------
    shape : int or tuple of int
        The shape of the ring buffer. If an integer is provided, it
        specifies the capacity. If a tuple is provided, it specifies
        the shape including the capacity as the first element.

    dtype : data-type, optional
        Desired type of buffer elements. Use a type like (float, 2) to
        produce a buffer with shape (N, 2). Default is np.float64.
    """

    capacity: uint32
    _left_index_: uint32
    _right_index_: uint32
    _array_: float64[:]

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._left_index_ = 0
        self._right_index_ = 0
        self._array_ = np.zeros(shape=capacity, dtype=float64)

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
    def shape(self) -> Tuple[int, ...]:
        return (len(self),)

    def as_array(self) -> np.ndarray:
        """
        Copy the data from this buffer into unwrapped form.

        Returns
        -------
        np.ndarray
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
        Corrects the indices if they exceed the buffer's capacity.

        This method adjusts the left and right indices to ensure they
        stay within the bounds of the buffer's capacity.
        """
        if self._left_index_ >= self.capacity:
            self._left_index_ -= self.capacity
            self._right_index_ -= self.capacity
        elif self._left_index_ < 0:
            self._left_index_ += self.capacity
            self._right_index_ += self.capacity

    def append(self, value: float) -> None:
        """
        Adds an element to the end of the buffer.

        Parameters
        ----------
        value : float
            The value to be added to the buffer.
        """
        if self.is_full:
            self._left_index_ += 1

        self._array_[self._right_index_ % self.capacity] = value
        self._right_index_ += 1
        self._fix_indices_()

    def popright(self) -> float:
        """
        Removes and returns an element from the end of the buffer.

        Returns
        -------
        float
            The value removed from the buffer.

        Raises
        ------
        IndexError
            If the buffer is empty.
        """
        assert len(self) > 0, "Cannot pop from an empty RingBuffer"

        self._right_index_ -= 1
        self._fix_indices_()
        res = self._array_[self._right_index_ % self.capacity]
        return res

    def popleft(self) -> float:
        """
        Removes and returns an element from the start of the buffer.

        Returns
        -------
        float
            The value removed from the buffer.

        Raises
        ------
        IndexError
            If the buffer is empty.
        """
        assert len(self) > 0, "Cannot pop from an empty RingBuffer"

        res = self._array_[self._left_index_]
        self._left_index_ += 1
        self._fix_indices_()
        return res

    def reset(self) -> np.ndarray[float]:
        """
        Clears the buffer and resets it to its initial state.

        Returns
        -------
        np.ndarray
            The contents of the buffer before it was reset.
        """
        res = self.as_array()
        self._array_.fill(0.0)
        self._left_index_ = 0
        self._right_index_ = 0
        return res

    def __contains__(self, num: float) -> bool:
        return np.any(num == self._array_)

    def __eq__(self, ringbuffer: "RingBufferSingleDimFloat") -> bool:
        assert isinstance(ringbuffer, RingBufferSingleDimFloat)
        return np.array_equal(ringbuffer.as_array(), self.as_array())

    def __len__(self) -> int:
        return self._right_index_ - self._left_index_

    def __getitem__(self, item: int) -> float:
        return self.as_array()[item]

    def __str__(self) -> str:
        return (
            f"RingBufferSingleDimFloat(capacity={self.capacity}, "
            f"dtype=float64, "
            f"current_length={len(self)}, "
            f"data={self.as_array()})"
        )


@jitclass
class RingBufferSingleDimInt:
    """
    A 1-dimensional fixed-size circular buffer, only supporting
    ints. Optimized for super high performance, sacrificing
    safety and ease of use. Be careful!

    Parameters
    ----------
    shape : int or tuple of int
        The shape of the ring buffer. If an integer is provided, it
        specifies the capacity. If a tuple is provided, it specifies
        the shape including the capacity as the first element.

    dtype : data-type, optional
        Desired type of buffer elements. Use a type like (int, 2) to
        produce a buffer with shape (N, 2). Default is np.int64.
    """

    capacity: uint32
    _left_index_: uint32
    _right_index_: uint32
    _array_: int64[:]

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._left_index_ = 0
        self._right_index_ = 0
        self._array_ = np.zeros(shape=capacity, dtype=int64)

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
    def shape(self) -> Tuple[int, ...]:
        return (len(self),)

    def as_array(self) -> np.ndarray:
        """
        Copy the data from this buffer into unwrapped form.

        Returns
        -------
        np.ndarray
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
        Corrects the indices if they exceed the buffer's capacity.

        This method adjusts the left and right indices to ensure they
        stay within the bounds of the buffer's capacity.
        """
        if self._left_index_ >= self.capacity:
            self._left_index_ -= self.capacity
            self._right_index_ -= self.capacity
        elif self._left_index_ < 0:
            self._left_index_ += self.capacity
            self._right_index_ += self.capacity

    def append(self, value: int) -> None:
        """
        Adds an element to the end of the buffer.

        Parameters
        ----------
        value : int
            The value to be added to the buffer.
        """
        if self.is_full:
            self._left_index_ += 1

        self._array_[self._right_index_ % self.capacity] = value
        self._right_index_ += 1
        self._fix_indices_()

    def popright(self) -> int:
        """
        Removes and returns an element from the end of the buffer.

        Returns
        -------
        int
            The value removed from the buffer.

        Raises
        ------
        IndexError
            If the buffer is empty.
        """
        assert len(self) > 0, "Cannot pop from an empty RingBuffer"

        self._right_index_ -= 1
        self._fix_indices_()
        res = self._array_[self._right_index_ % self.capacity]
        return res

    def popleft(self) -> int:
        """
        Removes and returns an element from the start of the buffer.

        Returns
        -------
        int
            The value removed from the buffer.

        Raises
        ------
        IndexError
            If the buffer is empty.
        """
        assert len(self) > 0, "Cannot pop from an empty RingBuffer"

        res = self._array_[self._left_index_]
        self._left_index_ += 1
        self._fix_indices_()
        return res

    def reset(self) -> np.ndarray[int]:
        """
        Clears the buffer and resets it to its initial state.

        Returns
        -------
        np.ndarray
            The contents of the buffer before it was reset.
        """
        res = self.as_array()
        self._array_.fill(0.0)
        self._left_index_ = 0
        self._right_index_ = 0
        return res

    def __contains__(self, num: int) -> bool:
        return np.any(num == self._array_)

    def __eq__(self, ringbuffer: "RingBufferSingleDimInt") -> bool:
        assert isinstance(ringbuffer, RingBufferSingleDimInt)
        return np.array_equal(ringbuffer.as_array(), self.as_array())

    def __len__(self) -> int:
        return self._right_index_ - self._left_index_

    def __getitem__(self, item: int) -> int:
        return self.as_array()[item]

    def __str__(self) -> str:
        return (
            f"RingBufferSingleDimInt(capacity={self.capacity}, "
            f"dtype=int64, "
            f"current_length={len(self)}, "
            f"data={self.as_array()})"
        )
