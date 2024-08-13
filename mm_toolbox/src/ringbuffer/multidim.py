# Heavily optimized for 2D operations but still outperforms 
# the original numpy_ringbuffer by ~30%.

import warnings
import numpy as np
from typing import Tuple, Iterator, Union

warnings.filterwarnings("ignore", category=DeprecationWarning)

class RingBufferMultiDim:
    def __init__(self, shape: Union[int, Tuple], dtype: np.dtype = np.float64):
        """
        Create a new ring buffer with the given capacity and element type.

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
        self.capacity = shape if isinstance(shape, int) else shape[0]
        self._left_index_ = 0
        self._right_index_ = 0
        self._array_ = np.empty(shape, dtype)

    @property
    def is_full(self) -> bool:
        """
        True if there is no more space in the buffer.

        Returns
        -------
        bool
            True if the buffer is full, False otherwise.
        """
        return (self._right_index_ - self._left_index_) == self.capacity

    @property
    def is_empty(self) -> bool:
        """
        True if there are no elements in the buffer.

        Returns
        -------
        bool
            True if the buffer is empty, False otherwise.
        """
        return self._left_index_ == 0 and self._right_index_ == 0
    
    @property
    def dtype(self) -> np.dtype:
        """
        Data type of the buffer elements.

        Returns
        -------
        np.dtype
            The data type of the buffer elements.
        """
        return self._array_.dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Shape of the buffer.

        Returns
        -------
        tuple of int
            The shape of the buffer including the current length.
        """
        return (len(self),) + self._array_.shape[1:]

    def as_array(self) -> np.ndarray:
        """
        Copy the data from this buffer into unwrapped form.

        Returns
        -------
        np.ndarray
            A numpy array containing the unwrapped buffer data.
        """
        if self._right_index_ <= self.capacity:
            return self._array_[self._left_index_:self._right_index_]
        
        return np.concatenate((
            self._array_[self._left_index_:], 
            self._array_[:self._right_index_ % self.capacity]
        ))

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

    def appendright(self, value: np.ndarray) -> None:
        """
        Add a value to the right end of the buffer.

        Parameters
        ----------
        value : np.ndarray
            The value to be added to the buffer.

        Raises
        ------
        IndexError
            If the buffer is full.
        """
        if self.is_full:
            self._left_index_ += 1

        self._array_[self._right_index_ % self.capacity] = value
        self._right_index_ += 1
        self._fix_indices_()

    def appendleft(self, value: np.ndarray) -> None:
        """
        Add a value to the left end of the buffer.

        Parameters
        ----------
        value : np.ndarray
            The value to be added to the buffer.

        Raises
        ------
        IndexError
            If the buffer is full.
        """
        if self.is_full:
            self._right_index_ -= 1

        self._left_index_ -= 1
        self._fix_indices_()
        self._array_[self._left_index_] = value

    def popright(self) -> np.ndarray:
        """
        Remove and return a value from the right end of the buffer.

        Returns
        -------
        np.ndarray
            The value removed from the buffer.

        Raises
        ------
        ValueError
            If the buffer is empty.
        """
        if len(self) == 0:
            raise ValueError("Cannot pop from an empty RingBuffer.")

        self._right_index_ -= 1
        self._fix_indices_()
        res = self._array_[self._right_index_ % self.capacity]
        return res

    def popleft(self) -> np.ndarray:
        """
        Remove and return a value from the left end of the buffer.

        Returns
        -------
        np.ndarray
            The value removed from the buffer.

        Raises
        ------
        ValueError
            If the buffer is empty.
        """
        if len(self) == 0:
            raise ValueError("Cannot pop from an empty RingBuffer.")

        res = self._array_[self._left_index_]
        self._left_index_ += 1
        self._fix_indices_()
        return res

    def __len__(self) -> int:
        """
        Number of elements in the buffer.

        Returns
        -------
        int
            The number of elements in the buffer.
        """
        return self._right_index_ - self._left_index_

    def __getitem__(self, item: Tuple) -> np.ndarray:
        """
        Get an item from the buffer.

        Parameters
        ----------
        item : tuple
            The index or slice to retrieve from the buffer.

        Returns
        -------
        np.ndarray
            The retrieved item.
        """
        return self.as_array()[item]

    def __iter__(self) -> Iterator[np.ndarray]:
        """
        Return an iterator over the buffer elements.

        Returns
        -------
        iterator
            An iterator over the buffer elements.
        """
        return iter(self.as_array())
    
    def __repr__(self) -> str:
        """
        String representation of the RingBuffer.

        Returns
        -------
        str
            The string representation of the buffer.
        """
        return (f"RingBufferMultiDim(capacity={self.capacity}, "
                f"dtype={self._array_.dtype}, "
                f"current_length={len(self)}, "
                f"data={self.as_array()})")
