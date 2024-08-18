import warnings
import numpy as np
from typing import Tuple, Union

from mm_toolbox.src.numba import nbisin

warnings.filterwarnings("ignore", category=DeprecationWarning)

class RingBufferMultiDim:
    """
    A multi-dimensional fixed-size circular buffer for any numpy dtype.

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

    def __init__(self, shape: Union[int, Tuple], dtype: np.dtype = np.float64):
        self.capacity = shape if isinstance(shape, int) else shape[0]
        self._dtype_ = dtype 
        self._left_index_ = 0
        self._right_index_ = 0
        self._array_ = np.empty(shape, self._dtype_)

    @property
    def is_full(self) -> bool:
        return (self._right_index_ - self._left_index_) == self.capacity

    @property
    def is_empty(self) -> bool:
        return self._left_index_ == 0 and self._right_index_ == 0
    
    @property
    def dtype(self) -> np.dtype:
        return self._dtype_

    @property
    def shape(self) -> Tuple[int, ...]:
        return (len(self),) + self._array_.shape[1:]

    def as_array(self) -> np.ndarray[Union[int, float, str, bytes, np.ndarray]]:
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

    def append(self, value: Union[int, float, str, bytes, np.ndarray]) -> None:
        """
        Add a value to the right end of the buffer.

        Parameters
        ----------
        value : Union[int, float, str, bytes, np.ndarray]
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

    def popright(self) -> Union[int, float, str, bytes, np.ndarray]:
        """
        Remove and return a value from the right end of the buffer.

        Returns
        -------
        Union[int, float, str, bytes, np.ndarray]
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

    def popleft(self) -> Union[int, float, str, bytes, np.ndarray]:
        """
        Remove and return a value from the left end of the buffer.

        Returns
        -------
        Union[int, float, str, bytes, np.ndarray]
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

    def __contains__(self, value: Union[np.dtype, np.ndarray]) -> bool:
        if self.is_empty:
            return False
        
        match type(value):
            case self.dtype:
                # Works for both 1D and 2D buffers
                return nbisin(value, self._array_)

            case np.ndarray:
                # If the buffer is 1D and value is 1D
                if self._array_.ndim == 1 and value.ndim == 1:
                    return np.any(np.all(self._array_ == value))

                # If the buffer is 2D and value is 1D (search for matching rows)
                elif self._array_.ndim == 2 and value.ndim == 1:
                    for i in range(len(self)):
                        if np.array_equal(self._array_[i], value):
                            return True
                    return False
                
            case _:
                raise TypeError("Only np.dtype/np.ndarray allowed.")
                    
    def __eq__(self, ringbuffer: 'RingBufferMultiDim') -> bool:
        assert isinstance(ringbuffer, RingBufferMultiDim)
        return ringbuffer.as_array() == self.as_array()
    
    def __len__(self) -> int:
        return self._right_index_ - self._left_index_

    def __getitem__(self, item: int) -> Union[int, float, str, bytes, np.ndarray]:
        return self.as_array()[item]
    
    def __str__(self) -> str:
        return (f"RingBufferMultiDim(capacity={self.capacity}, "
                f"dtype={self.dtype}, "
                f"current_length={len(self)}, "
                f"data={self.as_array()})")