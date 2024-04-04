import numpy as np
from numba.types import int64, float64, Array
from numba.experimental import jitclass


def get_ring_buffer_spec(data_type):
    return [
        ("_arr_", Array(data_type, 1, "C")),  # Dynamic array based on data_type
        ("_left_index_", int64),
        ("_right_index_", int64),
        ("capacity", int64),
    ]


def RingBuffer(capacity: int, data_type=float64):
    spec = get_ring_buffer_spec(data_type)

    @jitclass(spec)
    class RingBufferFactory:
        """
        A fixed-size circular buffer using Numba JIT compilation.

        Parameters
        ----------
        capacity : int
            The maximum number of elements the buffer can hold.

        data_type : dtype
            The data type of the elements that the buffer will store. This is determined
            dynamically when the ring buffer is created through the factory function.
        """

        def __init__(self):
            self._arr_ = np.empty(shape=capacity, dtype=data_type)
            self._left_index_ = 0
            self._right_index_ = 0
            self.capacity = capacity

        def __len__(self) -> int:
            return self._right_index_ - self._left_index_

        def __getitem__(self, item: int) -> float:
            return self._unwrap_()[item]

        @property
        def dtype(self) -> Array:
            """Data type of the array's contents"""
            return self._arr_.dtype

        @property
        def is_full(self) -> bool:
            """True if there is no more space in the buffer"""
            return len(self) == self.capacity

        def _unwrap_(self) -> Array:
            """Returns a linearized form of the buffer's contents."""
            return np.concatenate(
                (
                    self._arr_[
                        self._left_index_ : min(self._right_index_, self.capacity)
                    ],
                    self._arr_[: max(self._right_index_ - self.capacity, 0)],
                )
            )

        def _fix_indices_(self) -> None:
            """Corrects the indices if they exceed the buffer's capacity."""
            if self._left_index_ >= self.capacity:
                self._left_index_ -= self.capacity
                self._right_index_ -= self.capacity
            elif self._left_index_ < 0:
                self._left_index_ += self.capacity
                self._right_index_ += self.capacity

        def appendright(self, value) -> None:
            """Adds an element to the end of the buffer."""
            if self.is_full:
                self._left_index_ += 1

            self._arr_[self._right_index_ % self.capacity] = value
            self._right_index_ += 1
            self._fix_indices_()

        def appendleft(self, value) -> None:
            """Adds an element to the start of the buffer."""
            if self.is_full:
                self._right_index_ -= 1

            self._left_index_ -= 1
            self._fix_indices_()
            self._arr_[self._left_index_] = value

        def popright(self) -> Array:
            """Removes and returns an element from the end of the buffer."""
            if len(self) == 0:
                raise IndexError("Cannot pop from an empty RingBuffer")

            self._right_index_ -= 1
            self._fix_indices_()
            res = self._arr_[self._right_index_ % self.capacity]
            return res

        def popleft(self) -> Array:
            """Removes and returns an element from the start of the buffer."""
            if len(self) == 0:
                raise IndexError("Cannot pop from an empty RingBuffer")

            res = self._arr_[self._left_index_]
            self._left_index_ += 1
            self._fix_indices_()
            return res

        def reset(self) -> Array:
            """Clears the buffer and resets it to its initial state."""
            res = self._unwrap_()
            self._arr_ = np.empty(shape=capacity, dtype=data_type)
            self._left_index_ = 0
            self._right_index_ = 0
            return res

    return RingBufferFactory()
