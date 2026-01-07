import numpy as np
cimport numpy as cnp

from mm_toolbox.ringbuffer.numeric cimport NumericRingBuffer

cdef class MovingAverage:
    def __init__(self, int window, bint is_fast=False):
        if window <= 1:
            raise ValueError(f"Moving average window must be positive; expected >1 but got {window}")

        self._window = window
        self._is_fast = is_fast
        self._is_warm = False
        self._value = 0.0
        self._values = NumericRingBuffer(window, dtype=np.dtype(np.float64))

    cpdef double initialize(self, cnp.ndarray values):
        """Initializes the moving average with the given values. """
        pass

    cpdef double next(self, double value):
        """Calculates the next value without updating the internal state. Useful for plotting."""
        pass

    cpdef double update(self, double value):
        """Calculates the next value and updates the internal state."""
        pass

    cpdef double get_value(self):
        """Returns the current value of the moving average."""
        return self._value

    cpdef cnp.ndarray get_values(self):
        """Returns the historical values in the moving average as a numpy array."""
        self.__enforce_not_fast()
        return self._values.unwrapped()
    
    def __len__(self):
        """Get the number of elements currently in the buffer."""
        self.__enforce_not_fast()
        return len(self._values) 

    def __iter__(self):
        """Iterate over the elements in the buffer in order from oldest to newest."""
        self.__enforce_not_fast()
        return iter(self._values)

    def __getitem__(self, int idx):
        """Get the element at the given index."""
        self.__enforce_not_fast()
        return self._values[idx]
    
    cdef inline void __enforce_moving_average_initialized(self):
        if not self._is_warm:
            raise ValueError("Moving average must be initialized before use; call initialize() first")

    cdef inline void __enforce_not_fast(self):
        if self._is_fast:
            raise ValueError("Cannot store/return historical values in fast mode;")
