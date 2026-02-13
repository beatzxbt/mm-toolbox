"""
Moving average base class and shared utilities.

Provides:
- MovingAverage base API for derived moving averages
- shared ringbuffer storage for historical values
- warm-state and fast-mode helpers
"""

from __future__ import annotations

import numpy as np
cimport numpy as cnp

from mm_toolbox.ringbuffer.numeric cimport NumericRingBuffer

cdef class MovingAverage:
    """Base class for moving average implementations.

    Args:
        window (int): Number of samples in the moving average window.
        is_fast (bool): If True, skip storing historical values.
    """

    def __init__(self, int window, bint is_fast=False):
        """Initialize the moving average base class.

        Args:
            window (int): Number of samples in the moving average window.
            is_fast (bool): If True, skip storing historical values.

        Returns:
            None: This initializer does not return a value.

        Raises:
            ValueError: If the window size is <= 1.
        """
        if window <= 1:
            raise ValueError(f"Moving average window must be positive; expected >1 but got {window}")

        self._window = window
        self._is_fast = is_fast
        self._is_warm = False
        self._value = 0.0
        self._values = NumericRingBuffer(
            window,
            dtype=np.dtype(np.float64),
            disable_async=is_fast,
        )

    cpdef double initialize(self, cnp.ndarray values):
        """Initialize the moving average with the given values.

        Args:
            values (cnp.ndarray): Initial values for the moving average window.

        Returns:
            double: The initialized moving average value.
        """
        pass

    cpdef double next(self, double value):
        """Calculate the next value without updating internal state.

        Args:
            value (double): New input value to evaluate.

        Returns:
            double: The next moving average value.
        """
        pass

    cpdef double update(self, double value):
        """Update the moving average with the next value.

        Args:
            value (double): New input value to ingest.

        Returns:
            double: The updated moving average value.
        """
        pass

    cpdef double get_value(self):
        """Return the current moving average value.

        Returns:
            double: The current moving average value.
        """
        return self._value

    cpdef cnp.ndarray get_values(self):
        """Return historical values as a NumPy array.

        Returns:
            cnp.ndarray: Historical moving average values.
        """
        self.__enforce_not_fast()
        return self._values.unwrapped()
    
    def __len__(self):
        """Return the number of stored values.

        Returns:
            int: Count of stored values.
        """
        self.__enforce_not_fast()
        return len(self._values) 

    def __iter__(self):
        """Iterate over stored values from oldest to newest.

        Returns:
            Iterator[float]: Iterator over stored values.
        """
        self.__enforce_not_fast()
        return iter(self._values)

    def __getitem__(self, int idx):
        """Get the stored value at the given index.

        Args:
            idx (int): Index of the element to retrieve.

        Returns:
            float: Stored moving average value.
        """
        self.__enforce_not_fast()
        return self._values[idx]

    cdef inline void ensure_warm(self):
        """Ensure the moving average has been initialized.

        Args:
            None: This helper does not accept arguments.

        Returns:
            None: This helper does not return a value.

        Raises:
            ValueError: If the moving average is not initialized.
        """
        self.__enforce_moving_average_initialized()

    cdef inline void push_to_ringbuffer(self):
        """Store the current value in the ringbuffer unless in fast mode.

        Args:
            None: This helper does not accept arguments.

        Returns:
            None: This helper does not return a value.
        """
        if self._is_fast:
            return
        self._values.insert(self._value)
    
    cdef inline void __enforce_moving_average_initialized(self):
        if not self._is_warm:
            raise ValueError("Moving average must be initialized before use; call initialize() first")

    cdef inline void __enforce_not_fast(self):
        if self._is_fast:
            raise ValueError("Cannot store/return historical values in fast mode;")
