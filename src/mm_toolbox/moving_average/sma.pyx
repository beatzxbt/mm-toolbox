"""Simple moving average (SMA) implementation.

The SMA uses equal weights 1/N, where N is the window size.
"""

cimport numpy as cnp

from mm_toolbox.ringbuffer.numeric cimport NumericRingBuffer
from mm_toolbox.moving_average.base cimport MovingAverage

cdef class SimpleMovingAverage(MovingAverage):
    """Simple moving average with equal weights.

    Maintains a rolling sum for O(1) updates.

    Attributes:
        _raw_values (NumericRingBuffer): Ring buffer of raw input values.
        _rolling_sum (double): Cached sum of the current window.
    """

    def __init__(self, int window, bint is_fast=False):
        """Initialize the SMA.

        Args:
            window (int): Number of samples in the moving average window.
            is_fast (bool): If True, skip storing historical values.
        """
        super().__init__(window, is_fast)
        
        self._raw_values = NumericRingBuffer(window, dtype='float64') 
        self._rolling_sum = 0.0

    cpdef double initialize(self, cnp.ndarray values):
        """Warm the SMA from an array of historical values.

        Args:
            values (cnp.ndarray): Array of length ``window``.

        Returns:
            double: The initial SMA value.

        Raises:
            ValueError: If the input length does not match ``window``.
        """
        cdef:
            int i
            int n = values.shape[0]
            double raw_value
            double[:] values_view = values
    
        if n != self._window:
            raise ValueError(
                f"Input array length must match window; expected {self._window} but got {n}"
            )

        self._values.clear()
        self._rolling_sum = 0.0
        
        for i in range(n):
            raw_value = values_view[i]
            self._rolling_sum += raw_value
            self._raw_values.insert(raw_value)
        
        self._value = self._rolling_sum / self._window
        self._is_warm = True
        self.push_to_ringbuffer()

        return self._value

    cpdef double next(self, double new_val):
        """Calculate the next SMA value without updating state.

        Args:
            new_val (double): New input value to evaluate.

        Returns:
            double: The projected SMA value.
        """
        self.ensure_warm()

        cdef: 
            double new_rolling_sum
            double old_raw_value = self._raw_values.peekleft()
        
        new_rolling_sum = self._rolling_sum + (new_val - old_raw_value)
        return new_rolling_sum / self._window
        
    cpdef double update(self, double new_val):
        """Update the SMA with a new value.

        Args:
            new_val (double): New input value to ingest.

        Returns:
            double: The updated SMA value.
        """
        self.ensure_warm()
        
        cdef double old_raw_value = self._raw_values.consume()
        
        self._rolling_sum += (new_val - old_raw_value)
        self._raw_values.insert(new_val)
        self._value = self._rolling_sum / self._window
        self.push_to_ringbuffer()
        return self._value
