"""Weighted moving average (WMA) implementation.

The WMA uses linearly increasing weights from 1 to N, where N is the
window size.
"""

cimport numpy as cnp

from mm_toolbox.ringbuffer.numeric cimport NumericRingBuffer
from mm_toolbox.moving_average.base cimport MovingAverage

cdef class WeightedMovingAverage(MovingAverage):
    """Weighted moving average with linearly increasing weights.

    Maintains rolling sums for O(1) updates.

    Attributes:
        _window_double (double): Window size cast to double for arithmetic.
        _raw_values (NumericRingBuffer): Ring buffer of raw input values.
        _rolling_sum (double): Cached sum of the current window.
        _rolling_wsum (double): Cached weighted sum of the current window.
    """

    def __init__(self, int window, bint is_fast=False):
        """Initialize the WMA.

        Args:
            window (int): Number of samples in the moving average window.
            is_fast (bool): If True, skip storing historical values.
        """
        super().__init__(window, is_fast)
        
        # Cast window to a double as many calculations require
        # so below. Messy, but speeds things up considerably.
        self._window_double = <double>self._window

        self._raw_values = NumericRingBuffer(window, dtype='float64')
        self._rolling_sum = 0.0    
        self._rolling_wsum = 0.0   

    cpdef double initialize(self, cnp.ndarray values):
        """Warm the WMA from an array of historical values.

        Args:
            values (cnp.ndarray): Array of length ``window``.

        Returns:
            double: The initial WMA value.

        Raises:
            ValueError: If the input length does not match ``window``.
        """
        cdef:
            int i, n = values.shape[0]
            double val
            double[:] values_view = values

        if n != self._window:
            raise ValueError(
                f"Input array length must match window; expected {self._window} but got {n}"
            )

        self._values.clear()
        self._raw_values.clear()
        self._rolling_sum = 0.0
        self._rolling_wsum = 0.0

        for i in range(n):
            val = values_view[i]
            self._rolling_sum += val
            self._rolling_wsum += (i + 1) * val
            self._raw_values.insert(val)

        # Denominator = N(N + 1)/2
        self._value = self._rolling_wsum / (self._window_double * (self._window_double + 1.0) / 2.0)
        self._is_warm = True
        self.push_to_ringbuffer()

        return self._value

    cpdef double next(self, double new_val):
        """Calculate the next WMA value without updating state.

        Args:
            new_val (double): New input value to evaluate.

        Returns:
            double: The projected WMA value.
        """
        self.ensure_warm()

        cdef:
            double new_sum
            double new_wsum
            double old_val = self._raw_values.peekleft()

        new_sum = self._rolling_sum - old_val + new_val
        new_wsum = self._rolling_wsum - self._rolling_sum + self._window_double * new_val
        return new_wsum / (self._window_double * (self._window_double + 1.0) / 2.0)

    cpdef double update(self, double new_val):
        """Update the WMA with a new value.

        Args:
            new_val (double): New input value to ingest.

        Returns:
            double: The updated WMA value.
        """
        self.ensure_warm()

        cdef:
            double old_val = self._raw_values.consume()
            double old_sum = self._rolling_sum
            double old_wsum = self._rolling_wsum

        self._rolling_sum = old_sum - old_val + new_val
        self._rolling_wsum = old_wsum - old_sum + self._window_double * new_val
        self._raw_values.insert(new_val)

        # Denominator = N(N + 1)/2
        self._value = self._rolling_wsum / (self._window_double * (self._window_double + 1.0) / 2.0)
        self.push_to_ringbuffer()
        return self._value
