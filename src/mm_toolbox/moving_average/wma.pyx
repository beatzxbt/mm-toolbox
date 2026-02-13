cimport numpy as cnp

from mm_toolbox.ringbuffer.numeric cimport NumericRingBuffer
from mm_toolbox.moving_average.base cimport MovingAverage

cdef class WeightedMovingAverage(MovingAverage):
    """
    The WMA uses linearly increasing weights from 1 to N, where N is the window size.
    """

    def __init__(self, int window, bint fast=False):
        super().__init__(window, fast)
        
        # Cast window to a double as many calculations require
        # so below. Messy, but speeds things up considerably.
        self._window_double = <double>self._window

        self._raw_values = NumericRingBuffer(window, dtype='float64')
        self._rolling_sum = 0.0    
        self._rolling_wsum = 0.0   

    cpdef double initialize(self, cnp.ndarray values):
        cdef:
            int i, n = values.shape[0]
            double val
            double[:] values_view = values

        if n != self._window:
            raise ValueError(
                f"Input array length must match window; expected {self._window} but got {n}"
            )

        self._values.fast_reset()
        self._raw_values.fast_reset()
        self._rolling_sum = 0.0
        self._rolling_wsum = 0.0

        for i in range(n):
            val = values_view[i]
            self._rolling_sum += val
            self._rolling_wsum += (i + 1) * val
            self._raw_values.append(val)

        # Denominator = N(N + 1)/2
        self._value = self._rolling_wsum / (self._window_double * (self._window_double + 1.0) / 2.0)
        self._is_warm = True
        self.push_to_ringbuffer()

        return self._value

    cpdef double next(self, double new_val):
        self.ensure_warm()

        cdef:
            double new_sum
            double new_wsum
            double old_val = self._raw_values[0]

        new_sum = self._rolling_sum - old_val + new_val
        new_wsum = self._rolling_wsum - self._rolling_sum + self._window_double * new_val
        return new_wsum / (self._window_double * (self._window_double + 1.0) / 2.0)

    cpdef double update(self, double new_val):
        self.ensure_warm()

        cdef:
            double old_val = self._raw_values.popleft()
            double old_sum = self._rolling_sum
            double old_wsum = self._rolling_wsum

        self._rolling_sum = old_sum - old_val + new_val
        self._rolling_wsum = old_wsum - old_sum + self._window_double * new_val
        self._raw_values.append(new_val)

        # Denominator = N(N + 1)/2
        self._value = self._rolling_wsum / (self._window_double * (self._window_double + 1.0) / 2.0)
        self.push_to_ringbuffer()
        return self._value
