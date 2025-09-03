cimport numpy as cnp

from mm_toolbox.ringbuffer.numeric cimport NumericRingBuffer
from mm_toolbox.moving_average.base cimport MovingAverage

cdef class WeightedMovingAverage(MovingAverage):
    cdef:
        double              _window_double
        NumericRingBuffer   _raw_values
        double              _rolling_sum
        double              _rolling_wsum

    cpdef double            initialize(self, cnp.ndarray values)
    cpdef double            next(self, double new_val)
    cpdef double            update(self, double new_val)