cimport numpy as np

from mm_toolbox.ringbuffer.onedim cimport RingBufferOneDim
from .base cimport MovingAverage

cdef class WeightedMovingAverage(MovingAverage):
    cdef:
        double              _window_double
        RingBufferOneDim    _raw_values
        double              _rolling_sum
        double              _rolling_wsum

    cpdef double            initialize(self, np.ndarray values)
    cpdef double            next(self, double new_val)
    cpdef double            update(self, double new_val)