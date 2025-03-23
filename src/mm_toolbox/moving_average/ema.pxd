cimport numpy as np

from .base cimport MovingAverage

cdef class ExponentialMovingAverage(MovingAverage):
    cdef:
        double     _alpha

    cpdef double   initialize(self, np.ndarray values)
    cpdef double   next(self, double value)
    cpdef double   update(self, double value)