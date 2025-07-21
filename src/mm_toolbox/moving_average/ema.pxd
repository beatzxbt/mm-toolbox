cimport numpy as cnp

from mm_toolbox.moving_average.base cimport MovingAverage

cdef class ExponentialMovingAverage(MovingAverage):
    cdef:
        double     _alpha

    cpdef double   initialize(self, cnp.ndarray values)
    cpdef double   next(self, double value)
    cpdef double   update(self, double value)