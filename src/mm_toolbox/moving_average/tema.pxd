cimport numpy as cnp

from mm_toolbox.moving_average.base cimport MovingAverage

cdef class TimeExponentialMovingAverage(MovingAverage):
    cdef:
        double      _time
        double      _lam
        
    cpdef double    initialize(self, cnp.ndarray values)
    cpdef double    next(self, double new_val)
    cpdef double    update(self, double new_val)