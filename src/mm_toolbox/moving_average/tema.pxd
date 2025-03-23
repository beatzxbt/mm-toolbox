cimport numpy as np

from .base cimport MovingAverage

cdef class TimeExponentialMovingAverage(MovingAverage):
    cdef:
        double      _time
        double      _lam
        
    cpdef double    initialize(self, np.ndarray values)
    cpdef double    next(self, double new_val)
    cpdef double    update(self, double new_val)