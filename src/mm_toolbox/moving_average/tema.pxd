cimport numpy as cnp
from libc.stdint cimport int64_t as i64

from mm_toolbox.moving_average.base cimport MovingAverage

cdef class TimeExponentialMovingAverage(MovingAverage):
    cdef:
        i64         _time_ns
        double      _lam

    cdef inline double _alpha_at(self, i64 time_now_ns) noexcept
    cdef double _next_at(self, double new_val, i64 time_now_ns) noexcept
    cdef double _update_at(self, double new_val, i64 time_now_ns)
    cpdef double    initialize(self, cnp.ndarray values)
    cpdef double    next(self, double new_val)
    cpdef double    update(self, double new_val)
