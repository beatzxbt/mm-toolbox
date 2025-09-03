cimport numpy as cnp

from mm_toolbox.ringbuffer.numeric cimport NumericRingBuffer

cdef class MovingAverage:
    cdef:
        int                 _window
        bint                _is_fast
        bint                _is_warm
        double              _value
        NumericRingBuffer   _values

    # def                   __init__(self, int window, bint is_fast=False)

    cpdef double            initialize(self, cnp.ndarray values)
    cpdef double            next(self, double value) 
    cpdef double            update(self, double value)
    cpdef double            get_value(self)
    cpdef cnp.ndarray       get_values(self)

    # def __len__(self)
    # def __iter__(self)
    # def __getitem__(self, int idx)
    cdef inline void        __enforce_moving_average_initialized(self)
    cdef inline void        __enforce_not_fast(self)
