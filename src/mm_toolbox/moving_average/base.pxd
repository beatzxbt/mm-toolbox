cimport numpy as cnp
from mm_toolbox.ringbuffer.onedim cimport RingBufferOneDim

cdef class MovingAverage:
    cdef:
        Py_ssize_t          _window
        bint                _is_fast
        bint                _is_warm
        double              _value
        RingBufferOneDim    _values

    # def void              __init__(self, Py_ssize_t window, bint is_fast)
    cdef inline void        ensure_warm(self)
    cdef inline void        ensure_not_fast(self)
    cdef inline void        push_to_ringbuffer(self)

    cpdef double            initialize(self, cnp.ndarray values)
    cpdef double            next(self, double value) 
    cpdef double            update(self, double value)
    cpdef double            get_value(self)
    cpdef cnp.ndarray       get_values(self)

    # def Py_ssize_t        __len__(self)
    # def bool              __contains__(self, double value)
    # def generator         __iter__(self)
    # def double            __getitem__(self, Py_ssize_t idx)
