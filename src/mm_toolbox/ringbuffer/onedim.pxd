import numpy as np

cimport numpy as np
from libc.stdint cimport uint32_t

cdef class RingBufferOneDim:
    cdef:
        uint32_t            _capacity
        uint32_t            _left_index
        uint32_t            _right_index
        uint32_t            _size
        double[:]           _buffer
        
    cpdef np.ndarray        raw(self)
    cpdef np.ndarray        unsafe_raw(self)
    cpdef np.ndarray        unwrapped(self)
    cpdef void              unsafe_write(self, double value)
    cpdef void              unsafe_push(self)
    cpdef void              append(self, double value)
    cpdef double            popright(self)
    cpdef double            popleft(self)
    cpdef np.ndarray        reset(self)
    cpdef void              fast_reset(self)
    cpdef bint              is_full(self)
    cpdef bint              is_empty(self)

    # def __contains__(self, double value)
    # def __iter__(self)
    # def __len__(self)
    # def __getitem__(self, int idx)
