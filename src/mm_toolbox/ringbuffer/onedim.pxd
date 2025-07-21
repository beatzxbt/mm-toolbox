cimport numpy as cnp

from libc.stdint cimport uint32_t as u32

cdef class RingBufferOneDim:
    cdef:
        u32             _capacity
        u32             _left_index
        u32             _right_index
        u32             _size
        double[:]       _buffer
    
    # def void          __cinit__(self, int capacity)
    cpdef cnp.ndarray   raw(self)
    cpdef cnp.ndarray   unwrapped(self)
    cpdef void          append(self, double value)
    cpdef double        popright(self)
    cpdef double        popleft(self)
    cpdef double        peekright(self)
    cpdef double        peekleft(self)
    cpdef cnp.ndarray   reset(self)
    cpdef void          fast_reset(self)
    cpdef bint          is_full(self)
    cpdef bint          is_empty(self)

    # def               __contains__(self, double value)
    # def               __iter__(self)
    # def               __len__(self)
    # def               __getitem__(self, int idx)
