cimport numpy as cnp
from libc.stdint cimport uint32_t as u32

cdef class RingBufferTwoDim:
    cdef: 
        u32                 _capacity
        u32                 _sub_array_len
        u32                 _left_index
        u32                 _right_index
        u32                 _size
        double[:, :]        _buffer

    # def void              __cinit__(self, int capacity, int sub_array_len)
    cpdef cnp.ndarray       raw(self)
    cpdef cnp.ndarray       unsafe_raw(self)
    cpdef cnp.ndarray       unwrapped(self)
    cpdef void              unsafe_write(self, cnp.ndarray values, int insert_idx=*)
    cpdef void              unsafe_push(self)
    cpdef void              append(self, cnp.ndarray values)
    cpdef cnp.ndarray       popright(self)
    cpdef cnp.ndarray       popleft(self)
    cpdef cnp.ndarray       peekright(self)
    cpdef cnp.ndarray       peekleft(self)
    cpdef cnp.ndarray       reset(self)
    cpdef void              fast_reset(self)
    cpdef bint              is_full(self)
    cpdef bint              is_empty(self)

    # def __contains__(self, double value)
    # def __iter__(self)
    # def __len__(self)
    # def __getitem__(self, int idx)