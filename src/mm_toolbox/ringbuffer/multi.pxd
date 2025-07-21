cimport numpy as cnp
from libc.stdint cimport uint32_t as u32

cdef class RingBufferMulti:
    cdef:
        u32                 _ndim     
        u32                 _capacity
        u32                 _sub_array_len      
        u32                 _left_index    
        u32                 _right_index   
        u32                 _size          
        object              _dtype       
        cnp.ndarray         _buffer   
    
    # def void              __init__(self, object shape, object dtype)
    cpdef cnp.ndarray       raw(self)
    cpdef cnp.ndarray       unsafe_raw(self)
    cpdef cnp.ndarray       unwrapped(self)
    cpdef void              unsafe_write(self, object value, int insert_idx=*)
    cpdef void              unsafe_push(self)
    cpdef void              append(self, object value)
    cpdef object            popright(self)
    cpdef object            popleft(self)
    cpdef cnp.ndarray       peekright(self)
    cpdef cnp.ndarray       peekleft(self)
    cpdef cnp.ndarray       reset(self)
    cpdef void              fast_reset(self)
    cpdef bint              is_full(self)
    cpdef bint              is_empty(self)

    # def bool              __contains__(self, double value)
    # def generator[dtype]  __iter__(self)
    # def int               __len__(self)
    # def dtype             __getitem__(self, int idx)