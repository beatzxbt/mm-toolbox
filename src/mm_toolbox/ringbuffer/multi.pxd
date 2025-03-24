cimport numpy as cnp

cdef class RingBufferMulti:
    cdef:
        Py_ssize_t          _ndim     
        Py_ssize_t          _capacity
        Py_ssize_t          _sub_array_len      
        Py_ssize_t          _left_index    
        Py_ssize_t          _right_index   
        Py_ssize_t          _size          
        object              _dtype       
        cnp.ndarray         _buffer   
    
    # def void              __init__(self, object shape, object dtype)
    cpdef cnp.ndarray       raw(self)
    cpdef cnp.ndarray       unsafe_raw(self)
    cpdef cnp.ndarray       unwrapped(self)
    cpdef void              unsafe_write(self, object value, Py_ssize_t insert_idx=*)
    cpdef void              unsafe_push(self)
    cpdef void              append(self, object value)
    cpdef object            popright(self)
    cpdef object            popleft(self)
    cpdef cnp.ndarray       reset(self)
    cpdef void              fast_reset(self)
    cpdef bint              is_full(self)
    cpdef bint              is_empty(self)

    # def bool              __contains__(self, double value)
    # def generator[dtype]  __iter__(self)
    # def int               __len__(self)
    # def dtype             __getitem__(self, int idx)