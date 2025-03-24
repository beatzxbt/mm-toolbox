cimport numpy as cnp

cdef class RingBufferTwoDim:
    cdef: 
        Py_ssize_t          _capacity
        Py_ssize_t          _sub_array_len
        Py_ssize_t          _left_index
        Py_ssize_t          _right_index
        Py_ssize_t          _size
        double[:, :]        _buffer

    # def void              __cinit__(self, Py_ssize_t capacity, Py_ssize_t sub_array_len)
    cpdef cnp.ndarray       raw(self)
    cpdef cnp.ndarray       unsafe_raw(self)
    cpdef cnp.ndarray       unwrapped(self)
    cpdef void              unsafe_write(self, cnp.ndarray values, Py_ssize_t insert_idx=*)
    cpdef void              unsafe_push(self)
    cpdef void              append(self, cnp.ndarray values)
    cpdef cnp.ndarray       popright(self)
    cpdef cnp.ndarray       popleft(self)
    cpdef cnp.ndarray       reset(self)
    cpdef void              fast_reset(self)
    cpdef bint              is_full(self)
    cpdef bint              is_empty(self)

    # def __contains__(self, double value)
    # def __iter__(self)
    # def __len__(self)
    # def __getitem__(self, int idx)