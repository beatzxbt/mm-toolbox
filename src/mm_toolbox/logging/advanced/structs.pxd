from libc.stdint cimport uint8_t

cpdef str log_level_to_str(uint8_t level)

cdef class MessageBuffer:
    cdef:
        Py_ssize_t      _capacity
        Py_ssize_t      _size
        double          _timeout_s
        double          _start_time
        list            _buffer
        object          _dump_to_queue_callback

    # def __init__(self, callable dump_to_queue_callback, Py_ssize_t capacity=1000, double timeout_s=1.0)
    cdef inline bint    _is_full(self)
    cdef void           append(self, object msg)    # msg: msgspec.Struct
    cdef list           acquire_all(self)           # return: list[msgspec.Struct]