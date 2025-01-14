from libc.stdint cimport uint16_t

cdef enum LogLevel:
    TRACE = 1
    DEBUG = 2
    INFO = 3
    WARNING = 4
    ERROR = 5
    CRITICAL = 6

cdef class MessageBuffer:
    cdef:
        uint16_t        _capacity
        uint16_t        _size
        double          _timeout_s
        double          _start_time
        list            _buffer
        object          _dump_to_queue_callback

    # def __init__(self, callable dump_to_queue_callback, uint16_t capacity=UINT16_MAX, double timeout_s=2.5)
    cdef inline bint    _is_full(self)
    cdef void           append(self, object msg) # msg: msgspec.Struct
    cdef list           acquire_all(self) # return: list[msgspec.Struct]