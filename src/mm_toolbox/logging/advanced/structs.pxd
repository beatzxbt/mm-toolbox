
cdef enum LogLevel:
    TRACE = 1
    DEBUG = 2
    INFO = 3
    WARNING = 4
    ERROR = 5
    CRITICAL = 6

# Useful later to map LogLevel's back to their readable form.
cdef object LogLevelMap = {
    1: "TRACE",
    2: "DEBUG",
    3: "INFO",
    4: "WARNING",
    5: "ERROR",
    6: "CRITICAL"
}

cdef class MessageBuffer:
    cdef:
        Py_ssize_t      _capacity
        Py_ssize_t      _size
        double          _timeout_s
        double          _start_time
        list            _buffer
        object          _dump_to_queue_callback

    # def __init__(self, callable dump_to_queue_callback, Py_ssize_t capacity=UINT16_MAX, double timeout_s=2.5)
    cdef inline bint    _is_full(self)
    cdef void           append(self, object msg) # msg: msgspec.Struct
    cdef list[object]   acquire_all(self) # return: list[msgspec.Struct]