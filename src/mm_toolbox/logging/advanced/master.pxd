from mm_toolbox.logging.advanced.config cimport LoggerConfig
from mm_toolbox.logging.advanced.worker cimport WorkerLogger

cdef class MasterLogger:
    cdef:
        LoggerConfig    _config
        list            _log_handlers
        WorkerLogger    _worker
        object          _transport
        object          _timed_operations_thread
        bint            _is_running

    # def __cinit__(self, LoggerConfig config=None, list log_handlers=None)
    cdef list           _decode_worker_message(self, bytes internal_message)
    cpdef void          _timed_operations(self)

    cpdef void          trace(self, str msg_str=*, bytes msg_bytes=*)
    cpdef void          debug(self, str msg_str=*, bytes msg_bytes=*)
    cpdef void          info(self, str msg_str=*, bytes msg_bytes=*)
    cpdef void          warning(self, str msg_str=*, bytes msg_bytes=*)
    cpdef void          error(self, str msg_str=*, bytes msg_bytes=*)
    cpdef void          shutdown(self)

    cpdef bint          is_running(self)
    cpdef LoggerConfig  get_config(self)
