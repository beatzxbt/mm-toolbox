from mm_toolbox.logging.advanced.config cimport LoggerConfig
from mm_toolbox.logging.advanced.log cimport CLogLevel

cdef class MasterLogger:
    cdef:
        LoggerConfig    _config
        list            _log_handlers
        object          _shm_producer
        object          _transport
        object          _timed_operations_thread
        object          _stop_event

    cpdef list          _decode_worker_message(self, bytes internal_message)
    cpdef void          _timed_operations(self)
    cdef void           _log_direct(self, CLogLevel level, bytes msg_bytes)

    cpdef void          trace(self, bytes msg_bytes=*)
    cpdef void          debug(self, bytes msg_bytes=*)
    cpdef void          info(self, bytes msg_bytes=*)
    cpdef void          warning(self, bytes msg_bytes=*)
    cpdef void          error(self, bytes msg_bytes=*)
    cpdef void          shutdown(self)

    cpdef bint          is_running(self)
    cpdef LoggerConfig  get_config(self)
