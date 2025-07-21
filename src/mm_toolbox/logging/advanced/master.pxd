from mm_toolbox.logging.advanced.config cimport LoggerConfig
from mm_toolbox.logging.advanced.structs cimport CLogLevel, LogLevel, LogBatch

cdef class MasterLogger:
    """
    Cython interface for the MasterLogger class, exposing the constructor
    and methods for use in other Cython modules.
    """
    cdef:
        LoggerConfig    _config
        bytes           _name
        list            _log_handlers
        LogBatch        _log_batch
        dict            _heartbeats
        object          _conn
        object          _timed_operations_thread
        bint            _is_running

    # def __cinit__(self, LoggerConfig config=None, list log_handlers=None)
    cpdef void          _process_worker_msg(self, bytes msg)
    cpdef void          _timed_operations(self)

    cpdef void          set_log_level(self, LogLevel level)
    cpdef void          trace(self, str msg_str=*, bytes msg_bytes=*)
    cpdef void          debug(self, str msg_str=*, bytes msg_bytes=*)
    cpdef void          info(self, str msg_str=*, bytes msg_bytes=*)
    cpdef void          warning(self, str msg_str=*, bytes msg_bytes=*)
    cpdef void          error(self, str msg_str=*, bytes msg_bytes=*)
    cpdef void          critical(self, str msg_str=*, bytes msg_bytes=*)
    cpdef void          shutdown(self)

    cpdef bint          is_running(self)
    cpdef LoggerConfig  get_config(self)
