from mm_toolbox.logging.advanced.structs cimport LogLevel, LogBatch
from mm_toolbox.logging.advanced.config cimport LoggerConfig

cdef class WorkerLogger:
    """
    Cython interface for the WorkerLogger class, exposing
    its constructor and methods to other Cython modules.
    """
    cdef:
        LoggerConfig    _config
        bytes           _name
        object          _conn
        LogBatch        _log_batch
        bint            _is_running
        object          _timed_operations_thread

    # def               __cinit__(self, LoggerConfig config=None, str name="")
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
    cpdef str           get_name(self)
    cpdef LoggerConfig  get_config(self)
