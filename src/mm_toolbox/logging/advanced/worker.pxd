from libc.stdint cimport uint8_t

from .structs cimport MessageBuffer
from .config cimport LoggerConfig

cdef class WorkerLogger:
    """
    Cython interface for the WorkerLogger class, exposing
    its constructor and methods to other Cython modules.
    """
    cdef:
        LoggerConfig    _config
        dict            _system_info
        object          _ev_loop
        bytes           _name
        object          _batch_encoder
        object          _master_conn
        MessageBuffer   _log_buffer
        MessageBuffer   _data_buffer
        bint            _is_running

    # def void          __init__(self, LoggerConfig config, str name="")
    cdef void           _ensure_running(self)
    cdef void           _logs_dump_to_queue_callback(self, list raw_log_buffer)
    cdef void           _data_dump_to_queue_callback(self, list raw_data_buffer)
    cdef inline void    _process_log(self, uint8_t level, bytes msg)
    cdef inline void    _process_data(self, object msg)
    
    cpdef void          set_format(self, str format_string)
    cpdef void          set_log_level(self, int level)
    cpdef void          data(self, object data, bint unsafe=*)
    cpdef void          trace(self, str msg)
    cpdef void          debug(self, str msg)
    cpdef void          info(self, str msg)
    cpdef void          warning(self, str msg)
    cpdef void          error(self, str msg)
    cpdef void          critical(self, str msg)
    cpdef void          shutdown(self)

    cpdef bint          is_running(self)
    cpdef str           get_name(self)
    cpdef LoggerConfig  get_config(self)
