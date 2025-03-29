from .config cimport LoggerConfig
from .structs cimport MessageBuffer, LogLevel

cdef class MasterLogger:
    """
    Cython interface for the MasterLogger class, exposing the constructor
    and methods for use in other Cython modules.
    """
    cdef:
        LoggerConfig    _config
        dict            _system_info
        object          _ev_loop
        bytes           _name
        object          _batch_encoder
        object          _batch_decoder
        object          _master_conn
        list            _log_handlers
        list            _data_handlers
        MessageBuffer   _log_buffer
        MessageBuffer   _data_buffer
        bint            _is_running

    # def __init__(self, LoggerConfig config, list log_handlers, list data_handlers, str name="")
    cdef inline void    _ensure_running(self)
    cpdef void          _process_worker_msg(self, bytes msg)
    cdef void           _logs_dump_to_queue_callback(self, list raw_log_buffer)
    cdef void           _data_dump_to_queue_callback(self, list raw_data_buffer)
    cdef inline void    _process_log(self, LogLevel level, bytes msg)
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
    cpdef dict          get_system_info(self)
