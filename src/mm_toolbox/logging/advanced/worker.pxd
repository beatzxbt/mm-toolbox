from libc.stdint cimport uint8_t

from .structs cimport MessageBuffer

cdef class WorkerLogger:
    """
    Cython interface for the WorkerLogger class, exposing
    its constructor and methods to other Cython modules.
    """
    cdef:
        dict            _system_info
        object          _ev_loop
        bytes           _srcfilename
        object          _batch_encoder
        object          _master_conn
        MessageBuffer   _log_buffer
        MessageBuffer   _data_buffer
        bint            _is_running

    # def __init__(self, LoggerConfig config, str srcfilename="")

    cdef void           _logs_dump_to_queue_callback(self, list raw_log_buffer)
    cdef void           _data_dump_to_queue_callback(self, list raw_data_buffer)
    cdef inline void    _process_log(self, uint8_t level, bytes msg)
    cdef inline void    _process_data(self, object data)

    cpdef void          data(self, object data)
    cpdef void          trace(self, bytes msg)
    cpdef void          debug(self, bytes msg)
    cpdef void          info(self, bytes msg)
    cpdef void          warning(self, bytes msg)
    cpdef void          error(self, bytes msg)
    cpdef void          critical(self, bytes msg)
    cpdef void          close(self)
