from libc.stdint cimport uint8_t

from .structs cimport MessageBuffer

cdef class MasterLogger:
    """
    Cython interface for the MasterLogger class, exposing the constructor
    and methods for use in other Cython modules.
    """
    cdef:
        dict            _system_info
        object          _ev_loop
        bytes           _srcfilename
        object          _batch_encoder
        object          _batch_decoder
        object          _master_conn
        list            _log_handlers
        list            _data_handlers
        MessageBuffer   _log_buffer
        MessageBuffer   _data_buffer
        bint            _is_running

    # def __init__(self, LoggerConfig config, list log_handlers,  list data_handlers, str srcfilename="")

    cpdef void _process_worker_msg(self, bytes msg)

    cdef void _logs_dump_to_queue_callback(self, list raw_log_buffer)
    cdef void _data_dump_to_queue_callback(self, list raw_data_buffer)

    cdef inline void _process_log(self, uint8_t level, bytes msg)
    cdef inline void _process_data(self, object data)

    cpdef void data(self, object data)
    cpdef void trace(self, bytes msg)
    cpdef void debug(self, bytes msg)
    cpdef void info(self, bytes msg)
    cpdef void warning(self, bytes msg)
    cpdef void error(self, bytes msg)
    cpdef void critical(self, bytes msg)

    cpdef void close(self)
