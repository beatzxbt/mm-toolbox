from libc.stdint cimport uint32_t as u32

from mm_toolbox.logging.advanced.config cimport LoggerConfig

cdef class MasterLogger:
    cdef:
        LoggerConfig    _config
        bytes           _name
        list            _log_handlers
        u32             _num_pending_logs
        list            _pending_logs
        object          _transport
        object          _timed_operations_thread
        bint            _is_running

    # def __cinit__(self, LoggerConfig config=None, list log_handlers=None)
    cdef list           _decode_worker_message(self, bytes internal_message)
    cpdef void          _timed_operations(self)
    cdef inline object  _make_pylog(self, object level, bytes message)
    cdef void           _add_pylog_to_batch(self, object pylog)

    cpdef void          trace(self, str msg_str=*, bytes msg_bytes=*)
    cpdef void          debug(self, str msg_str=*, bytes msg_bytes=*)
    cpdef void          info(self, str msg_str=*, bytes msg_bytes=*)
    cpdef void          warning(self, str msg_str=*, bytes msg_bytes=*)
    cpdef void          error(self, str msg_str=*, bytes msg_bytes=*)
    cpdef void          shutdown(self)

    cpdef bint          is_running(self)
    cpdef LoggerConfig  get_config(self)
