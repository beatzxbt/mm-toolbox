# worker_logger.pxd
from libc.stdint cimport uint32_t as u32

from mm_toolbox.logging.advanced.protocol cimport BinaryWriter
from mm_toolbox.logging.advanced.log cimport CLogLevel
from mm_toolbox.logging.advanced.config cimport LoggerConfig

cdef class WorkerLogger:
    """
    Cython interface for the WorkerLogger class, exposing
    its constructor and methods to other Cython modules.
    """
    cdef:
        object          _config
        bytes           _name
        u32             _len_name
        unsigned char*  _name_as_chars
        u32             _num_pending_logs
        BinaryWriter    _batch_writer
        object          _transport
        bint            _is_running
        object          _timed_operations_thread
        object          _stop_event

    # cdef                __cinit__(self, object config=None, str name=None)
    cpdef void          _timed_operations(self)
    cdef void           _flush_logs(self)
    cdef void           _add_log_to_batch(self, CLogLevel clevel, u32 message_len, unsigned char* message)
    
    cpdef void          trace(self, str msg_str=*, bytes msg_bytes=*)
    cpdef void          debug(self, str msg_str=*, bytes msg_bytes=*)
    cpdef void          info(self, str msg_str=*, bytes msg_bytes=*)
    cpdef void          warning(self, str msg_str=*, bytes msg_bytes=*)
    cpdef void          error(self, str msg_str=*, bytes msg_bytes=*)
    cpdef void          shutdown(self)

    cpdef bint          is_running(self)
    cpdef str           get_name(self)
    cpdef object        get_config(self)