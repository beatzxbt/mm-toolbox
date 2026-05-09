# worker.pxd
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
    cdef public:
        u32             _num_pending_logs
        BinaryWriter    _batch_writer
        object          _batch_lock
        object          _transport
        object          _timed_operations_thread
        object          _stop_event

    cpdef void          _timed_operations(self)
    cdef void           _flush_logs(self) except *
    cdef void           _flush_logs_locked(self)
    cdef void           _add_log_to_batch(self, CLogLevel clevel, u32 message_len, unsigned char* message) except *

    cpdef void          trace(self, bytes msg_bytes=*)
    cpdef void          debug(self, bytes msg_bytes=*)
    cpdef void          info(self, bytes msg_bytes=*)
    cpdef void          warning(self, bytes msg_bytes=*)
    cpdef void          error(self, bytes msg_bytes=*)
    cpdef void          shutdown(self)

    cpdef bint          is_running(self)
    cpdef str           get_name(self)
    cpdef object        get_config(self)
