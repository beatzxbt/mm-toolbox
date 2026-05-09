import os
import threading

from libc.stdint cimport (
    uint8_t as u8,
    uint32_t as u32,
    uint64_t as u64,
)

from mm_toolbox.time.time cimport time_ns
from mm_toolbox.ringbuffer.shm.mpsc import ShmMpscProducer

from mm_toolbox.logging.advanced.log cimport CLogLevel
from mm_toolbox.logging.advanced.protocol cimport (
    BinaryWriter,
    MessageType
)

from mm_toolbox.logging.advanced.config cimport LoggerConfig
from cpython.bytes cimport PyBytes_FromStringAndSize

_worker_logger_local = threading.local()


cdef class WorkerLogger:
    """A lightweight worker logger that sends log messages to the master logger."""

    def __cinit__(
        self,
        LoggerConfig config=None,
        str name=None,
    ):
        if hasattr(_worker_logger_local, 'logger'):
            raise RuntimeError(
                f"Only one WorkerLogger allowed per thread. "
                f"Existing: {_worker_logger_local.logger.get_name()}"
            )
        _worker_logger_local.logger = self

        self._config = config if config else LoggerConfig()

        self._name = (name if name else f"WORKER{os.getpid()}").encode('utf-8')
        self._len_name = len(self._name)
        self._name_as_chars = <unsigned char*>self._name

        self._num_pending_logs = 0
        self._batch_writer = BinaryWriter(initial_capacity=1*1024*1024)  # 1MB fixed
        self._batch_lock = threading.Lock()

        self._transport = ShmMpscProducer(
            path=self._config.path,
            capacity_bytes=self._config.shm_capacity_bytes,
            num_rings=0,
            create=False,  # Workers attach, don't create
        )

        self._timed_operations_thread = threading.Thread(
            target=self._timed_operations,
            daemon=True
        )
        self._stop_event = threading.Event()
        self._timed_operations_thread.start()

        if self._config.emit_internal:
            self.debug(b"WorkerLogger started")

    cpdef void _timed_operations(self):
        """Background processing loop."""
        while not self._stop_event.is_set():
            if self._stop_event.wait(self._config.flush_interval_s):
                break
            try:
                if self._num_pending_logs > 0:
                    self._flush_logs()
            except Exception:
                # Reset to prevent infinite retry / duplicate logs
                with self._batch_lock:
                    self._batch_writer.reset()
                    self._num_pending_logs = 0

    cdef void _flush_logs(self) except *:
        """Flush pending logs."""
        with self._batch_lock:
            if self._num_pending_logs == 0:
                return
            self._flush_logs_locked()

    cdef void _flush_logs_locked(self):
        """Flush pending logs; must hold _batch_lock."""
        cdef u32 batch_len = self._batch_writer.length()
        cdef u32 data_len = 4 + self._len_name + 4 + batch_len
        cdef BinaryWriter writer = BinaryWriter(1 + 8 + 4 + data_len)
        writer.write_u8(<u8>MessageType.LOG)
        writer.write_u64(time_ns())
        writer.write_u32(data_len)
        writer.write_u32(self._len_name)
        writer.write_chars(self._name_as_chars, self._len_name)
        writer.write_u32(self._num_pending_logs)
        writer.write_chars(self._batch_writer._buffer, batch_len)
        cdef bytes payload = PyBytes_FromStringAndSize(<char*>writer._buffer, writer.length())
        self._transport.insert(payload)

        self._batch_writer.reset()
        self._num_pending_logs = 0

    cdef void _add_log_to_batch(self, CLogLevel clevel, u32 message_len, unsigned char* message) except *:
        """Add a log to the batch."""
        cdef u64 time_now_ns
        with self._batch_lock:
            time_now_ns = time_ns()
            self._batch_writer.write_u64(time_now_ns)
            self._batch_writer.write_u8(<u8>clevel)
            self._batch_writer.write_u32(message_len)
            self._batch_writer.write_chars(message, message_len)
            self._num_pending_logs += 1

            # Size-based flush trigger
            if (self._num_pending_logs >= self._config.max_batch_messages or
                self._batch_writer.length() >= self._config.max_batch_bytes):
                self._flush_logs_locked()

    cpdef void trace(self, bytes msg_bytes=b""):
        """Send a trace-level log message."""
        if not self._stop_event.is_set() and CLogLevel.TRACE >= self._config.base_level:
            self._add_log_to_batch(CLogLevel.TRACE, len(msg_bytes), <unsigned char*>msg_bytes)

    cpdef void debug(self, bytes msg_bytes=b""):
        """Send a debug-level log message."""
        if not self._stop_event.is_set() and CLogLevel.DEBUG >= self._config.base_level:
            self._add_log_to_batch(CLogLevel.DEBUG, len(msg_bytes), <unsigned char*>msg_bytes)

    cpdef void info(self, bytes msg_bytes=b""):
        """Send an info-level log message."""
        if not self._stop_event.is_set() and CLogLevel.INFO >= self._config.base_level:
            self._add_log_to_batch(CLogLevel.INFO, len(msg_bytes), <unsigned char*>msg_bytes)

    cpdef void warning(self, bytes msg_bytes=b""):
        """Send a warning-level log message."""
        if not self._stop_event.is_set() and CLogLevel.WARNING >= self._config.base_level:
            self._add_log_to_batch(CLogLevel.WARNING, len(msg_bytes), <unsigned char*>msg_bytes)

    cpdef void error(self, bytes msg_bytes=b""):
        """Send an error-level log message."""
        if not self._stop_event.is_set() and CLogLevel.ERROR >= self._config.base_level:
            self._add_log_to_batch(CLogLevel.ERROR, len(msg_bytes), <unsigned char*>msg_bytes)

    cpdef void shutdown(self):
        """Shutdown with proper cleanup."""
        if self._stop_event.is_set():
            return

        if self._config.emit_internal:
            self.debug(b"Shutting down worker logger")
        self._stop_event.set()

        # Final flush, having kept _stop_event unset until this
        # point ensures that no more logs will be added into the batch.
        self._flush_logs()

        # Wait for thread and cleanup
        self._timed_operations_thread.join()
        self._transport.close()

        if hasattr(_worker_logger_local, 'logger'):
            delattr(_worker_logger_local, 'logger')

    cpdef bint is_running(self):
        """Check if the logger is running."""
        return not self._stop_event.is_set()

    cpdef str get_name(self):
        """Get the name of the logger."""
        return self._name.decode()

    cpdef object get_config(self):
        """Get the configuration of the logger."""
        return self._config
