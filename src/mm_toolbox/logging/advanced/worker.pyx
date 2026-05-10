"""WorkerLogger implementation.

Lightweight per-thread logger that batches messages and forwards them
via shared memory to the MasterLogger.  Enforces a singleton per thread.
"""

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


# Per-thread singleton guard
_worker_logger_local = threading.local()


cdef class WorkerLogger:
    """Lightweight per-thread logger that batches messages for the master.

    Enforces a singleton per thread and flushes batched logs to a shared-memory
    MPSC ring either periodically or when size limits are reached.

    Attributes:
        _config (LoggerConfig): Active logger configuration.
        _name (bytes): UTF-8 encoded logger name.
        _len_name (int): Byte length of the logger name.
        _name_as_chars (unsigned char*): Pointer to the name bytes.
        _num_pending_logs (int): Number of logs in the current batch.
        _batch_writer (BinaryWriter): Accumulates batched log records.
        _batch_lock (threading.Lock): Protects batch operations.
        _transport (ShmMpscProducer): Shared-memory transport to the master.
        _stop_event (threading.Event): Signals shutdown.
        _timed_operations_thread (threading.Thread): Background flush thread.
    """

    def __cinit__(
        self, 
        LoggerConfig config=None, 
        str name=None,
    ):
        """Initialize the worker logger.

        Args:
            config (LoggerConfig): Logger settings; defaults to a new LoggerConfig().
            name (str): Worker name; defaults to ``WORKER<pid>``.

        Raises:
            RuntimeError: If another WorkerLogger is already active in this thread.
        """
        cdef bint should_create = False

        self._config = config if config else LoggerConfig() 

        self._name = (name if name else f"WORKER{os.getpid()}").encode('utf-8')
        self._len_name = len(self._name)
        self._name_as_chars = <unsigned char*>self._name
        
        self._num_pending_logs = 0
        self._batch_writer = BinaryWriter(initial_capacity=1*1024*1024)  # 1MB baseline
        self._batch_lock = threading.Lock()

        # Singleton guard: only one WorkerLogger per thread
        if hasattr(_worker_logger_local, 'logger'):
            existing = _worker_logger_local.logger
            if existing.is_running():
                raise RuntimeError(
                    f"Only one WorkerLogger allowed per thread. "
                    f"Existing: {existing.get_name()}"
                )
            # If existing is not running, remove it and allow new creation
            delattr(_worker_logger_local, 'logger')

        # Try to attach to existing SHM ring first; create if not exists
        try:
            self._transport = ShmMpscProducer(
                path=self._config.path,
                capacity_bytes=self._config.shm_capacity_bytes,
                num_rings=0,
                create=False,
            )
        except (OSError, RuntimeError):
            self._transport = ShmMpscProducer(
                path=self._config.path,
                capacity_bytes=self._config.shm_capacity_bytes,
                num_rings=0,
                create=True,
                unlink_on_close=False,
            )
        
        self._stop_event = threading.Event()
        
        self._timed_operations_thread = threading.Thread(
            target=self._timed_operations,
            daemon=True
        )
        self._timed_operations_thread.start()

        # Register this logger for the current thread
        _worker_logger_local.logger = self

        if self._config.emit_internal:
            self.debug(msg_bytes=f"WorkerLogger started; name: {self._name.decode()}".encode('utf-8'))

    cpdef void _timed_operations(self):
        """Background loop that flushes pending logs on an interval.

        Sleeps for ``flush_interval_s`` and then triggers a flush if logs
        are pending.
        """
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
        """Flush the current batch to shared memory (thread-safe).

        Acquires ``_batch_lock`` before serializing and sending.
        """
        cdef u32 batch_len
        cdef u32 data_len
        cdef BinaryWriter writer
        with self._batch_lock:
            if self._num_pending_logs == 0:
                return

            batch_len = self._batch_writer.length()
            # Check for overflow in data_len calculation
            if batch_len > <u32>(0xFFFFFFFF) - (4 + self._len_name + 4):
                raise ValueError("Batch data length would overflow")
            data_len = 4 + self._len_name + 4 + batch_len
            # Check for overflow in writer allocation size
            if data_len > <u32>(0xFFFFFFFF) - (1 + 8 + 4):
                raise ValueError("Writer allocation would overflow")
            writer = BinaryWriter(1 + 8 + 4 + data_len)
            writer.write_u8(<u8>MessageType.LOG)
            writer.write_u64(time_ns())
            writer.write_u32(data_len)
            writer.write_u32(self._len_name)
            writer.write_chars(self._name_as_chars, self._len_name)
            writer.write_u32(self._num_pending_logs)
            writer.write_chars(self._batch_writer._buffer, batch_len)
            self._transport.insert(writer.finalize())

            self._batch_writer.reset()
            self._num_pending_logs = 0
    
    cdef void _add_log_to_batch(self, CLogLevel clevel, u32 message_len, unsigned char* message) except *:
        """Append a single log record to the batch.

        Args:
            clevel (CLogLevel): Severity level.
            message_len (u32): Byte length of the message.
            message (unsigned char*): Pointer to the message bytes.
        """
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

    cdef void _flush_logs_locked(self) except *:
        """Flush the current batch (caller must hold ``_batch_lock``)."""
        cdef u32 batch_len
        cdef u32 data_len
        cdef BinaryWriter writer
        if self._num_pending_logs == 0:
            return

        batch_len = self._batch_writer.length()
        # Check for overflow in data_len calculation
        if batch_len > <u32>(0xFFFFFFFF) - (4 + self._len_name + 4):
            raise ValueError("Batch data length would overflow")
        data_len = 4 + self._len_name + 4 + batch_len
        # Check for overflow in writer allocation size
        if data_len > <u32>(0xFFFFFFFF) - (1 + 8 + 4):
            raise ValueError("Writer allocation would overflow")
        writer = BinaryWriter(1 + 8 + 4 + data_len)
        writer.write_u8(<u8>MessageType.LOG)
        writer.write_u64(time_ns())
        writer.write_u32(data_len)
        writer.write_u32(self._len_name)
        writer.write_chars(self._name_as_chars, self._len_name)
        writer.write_u32(self._num_pending_logs)
        writer.write_chars(self._batch_writer._buffer, batch_len)
        self._transport.insert(writer.finalize())

        self._batch_writer.reset()
        self._num_pending_logs = 0

    cpdef void trace(self, bytes msg_bytes=b""):
        """Send a trace-level log message to the master.

        Args:
            msg_bytes (bytes): UTF-8 encoded message payload.
        """
        if not self._stop_event.is_set() and CLogLevel.TRACE >= self._config.base_level:
            self._add_log_to_batch(CLogLevel.TRACE, len(msg_bytes), <unsigned char*>msg_bytes)

    cpdef void debug(self, bytes msg_bytes=b""):
        """Send a debug-level log message to the master.

        Args:
            msg_bytes (bytes): UTF-8 encoded message payload.
        """
        if not self._stop_event.is_set() and CLogLevel.DEBUG >= self._config.base_level:
            self._add_log_to_batch(CLogLevel.DEBUG, len(msg_bytes), <unsigned char*>msg_bytes)
    
    cpdef void info(self, bytes msg_bytes=b""):
        """Send an info-level log message to the master.

        Args:
            msg_bytes (bytes): UTF-8 encoded message payload.
        """
        if not self._stop_event.is_set() and CLogLevel.INFO >= self._config.base_level:
            self._add_log_to_batch(CLogLevel.INFO, len(msg_bytes), <unsigned char*>msg_bytes)
    
    cpdef void warning(self, bytes msg_bytes=b""):
        """Send a warning-level log message to the master.

        Args:
            msg_bytes (bytes): UTF-8 encoded message payload.
        """
        if not self._stop_event.is_set() and CLogLevel.WARNING >= self._config.base_level:
            self._add_log_to_batch(CLogLevel.WARNING, len(msg_bytes), <unsigned char*>msg_bytes)
    
    cpdef void error(self, bytes msg_bytes=b""):
        """Send an error-level log message to the master.

        Args:
            msg_bytes (bytes): UTF-8 encoded message payload.
        """
        if not self._stop_event.is_set() and CLogLevel.ERROR >= self._config.base_level:
            self._add_log_to_batch(CLogLevel.ERROR, len(msg_bytes), <unsigned char*>msg_bytes)

    cpdef void shutdown(self):
        """Gracefully shut down the worker logger.

        Performs a final flush, stops the background thread, and removes
        the thread-local singleton registration.
        """
        if self._stop_event.is_set():
            return
        
        if self._config.emit_internal:
            try:
                self.debug(msg_bytes=f"Shutting down worker logger; name: {self._name.decode()}".encode('utf-8'))
            except Exception:
                pass
        
        self._stop_event.set()
        
        # Final flush, having kept self._stop_event clear until this 
        # point ensures that no more logs will be added into the batch.
        try:
            self._flush_logs()
        except Exception:
            pass
        
        # Wait for thread and cleanup
        try:
            self._timed_operations_thread.join()
        except Exception:
            pass
        
        try:
            self._transport.close()
        except Exception:
            pass

        # Remove singleton registration — MUST always run
        if hasattr(_worker_logger_local, 'logger'):
            delattr(_worker_logger_local, 'logger')

    cpdef bint is_running(self):
        """Check whether the logger is active.

        Returns:
            bool: True if the logger has not been shut down.
        """
        return not self._stop_event.is_set()
    
    cpdef str get_name(self):
        """Return the logger name.

        Returns:
            str: The decoded worker name.
        """
        return self._name.decode()

    cpdef object get_config(self):
        """Return the logger configuration.

        Returns:
            LoggerConfig: The active configuration object.
        """
        return self._config
