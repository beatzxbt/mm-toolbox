import threading

from libc.stdint cimport (
    uint8_t as u8,
    uint32_t as u32,
    uint64_t as u64,
)

from mm_toolbox.time.time cimport time_ns
from mm_toolbox.ringbuffer.ipc import IPCRingBufferConsumer, IPCRingBufferConfig

from mm_toolbox.logging.advanced.handlers.base import BaseLogHandler
from mm_toolbox.logging.advanced.config cimport LoggerConfig
from mm_toolbox.logging.advanced.log cimport CLogLevel
from mm_toolbox.logging.advanced.protocol cimport BinaryReader
from mm_toolbox.logging.advanced.pylog import PyLog, PyLogLevel

cdef class MasterLogger:
    """
    The MasterLogger acts as a central aggregator for log messages sent by worker loggers.

    It receives binary messages from workers, decodes them, and forwards them to handlers.

    Also can act as a logger itself, but it is not recommended to use it for this purpose.
    """
    def __cinit__(
        self, 
        LoggerConfig config=None, 
        list log_handlers=None,  
    ):
        self._config = config
        if self._config is None:
            self._config = LoggerConfig()

        self._log_handlers = log_handlers
        if self._log_handlers is None:
            self._log_handlers: list[BaseLogHandler] = []

        self._name = b"MASTER"

        # Verify that all handlers are valid 
        for handler in self._log_handlers:
            handler_base_class = handler.__class__.__base__
            if not handler_base_class == BaseLogHandler:
                raise TypeError(f"Invalid handler base class; expected BaseLogHandler but got {handler_base_class}")

            # Mainly for forwarding the str_format to the handler for formatting log messages
            # where the final point is not a code environment (eg Discord, Telegram, etc).
            handler.add_primary_config(self._config)

        self._num_pending_logs = 0
        self._pending_logs: list[PyLog] = []

        # Create IPC configuration for sending messages to master, multiple producers (workers), single consumer (master)
        self._transport = IPCRingBufferConsumer(
            IPCRingBufferConfig(
                path=self._config.path,
                backlog=10000,  
                num_producers=2,  # >1 workers to indicate MPSC
                num_consumers=1   # Single master
            )
        )

        self._is_running = True

        self._timed_operations_thread = threading.Thread(
            target=self._timed_operations,
            daemon=True
        )
        self._timed_operations_thread.start()

        self.info("Master logger started")

    cdef list _decode_worker_message(self, bytes serialized_message):
        """Decode binary CLog messages from workers into PyLog objects."""
        cdef BinaryReader reader = BinaryReader(serialized_message)
        cdef u8 msg_type = reader.read_u8()  # Ignore if not needed
        cdef u64 batch_ts = reader.read_u64()  # Ignore if not needed
        cdef u32 data_len = reader.read_u32()
        cdef bytes data = reader.read_bytes(data_len)

        cdef BinaryReader data_reader = BinaryReader(data)
        cdef u32 worker_name_len = data_reader.read_u32()
        cdef bytes worker_name = data_reader.read_bytes(worker_name_len)  # Batch-level name; can use or ignore
        cdef u32 num_logs = data_reader.read_u32()

        cdef list decoded_logs = []
        cdef u64 timestamp_ns
        cdef u32 name_len
        cdef bytes name
        cdef u8 level_int
        cdef object pylevel
        cdef u32 message_len
        cdef bytes message

        cdef int i
        for i in range(num_logs):
            timestamp_ns = data_reader.read_u64()
            name_len = data_reader.read_u32()
            name = data_reader.read_bytes(name_len)
            level_int = data_reader.read_u8()
            message_len = data_reader.read_u32()
            message = data_reader.read_bytes(message_len)

            # Convert CLogLevel int to PyLogLevel
            if level_int == 0:  # TRACE
                pylevel = PyLogLevel.TRACE
            elif level_int == 1:  # DEBUG
                pylevel = PyLogLevel.DEBUG
            elif level_int == 2:  # INFO
                pylevel = PyLogLevel.INFO
            elif level_int == 3:  # WARNING
                pylevel = PyLogLevel.WARNING
            elif level_int == 4:  # ERROR
                pylevel = PyLogLevel.ERROR
            else:
                pylevel = PyLogLevel.INFO  # Default fallback
            
            decoded_logs.append(PyLog(
                timestamp_ns=timestamp_ns,
                name=name,
                level=pylevel,
                message=message
            ))

        return decoded_logs

    cpdef void _timed_operations(self):
        """Background thread that periodically flushes the local log buffer."""
        while self._is_running:
            try:
                if message := self._transport.consume():
                    decoded_logs = self._decode_worker_message(message)
                    for handler in self._log_handlers:
                        handler.push(decoded_logs)

                if self._num_pending_logs > 0:
                    for handler in self._log_handlers:
                        handler.push(self._pending_logs[:self._num_pending_logs])
                    self._num_pending_logs = 0  # Reset counter; avoid clear() for minor perf

            except Exception as e:
                if self._is_running:
                    self.error(f"Error consuming messages: {e}")

    cdef inline object _make_pylog(self, object level, bytes message):
        """Make a PyLog object."""
        return PyLog(
            timestamp_ns=time_ns(),
            name=self._name,
            level=level,
            message=message
        )

    cdef void _add_pylog_to_batch(self, object pylog):
        """Add a log to the batch."""
        self._pending_logs.append(pylog)
        self._num_pending_logs += 1

    cpdef void trace(self, str msg_str=None, bytes msg_bytes=b""):
        """Send a trace-level log message."""
        if self._is_running and self._config.base_level <= CLogLevel.TRACE:
            message = msg_str.encode('utf-8') if msg_str else msg_bytes
            self._add_pylog_to_batch(self._make_pylog(PyLogLevel.TRACE, message))
    
    cpdef void debug(self, str msg_str=None, bytes msg_bytes=b""):
        """Send a debug-level log message."""
        if self._is_running and self._config.base_level <= CLogLevel.DEBUG:
            message = msg_str.encode('utf-8') if msg_str else msg_bytes
            self._add_pylog_to_batch(self._make_pylog(PyLogLevel.DEBUG, message))
    
    cpdef void info(self, str msg_str=None, bytes msg_bytes=b""):
        """Send an info-level log message."""
        if self._is_running and self._config.base_level <= CLogLevel.INFO:
            message = msg_str.encode('utf-8') if msg_str else msg_bytes
            self._add_pylog_to_batch(self._make_pylog(PyLogLevel.INFO, message))
    
    cpdef void warning(self, str msg_str=None, bytes msg_bytes=b""):
        """Send a warning-level log message."""
        if self._is_running and self._config.base_level <= CLogLevel.WARNING:
            message = msg_str.encode('utf-8') if msg_str else msg_bytes
            self._add_pylog_to_batch(self._make_pylog(PyLogLevel.WARNING, message))
    
    cpdef void error(self, str msg_str=None, bytes msg_bytes=b""):
        """Send an error-level log message.""" 
        if self._is_running and self._config.base_level <= CLogLevel.ERROR:
            message = msg_str.encode('utf-8') if msg_str else msg_bytes
            self._add_pylog_to_batch(self._make_pylog(PyLogLevel.ERROR, message))

    cpdef void shutdown(self):
        """
        Flush any remaining messages and shuts down the master logger.

        This method stops accepting new messages from worker loggers and 
        then stops the connection.

        Warning:
            After calling `shutdown()`, this logger cannot be used again.
        """
        if not self._is_running:
            return
        
        # Prevents any more logs from being added to the batch
        self._is_running = False

        # Final consume and flush
        while message := self._transport.consume():
            decoded_logs = self._decode_worker_message(message)
            for handler in self._log_handlers:
                handler.push(decoded_logs)

        if self._num_pending_logs > 0:
            for handler in self._log_handlers:
                handler.push(self._pending_logs[:self._num_pending_logs])
            self._num_pending_logs = 0

        self._timed_operations_thread.join()

        self._transport.stop()
    
    cpdef bint is_running(self):
        """Check if the master logger is running."""
        return self._is_running

    cpdef LoggerConfig get_config(self):
        """Get the configuration of the master logger."""
        return self._config