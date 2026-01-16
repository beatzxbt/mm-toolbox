import contextlib
import threading
import time

from libc.stdint cimport (
    uint8_t as u8,
    uint32_t as u32,
    uint64_t as u64,
)

from mm_toolbox.ringbuffer.ipc import IPCRingBufferConsumer, IPCRingBufferConfig

from mm_toolbox.logging.advanced.handlers.base import BaseLogHandler
from mm_toolbox.logging.advanced.config cimport LoggerConfig
from mm_toolbox.logging.advanced.log cimport CLogLevel
from mm_toolbox.logging.advanced.protocol cimport BinaryReader
from mm_toolbox.logging.advanced.pylog import PyLog, PyLogLevel
from mm_toolbox.logging.advanced.worker cimport WorkerLogger

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

        # Verify that all handlers are valid
        for handler in self._log_handlers:
            if not isinstance(handler, BaseLogHandler):
                raise TypeError(f"Invalid handler type; expected BaseLogHandler but got {handler.__class__}")

            # Mainly for forwarding the str_format to the handler for formatting log messages
            # where the final point is not a code environment (eg Discord, Telegram, etc).
            handler.add_primary_config(self._config)

        # Transport is created and owned by the background thread to avoid cross-thread ZMQ usage
        self._transport = None

        self._is_running = True
        self._worker = WorkerLogger(config=self._config, name="MASTER")

        self._timed_operations_thread = threading.Thread(
            target=self._timed_operations,
            daemon=True
        )
        self._timed_operations_thread.start()
        if self._config.emit_internal:
            self.debug("Master logger started")

    cdef list _decode_worker_message(self, bytes serialized_message):
        """Decode binary CLog messages from workers into PyLog objects."""
        cdef:
            BinaryReader reader = BinaryReader(serialized_message)
            u8 msg_type = reader.read_u8()  # Ignore if not needed
            u64 batch_ts = reader.read_u64()  # Ignore if not needed
            u32 data_len = reader.read_u32()
            bytes data = reader.read_bytes(data_len)

            BinaryReader data_reader = BinaryReader(data)
            u32 worker_name_len = data_reader.read_u32()
            bytes worker_name = data_reader.read_bytes(worker_name_len)  # Batch-level name; can use or ignore
            u32 num_logs = data_reader.read_u32()

            list decoded_logs = []
            u64 timestamp_ns
            u32 name_len
            bytes name
            u8 level_int
            object pylevel
            u32 message_len
            bytes message

            u32 i

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
                pylevel = PyLogLevel.INFO  # Default fallback, should never happen though
            
            decoded_logs.append(PyLog(
                timestamp_ns=timestamp_ns,
                name=name,
                level=pylevel,
                message=message
            ))

        return decoded_logs

    cpdef void _timed_operations(self):
        """Background thread that periodically receives and flushes logs."""
        # Create IPC transport in this thread and own its lifetime here
        self._transport = IPCRingBufferConsumer(
            IPCRingBufferConfig(
                path=self._config.path,
                backlog=10000,
                num_producers=2,  # >1 workers to indicate MPSC
                num_consumers=1,  # Single master
                linger_ms=self._config.ipc_linger_ms,
            )
        )

        try:
            while self._is_running:
                try:
                    # Non-blocking drain to avoid cross-thread close on a blocking recv
                    for message in self._transport.consume_all():
                        decoded_logs = self._decode_worker_message(message)
                        for handler in self._log_handlers:
                            try:
                                handler.push(decoded_logs)
                            except Exception as e:
                                handler._handle_exception(e, "push")  # type: ignore[attr-defined]
                except Exception as e:
                    if self._is_running:
                        self.error(f"Error consuming messages: {e}")

                # Pace the loop using the configured flush interval
                time.sleep(self._config.flush_interval_s)

            # Final best-effort drain after stop signal
            for message in self._transport.consume_all():
                decoded_logs = self._decode_worker_message(message)
                for handler in self._log_handlers:
                    try:
                        handler.push(decoded_logs)
                    except Exception as e:
                        handler._handle_exception(e, "push")  # type: ignore[attr-defined]
        finally:
            if self._transport is not None:
                self._transport.stop()

    cpdef void trace(self, str msg_str=None, bytes msg_bytes=b""):
        """Send a trace-level log message."""
        if msg_str is not None and msg_bytes:
            raise TypeError("Provide only one of msg_str or msg_bytes")
        if (
            self._is_running
            and self._worker.is_running()
            and self._config.base_level <= CLogLevel.TRACE
        ):
            self._worker.trace(msg_str=msg_str, msg_bytes=msg_bytes)
    
    cpdef void debug(self, str msg_str=None, bytes msg_bytes=b""):
        """Send a debug-level log message."""
        if msg_str is not None and msg_bytes:
            raise TypeError("Provide only one of msg_str or msg_bytes")
        if (
            self._is_running
            and self._worker.is_running()
            and self._config.base_level <= CLogLevel.DEBUG
        ):
            self._worker.debug(msg_str=msg_str, msg_bytes=msg_bytes)
    
    cpdef void info(self, str msg_str=None, bytes msg_bytes=b""):
        """Send an info-level log message."""
        if msg_str is not None and msg_bytes:
            raise TypeError("Provide only one of msg_str or msg_bytes")
        if (
            self._is_running
            and self._worker.is_running()
            and self._config.base_level <= CLogLevel.INFO
        ):
            self._worker.info(msg_str=msg_str, msg_bytes=msg_bytes)
    
    cpdef void warning(self, str msg_str=None, bytes msg_bytes=b""):
        """Send a warning-level log message."""
        if msg_str is not None and msg_bytes:
            raise TypeError("Provide only one of msg_str or msg_bytes")
        if (
            self._is_running
            and self._worker.is_running()
            and self._config.base_level <= CLogLevel.WARNING
        ):
            self._worker.warning(msg_str=msg_str, msg_bytes=msg_bytes)
    
    cpdef void error(self, str msg_str=None, bytes msg_bytes=b""):
        """Send an error-level log message.""" 
        if msg_str is not None and msg_bytes:
            raise TypeError("Provide only one of msg_str or msg_bytes")
        if (
            self._is_running
            and self._worker.is_running()
            and self._config.base_level <= CLogLevel.ERROR
        ):
            self._worker.error(msg_str=msg_str, msg_bytes=msg_bytes)

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
        with contextlib.suppress(Exception):
            if self._worker is not None:
                self._worker.shutdown()

        # Join background thread which owns the transport; it will perform final drain and stop
        self._timed_operations_thread.join()

        # Close handlers (best-effort)
        try:
            for handler in self._log_handlers:
                handler.close()
        except Exception:
            pass
    
    cpdef bint is_running(self):
        """Check if the master logger is running."""
        return self._is_running

    cpdef LoggerConfig get_config(self):
        """Get the configuration of the master logger."""
        return self._config
