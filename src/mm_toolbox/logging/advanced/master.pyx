"""MasterLogger implementation.

Aggregates binary log messages from WorkerLogger instances, decodes them,
and forwards decoded PyLog records to a list of handlers.  Runs a background
thread that drains a shared-memory MPSC ring.
"""

import contextlib
import threading
import time

from libc.string cimport memcpy
from libc.stdint cimport (
    uint8_t as u8,
    uint32_t as u32,
    uint64_t as u64,
)

from mm_toolbox.ringbuffer.shm.mpsc import ShmMpscConsumer, ShmMpscProducer

from mm_toolbox.logging.advanced.handlers.base import BaseLogHandler
from mm_toolbox.logging.advanced.config cimport LoggerConfig
from mm_toolbox.logging.advanced.log cimport CLogLevel
from mm_toolbox.logging.advanced.protocol cimport BinaryReader
from mm_toolbox.logging.advanced.pylog import PyLog, PyLogLevel
from mm_toolbox.time.time cimport time_ns

cdef class MasterLogger:
    """Central log aggregator that receives binary messages from workers.

    Decodes batched CLog messages and forwards PyLog records to registered
    handlers.  Runs a background thread to drain a shared-memory MPSC ring.

    Attributes:
        _config (LoggerConfig): Active logger configuration.
        _log_handlers (list[BaseLogHandler]): Registered output handlers.
        _shm_producer (ShmMpscProducer): Shared-memory ring producer.
        _transport (ShmMpscConsumer): IPC transport for message consumption.
        _stop_event (threading.Event): Signals shutdown to the background thread.
        _timed_operations_thread (threading.Thread): Background drain/flush thread.
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

        # Create the SHM ring. The master owns it and will clean it up on shutdown.
        self._shm_producer = ShmMpscProducer(
            path=self._config.path,
            capacity_bytes=self._config.shm_capacity_bytes,
            num_rings=0,
            create=True,
            unlink_on_close=True,
        )
        self._transport = None

        self._stop_event = threading.Event()

        self._timed_operations_thread = threading.Thread(
            target=self._timed_operations,
            daemon=True
        )
        self._timed_operations_thread.start()

    cpdef list _decode_worker_message(self, bytes serialized_message):
        """Decode a batched binary message from a worker into PyLog objects.

        Args:
            serialized_message (bytes): Raw binary payload from the SHM ring.

        Returns:
            list[PyLog]: Decoded log records.

        Raises:
            ValueError: If the message is malformed or truncated.
        """
        cdef:
            BinaryReader reader = BinaryReader(serialized_message)
            const unsigned char[:] buffer_view = serialized_message
            object serialized_view = memoryview(serialized_message)
            u8 msg_type = reader.read_u8()  # Ignore if not needed
            u64 batch_ts = reader.read_u64()  # Ignore if not needed
            u32 data_len = reader.read_u32()
            u32 data_start = reader._pos
            u32 data_end = data_start + data_len

            u32 cursor
            u32 worker_name_len
            u32 worker_name_start
            u32 worker_name_end
            bytes worker_name
            u32 num_logs

            list decoded_logs = []
            u64 timestamp_ns
            u8 level_int
            object pylevel
            u32 message_len
            u32 message_start
            u32 message_end
            object message_view

            u32 i

        # Integer overflow check
        if data_start > reader._len or data_len > reader._len - data_start:
            raise ValueError("Message data_len exceeds available buffer")

        if data_end > reader._len:
            raise ValueError("Buffer underrun reading bytes")

        cursor = data_start

        if cursor + 4 > data_end:
            raise ValueError("Buffer underrun reading worker name length")
        memcpy(&worker_name_len, &buffer_view[cursor], sizeof(u32))
        cursor += 4

        worker_name_start = cursor
        worker_name_end = worker_name_start + worker_name_len
        if worker_name_end > data_end:
            raise ValueError("Buffer underrun reading worker name")
        worker_name = bytes(serialized_view[worker_name_start:worker_name_end])
        cursor = worker_name_end

        if cursor + 4 > data_end:
            raise ValueError("Buffer underrun reading log count")
        memcpy(&num_logs, &buffer_view[cursor], sizeof(u32))
        cursor += 4

        # Validate num_logs against remaining bytes
        cdef u32 max_possible_logs = (data_end - cursor) / 13
        if num_logs > max_possible_logs:
            raise ValueError("num_logs impossibly large for payload")

        for i in range(num_logs):
            if cursor + 8 > data_end:
                raise ValueError("Buffer underrun reading log timestamp")
            memcpy(&timestamp_ns, &buffer_view[cursor], sizeof(u64))
            cursor += 8

            if cursor + 1 > data_end:
                raise ValueError("Buffer underrun reading log level")
            level_int = <u8>buffer_view[cursor]
            cursor += 1

            if cursor + 4 > data_end:
                raise ValueError("Buffer underrun reading message length")
            memcpy(&message_len, &buffer_view[cursor], sizeof(u32))
            cursor += 4

            message_start = cursor
            message_end = message_start + message_len
            if message_end > data_end:
                raise ValueError("Buffer underrun reading log message")
            message_view = serialized_view[message_start:message_end]
            cursor = message_end

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
                name=worker_name,
                level=pylevel,
                message=message_view
            ))

        return decoded_logs

    cpdef void _timed_operations(self):
        """Background thread target that drains and dispatches log messages.

        Runs until ``_stop_event`` is set, then performs a final best-effort
        drain before exiting.
        """
        # Create IPC transport in this thread and own its lifetime here
        self._transport = ShmMpscConsumer(path=self._config.path)

        try:
            while not self._stop_event.is_set():
                try:
                    # Non-blocking drain
                    messages = self._transport.consume_all()
                    for message in messages:
                        decoded_logs = self._decode_worker_message(message)
                        for handler in self._log_handlers:
                            try:
                                handler.push(decoded_logs)
                            except Exception as e:
                                # One bad handler must not kill the master
                                if self._config.emit_internal:
                                    self.error(msg_bytes=f"Handler error: {e}".encode('utf-8'))
                except Exception as e:
                    if not self._stop_event.is_set():
                        if self._config.emit_internal:
                            self.error(msg_bytes=f"Error consuming messages: {e}".encode('utf-8'))

                # Pace the loop using a short poll interval
                time.sleep(0.001)

            # Final best-effort drain after stop signal
            messages = self._transport.consume_all()
            for message in messages:
                decoded_logs = self._decode_worker_message(message)
                for handler in self._log_handlers:
                    try:
                        handler.push(decoded_logs)
                    except Exception:
                        pass
        finally:
            if self._transport is not None:
                self._transport.close()

    cdef void _log_direct(self, CLogLevel level, bytes msg_bytes):
        """Send a log directly to handlers without going through IPC.

        Args:
            level (CLogLevel): Severity level.
            msg_bytes (bytes): UTF-8 encoded message payload.
        """
        cdef object py_level
        if level == CLogLevel.TRACE:
            py_level = PyLogLevel.TRACE
        elif level == CLogLevel.DEBUG:
            py_level = PyLogLevel.DEBUG
        elif level == CLogLevel.INFO:
            py_level = PyLogLevel.INFO
        elif level == CLogLevel.WARNING:
            py_level = PyLogLevel.WARNING
        elif level == CLogLevel.ERROR:
            py_level = PyLogLevel.ERROR
        else:
            py_level = PyLogLevel.INFO

        cdef list log = [PyLog(
            timestamp_ns=time_ns(),
            name=b"MASTER",
            level=py_level,
            message=msg_bytes,
        )]
        for handler in self._log_handlers:
            try:
                handler.push(log)
            except Exception:
                pass

    cpdef void trace(self, bytes msg_bytes=b""):
        """Send a trace-level log message directly to handlers.

        Args:
            msg_bytes (bytes): UTF-8 encoded message payload.
        """
        if (
            not self._stop_event.is_set()
            and self._config.base_level <= CLogLevel.TRACE
        ):
            self._log_direct(CLogLevel.TRACE, msg_bytes)
    
    cpdef void debug(self, bytes msg_bytes=b""):
        """Send a debug-level log message directly to handlers.

        Args:
            msg_bytes (bytes): UTF-8 encoded message payload.
        """
        if (
            not self._stop_event.is_set()
            and self._config.base_level <= CLogLevel.DEBUG
        ):
            self._log_direct(CLogLevel.DEBUG, msg_bytes)
    
    cpdef void info(self, bytes msg_bytes=b""):
        """Send an info-level log message directly to handlers.

        Args:
            msg_bytes (bytes): UTF-8 encoded message payload.
        """
        if (
            not self._stop_event.is_set()
            and self._config.base_level <= CLogLevel.INFO
        ):
            self._log_direct(CLogLevel.INFO, msg_bytes)
    
    cpdef void warning(self, bytes msg_bytes=b""):
        """Send a warning-level log message directly to handlers.

        Args:
            msg_bytes (bytes): UTF-8 encoded message payload.
        """
        if (
            not self._stop_event.is_set()
            and self._config.base_level <= CLogLevel.WARNING
        ):
            self._log_direct(CLogLevel.WARNING, msg_bytes)
    
    cpdef void error(self, bytes msg_bytes=b""):
        """Send an error-level log message directly to handlers.

        Args:
            msg_bytes (bytes): UTF-8 encoded message payload.
        """
        if (
            not self._stop_event.is_set()
            and self._config.base_level <= CLogLevel.ERROR
        ):
            self._log_direct(CLogLevel.ERROR, msg_bytes)

    cpdef void shutdown(self):
        """
        Flush any remaining messages and shuts down the master logger.

        This method stops accepting new messages from worker loggers and 
        then stops the connection.

        Warning:
            After calling `shutdown()`, this logger cannot be used again.
        """
        if self._stop_event.is_set():
            return
        
        # Prevents any more logs from being added to the batch
        self._stop_event.set()

        # Join background thread which owns the transport; it will perform final drain and stop
        self._timed_operations_thread.join()

        # Close the SHM ring producer (which unlinks the backing file)
        with contextlib.suppress(Exception):
            if self._shm_producer is not None:
                self._shm_producer.close()

        # Close handlers (best-effort)
        try:
            for handler in self._log_handlers:
                handler.close()
        except Exception:
            pass
    
    cpdef bint is_running(self):
        """Check if the master logger is running."""
        return not self._stop_event.is_set()

    cpdef LoggerConfig get_config(self):
        """Get the configuration of the master logger."""
        return self._config
