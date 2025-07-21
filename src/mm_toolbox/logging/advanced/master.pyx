import zmq
import threading

from libc.stdint cimport (
    uint8_t as u8,
    uint64_t as u64,
)

from mm_toolbox.logging.utils.zmq import ZmqConnection
from mm_toolbox.logging.advanced.handlers.base import BaseLogHandler

from mm_toolbox.time.time cimport time_ns
from mm_toolbox.logging.advanced.config cimport LoggerConfig
from mm_toolbox.logging.advanced.structs cimport (
    BufMsgType,
    LogBatch,
    LogLevel, 
    CLogLevel,
    heartbeat_from_bytes,
)

cdef class MasterLogger:
    """
    The MasterLogger acts as a central aggregator for log messages sent by worker loggers.
    It receives binary, multipart messages, decodes them, and stores them in a buffer until full.
    Once the buffer is full, it flushes them to external handlers.
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
            handler.add_primary_config(config)

        self._log_batch = LogBatch(name=self._name)

        self._heartbeats: dict[bytes, tuple[int, int]] = {} # name -> (time_ns, next_checkin_time_ns)

        self._conn = ZmqConnection(
            socket_type=zmq.PULL,
            path=config.path,
            bind=True
        )
        self._conn.start()
        self._conn.listen(self._process_worker_msg)

        self._is_running = True

        self._timed_operations_thread = threading.Thread(
            target=self._timed_operations,
            daemon=True
        )
        self._timed_operations_thread.start()

        self.info(f"Master logger started;")

    cpdef void _process_worker_msg(self, bytes msg):
        """
        Process received multipart messages and decode them into appropriate message structs.
        
        Args:
            msg (bytes): The raw message bytes to process.
        """
        if not self._is_running:
            return 
            
        # First byte is the message type
        cdef u8 msg_type = msg[0]

        if msg_type == BufMsgType.LOG:
            cdef tuple[bytes, list[tuple]] log_batch = LogBatch.from_bytes(msg)
            name, logs = log_batch
            for handler in self._log_handlers:
                handler.push(name, logs)

        elif msg_type == BufMsgType.HEARTBEAT:
            cdef tuple[bytes, u64, u64] heartbeat = heartbeat_from_bytes(msg)
            name, time, next_checkin_time = heartbeat
            self._heartbeats[name] = (time, next_checkin_time)
            self.trace(f"Received heartbeat from worker; name: {name.decode()}")            

    cpdef void _timed_operations(self):
        """
        Background thread that periodically flushes the local log buffer and 
        checks worker heartbeats for being late.
        """
        cdef u64    ONE_SECOND_IN_NS = 1000000000
        cdef u64    current_time_ns = time_ns()
        cdef double current_time_s = <double>(current_time_ns / ONE_SECOND_IN_NS)
        cdef double last_flush_time = current_time_s

        while self._is_running:
            time.sleep(0.1)
            
            current_time_ns = time_ns()
            current_time_s = <double>(current_time_ns / ONE_SECOND_IN_NS)

            # Check if it's time to flush the log buffer
            if (current_time_s - last_flush_time) >= self._config.flush_interval_s:
                if self._log_batch.num_logs_in_batch > 0:
                    logs = self._log_batch.get_all_logs(reset=True)
                    for handler in self._log_handlers:
                        handler.push(self._name, logs)
                last_flush_time = current_time_s

            # Check if any heartbeats are overdue (allow 1s delay before emitting warnings of dead workers)
            for name, (time, next_checkin_time) in self._heartbeats.items():                
                if current_time_ns < next_checkin_time + ONE_SECOND_IN_NS: 
                    self.warning(f"Worker {name.decode()} has not checked in for {next_checkin_time - time} seconds")
                    self._heartbeats.pop(name)
            
    cpdef void set_log_level(self, LogLevel level):
        """
        Modify the logger's base log level at runtime.

        Args:
            level (LogLevel): The new base log level.
        """
        self.debug(f"Changing base log level to '{level.value}'")
        self._config.set_base_level(level)

    cpdef void trace(self, str msg_str=None, bytes msg_bytes=b""):
        """
        Send a trace-level log message.

        Args:
            msg_str (str, optional): The log message text as a string.
            msg_bytes (bytes, optional): The log message text as bytes.
        """
        cdef bint valid_level = self._config.base_level == CLogLevel.TRACE
        if self._is_running and valid_level:
            cdef bytes msg = msg_str.encode() if msg_str is not None else msg_bytes
            self._log_batch.add_log(CLogLevel.TRACE, msg)

    cpdef void debug(self, str msg_str=None, bytes msg_bytes=b""):
        """
        Send a debug-level log message.

        Args:
            msg_str (str, optional): The log message text as a string.
            msg_bytes (bytes, optional): The log message text as bytes.
        """
        cdef bint valid_level = self._config.base_level <= CLogLevel.DEBUG
        if self._is_running and valid_level:
            cdef bytes msg = msg_str.encode() if msg_str is not None else msg_bytes
            self._log_batch.add_log(CLogLevel.DEBUG, msg)

    cpdef void info(self, str msg_str=None, bytes msg_bytes=b""):
        """
        Send an info-level log message.

        Args:
            msg_str (str, optional): The log message text as a string.
            msg_bytes (bytes, optional): The log message text as bytes.
        """
        cdef bint valid_level = self._config.base_level <= CLogLevel.INFO
        if self._is_running and valid_level:
            cdef bytes msg = msg_str.encode() if msg_str is not None else msg_bytes
            self._log_batch.add_log(CLogLevel.INFO, msg)

    cpdef void warning(self, str msg_str=None, bytes msg_bytes=b""):
        """
        Send a warning-level log message.

        Args:
            msg_str (str, optional): The log message text as a string.
            msg_bytes (bytes, optional): The log message text as bytes.
        """
        cdef bint valid_level = self._config.base_level <= CLogLevel.WARNING
        if self._is_running and valid_level:
            cdef bytes msg = msg_str.encode() if msg_str is not None else msg_bytes
            self._log_batch.add_log(CLogLevel.WARNING, msg)

    cpdef void error(self, str msg_str=None, bytes msg_bytes=b""):
        """
        Send an error-level log message.

        Args:
            msg_str (str, optional): The log message text as a string.
            msg_bytes (bytes, optional): The log message text as bytes.
        """
        cdef bint valid_level = self._config.base_level <= CLogLevel.ERROR
        if self._is_running and valid_level:
            cdef bytes msg = msg_str.encode() if msg_str is not None else msg_bytes
            self._log_batch.add_log(CLogLevel.ERROR, msg)

    cpdef void critical(self, str msg_str=None, bytes msg_bytes=b""):
        """
        Send a critical-level log message.

        Args:
            msg_str (str, optional): The log message text as a string.
            msg_bytes (bytes, optional): The log message text as bytes.
        """
        cdef bint valid_level = self._config.base_level <= CLogLevel.CRITICAL
        if self._is_running and valid_level:
            cdef bytes msg = msg_str.encode() if msg_str is not None else msg_bytes
            self._log_batch.add_log(CLogLevel.CRITICAL, msg)

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
            
        # No longer accept new messages from worker loggers from this point on
        self._is_running = False

        if self._log_batch.num_logs_in_batch > 0:
            logs = self._log_batch.get_all_logs(reset=True)
            for handler in self._log_handlers:
                handler.push(self._name, logs)

        self._timed_operations_thread.join()

        self._conn.stop()
    
    cpdef bint is_running(self):
        """
        Check if the master logger is running.
        """
        return self._is_running

    cpdef LoggerConfig get_config(self):
        """
        Get the configuration of the master logger.
        """
        return self._config
