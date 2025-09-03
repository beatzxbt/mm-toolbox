import os
import zmq
import time
import threading

from libc.stdint cimport (
    uint8_t as u8,
    uint64_t as u64,
)

from mm_toolbox.logging.advanced.structs cimport (
    LogLevel,
    CLogLevel,
    LogBatch,
    heartbeat_to_bytes,
)

from mm_toolbox.logging.utils.zmq_connection import ZmqConnection
from mm_toolbox.logging.advanced.config cimport LoggerConfig

from mm_toolbox.time.time cimport time_ns


cdef class WorkerLogger:
    """A lightweight worker logger that sends log messages to the master logger."""

    def __cinit__(
        self, 
        LoggerConfig config=None, 
        str name=None,
    ):
        """
        Initialize the WorkerLogger.

        Args:
            config (LoggerConfig): Configuration details for connecting to the master logger.
            name (str, optional): The name of the worker to attach to log messages. 
                Defaults to an empty string.
        """
        self._config = config
        if self._config is None:
            self._config = LoggerConfig()

        self._name = name.encode() if name else f"WORKER{os.getpid()}".encode()
        
        self._log_batch = LogBatch(name=self._name)

        self._conn = ZmqConnection(
            socket_type=zmq.PUSH,
            path=config.path,
            bind=False
        )
        self._conn.start()
        
        self._is_running = True
        
        self._timed_operations_thread = threading.Thread(
            target=self._timed_operations,
            daemon=True
        )
        self._timed_operations_thread.start()

        self.info(f"WorkerLogger started; name: {self._name.decode()}")

    cpdef void _timed_operations(self):
        """
        Background thread that periodically sends heartbeat messages to the master logger.
        """
        cdef u8     CHECKIN_INTERVAL_S = 60 
        cdef u64    ONE_SECOND_IN_NS = 1000000000

        cdef u64    current_time_ns = time_ns()
        cdef double current_time_s = current_time_ns / ONE_SECOND_IN_NS
        cdef double last_flush_time = current_time_s
        cdef double last_heartbeat_time = current_time_s

        while self._is_running:
            time.sleep(0.1)
            
            current_time_ns = time_ns()
            current_time_s = current_time_ns / ONE_SECOND_IN_NS

            # Check if it's time to flush the log buffer
            if (current_time_s - last_flush_time) >= self._config.flush_interval_s:
                if self._log_batch.num_logs_in_batch > 0:
                    log_batch_bytes = self._log_batch.to_bytes(reset=True)
                    self._conn.send(log_batch_bytes)
                last_flush_time = current_time_s
            
            # Check if it's time to send heartbeat
            if (current_time_s - last_heartbeat_time) >= CHECKIN_INTERVAL_S:
                heartbeat_bytes = heartbeat_to_bytes(
                    name=self._name, 
                    time=current_time_ns, 
                    next_checkin_time=current_time_ns + CHECKIN_INTERVAL_S * ONE_SECOND_IN_NS
                )
                self._conn.send(heartbeat_bytes)
                last_heartbeat_time = current_time_s

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
        cdef bytes msg
        cdef bint valid_level = self._config.base_level == CLogLevel.CTRACE
        if self._is_running and valid_level:
            msg = msg_str.encode() if msg_str is not None else msg_bytes
            self._log_batch.add_log(CLogLevel.CTRACE, msg)

    cpdef void debug(self, str msg_str=None, bytes msg_bytes=b""):
        """
        Send a debug-level log message.

        Args:
            msg_str (str, optional): The log message text as a string.
            msg_bytes (bytes, optional): The log message text as bytes.
        """
        cdef bytes msg
        cdef bint valid_level = self._config.base_level <= CLogLevel.CDEBUG
        if self._is_running and valid_level:
            msg = msg_str.encode() if msg_str is not None else msg_bytes
            self._log_batch.add_log(CLogLevel.CDEBUG, msg)

    cpdef void info(self, str msg_str=None, bytes msg_bytes=b""):
        """
        Send an info-level log message.

        Args:
            msg_str (str, optional): The log message text as a string.
            msg_bytes (bytes, optional): The log message text as bytes.
        """
        cdef bytes msg
        cdef bint valid_level = self._config.base_level <= CLogLevel.CINFO
        if self._is_running and valid_level:
            msg = msg_str.encode() if msg_str is not None else msg_bytes
            self._log_batch.add_log(CLogLevel.CINFO, msg)

    cpdef void warning(self, str msg_str=None, bytes msg_bytes=b""):
        """
        Send a warning-level log message.

        Args:
            msg_str (str, optional): The log message text as a string.
            msg_bytes (bytes, optional): The log message text as bytes.
        """
        cdef bytes msg
        cdef bint valid_level = self._config.base_level <= CLogLevel.CWARNING
        if self._is_running and valid_level:
            msg = msg_str.encode() if msg_str is not None else msg_bytes
            self._log_batch.add_log(CLogLevel.CWARNING, msg)

    cpdef void error(self, str msg_str=None, bytes msg_bytes=b""):
        """
        Send an error-level log message.

        Args:
            msg_str (str, optional): The log message text as a string.
            msg_bytes (bytes, optional): The log message text as bytes.
        """
        cdef bytes msg
        cdef bint valid_level = self._config.base_level <= CLogLevel.CERROR
        if self._is_running and valid_level:
            msg = msg_str.encode() if msg_str is not None else msg_bytes
            self._log_batch.add_log(CLogLevel.CERROR, msg)

    cpdef void shutdown(self):
        """
        Flush any remaining messages and shuts down the worker logger.

        This method drains the log buffer, sending any remaining items to the master logger, 
        and then stops the connection.

        Warning:
            After calling `shutdown()`, this logger should not be used again.
        """
        if not self._is_running:
            return
        
        self.warning("Shutting down worker logger; flushing buffers and stopping connection")

        # No new logs can enter the buffers after this point
        self._is_running = False

        if self._log_batch.num_logs_in_batch > 0:
            self._conn.send(self._log_batch.to_bytes(reset=True))
        
        self._timed_operations_thread.join()

        # Send a veeery long heartbeat time to ensure the MasterLogger doesnt
        # follow up on this worker. This is simpler than having a dedicated
        # shutdown message.
        heartbeat_bytes = heartbeat_to_bytes(
            name=self._name, 
            time=time_ns(), 
            next_checkin_time=time_ns() + 1000000000000000000
        )
        self._conn.send(heartbeat_bytes)

        self._conn.stop()

    cpdef bint is_running(self):
        """
        Check if the logger is running.
        """
        return self._is_running
    
    cpdef str get_name(self):
        """
        Get the name of the logger.
        """
        return self._name.decode()

    cpdef LoggerConfig get_config(self):
        """
        Get the configuration of the logger.
        """
        return self._config