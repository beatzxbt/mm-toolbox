import zmq
import msgspec
from libc.stdint cimport uint8_t

from mm_toolbox.time.time cimport time_ns, time_us

from ..utils import ZmqConnection, _get_system_info

from .config cimport LoggerConfig
from .structs import (
    LogMessage, 
    LogMessageBatch, 
    DataMessage, 
    DataMessageBatch,
    log_level_to_str
)
from .structs cimport LogLevel, MessageBuffer

cdef class WorkerLogger:
    """
    A lightweight worker logger that sends log messages and data to the master logger.

    This class manages two internal buffers—one for log messages and one for data messages—
    and pushes them to the master logger when full or upon request.
    """

    def __init__(
        self, 
        LoggerConfig config=None, 
        str name="",
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
            
        self._system_info = _get_system_info(
            machine=True, 
            network=True, 
            op_sys=True
        )
        self._name = name.encode()

        self._batch_encoder = msgspec.msgpack.Encoder()

        # PUSH/PULL sockets are more performant for MPSC style queues.
        self._master_conn = ZmqConnection(
            socket_type=zmq.PUSH,
            path=config.path,
            bind=False
        )
        self._master_conn.start()

        self._log_buffer = MessageBuffer(
            dump_to_queue_callback=self._logs_dump_to_queue_callback,
            capacity=config.log_buffer_size,
            timeout_s=config.log_timeout_s,
        )
        self._data_buffer = MessageBuffer(
            dump_to_queue_callback=self._data_dump_to_queue_callback,
            capacity=config.data_buffer_size,
            timeout_s=config.data_timeout_s,
        )

        self._is_running = True

        self.info(f"Worker logger initialized; name: {self._name.decode()}")

    cdef inline void _ensure_running(self):
        """
        Ensure that the worker logger is running. 
        If it is not, raise a RuntimeError.
        """
        if not self._is_running:
            raise RuntimeError("Worker logger is not running; cannot send/recv messages")

    cdef void _logs_dump_to_queue_callback(self, list raw_log_buffer):
        """
        Internal callback to push batched log messages to the master logger.

        Args:
            raw_log_buffer (list): A list of LogMessage objects to be encoded and sent.
        """ 
        # This can be faster if we output the size of the buffer alongside
        # the list of LogMessage objects as a tuple. But this is a micro-optimization
        # that should be saved for the next minor release.
        cdef Py_ssize_t log_buffer_size = len(raw_log_buffer)

        cdef object log_batch = LogMessageBatch(
            system=self._system_info,
            name=self._name,
            time=time_us(),
            size=log_buffer_size,
            data=raw_log_buffer
        ) 

        cdef bytes log_batch_encoded = self._batch_encoder.encode(log_batch)
        self._master_conn.send(log_batch_encoded)

    cdef void _data_dump_to_queue_callback(self, list raw_data_buffer):
        """
        Internal callback to push batched data messages to the master logger.

        Args:
            raw_data_buffer (list[DataMessage]): A list of DataMessage objects.
        """
        # This can be faster if we output the size of the buffer alongside
        # the list of DataMessage objects as a tuple. But this is a micro-optimization
        # that should be saved for the next minor release.
        cdef Py_ssize_t data_buffer_size = len(raw_data_buffer)

        cdef object data_batch = DataMessageBatch(
            system=self._system_info,
            name=self._name,
            time=time_us(),
            size=data_buffer_size,
            data=raw_data_buffer
        ) 

        cdef bytes data_batch_encoded = self._batch_encoder.encode(data_batch)
        self._master_conn.send(data_batch_encoded)

    cdef inline void _process_log(self, uint8_t level, bytes msg):
        """
        Process a single log message by creating a LogMessage struct and buffering it.

        Args:
            level (uint8_t): The log level value, e.g. LogLevel.INFO (would be == 2)
            msg (bytes): The log message text, encoded as bytes.
        """   
        # We could use nanosecond time here, but realistically the maximum
        # granularity needed would be microseconds. If nanoseconds are needed,
        # just swap it out for time_ns() accordingly.
        cdef object log_msg_struct = LogMessage(
            time=time_us(), 
            level=level,
            msg=msg
        )
        self._log_buffer.append(log_msg_struct)

    cdef inline void _process_data(self, object msg):
        """
        Process a single data item by buffering it for later batch sending.

        Args:
            msg (object): The msg to buffer. DataMessage.
        """
        # We could use nanosecond time here, but realistically the maximum
        # granularity needed would be microseconds. If nanoseconds are needed,
        # just swap it out for time_ns() accordingly.
        data_msg_struct = DataMessage(
            time=time_us(),
            msg=msg
        )
        self._data_buffer.append(data_msg_struct)

    cpdef void set_format(self, str format_string):
        """
        Modify the format string for log messages in runtime.

        Args:
            format_string (str): The new format string.
                Supports {timestamp}, {level}, and {message} placeholders.
        """
        self.debug(f"Changing format string from {self._config.str_format} to {format_string}")
        self._config.str_format = format_string

    cpdef void set_log_level(self, int level):
        """
        Modify the logger's base log level at runtime.

        Args:
            level (LogLevel): The new base log level.
        """
        self.debug(f"Changing base log level from '{log_level_to_str(self._config.base_level)}' to '{log_level_to_str(level)}'")
        self._config.base_level = level

    cpdef void data(self, object data, bint unsafe=False):
        """
        Enqueue data to be sent to the master logger.

        Args:
            data (object): A msgspec.Struct representing the data.
            unsafe (bool, optional): If True, skips type checking on data for performance.
                Only use when you're certain all data objects are msgspec.Structs. Defaults to False.

        Raises:
            TypeError: If data is not a msgspec.Struct and type checking is enabled.
        """
        if not self._is_running:
            return
            
        if unsafe:
            self._process_data(data)
        else:
            if isinstance(data, msgspec.Struct):
                self._process_data(data)
            else:
                raise TypeError(
                    f"Invalid data type; expected 'msgspec.Struct' but got '{type(data).__name__}'"
                )

    cpdef void trace(self, str msg):
        """
        Send a trace-level log message.

        Args:
            msg (str): The log message text.
        """
        cdef bint valid_level = self._config.base_level == LogLevel.TRACE
        if self._is_running and valid_level:
            self._process_log(LogLevel.TRACE, msg.encode())

    cpdef void debug(self, str msg):
        """
        Send a debug-level log message.

        Args:
            msg (str): The log message text.
        """
        cdef bint valid_level = self._config.base_level <= LogLevel.DEBUG
        if self._is_running and valid_level:
            self._process_log(LogLevel.DEBUG, msg.encode())

    cpdef void info(self, str msg):
        """
        Send an info-level log message.

        Args:
            msg (str): The log message text.
        """
        cdef bint valid_level = self._config.base_level <= LogLevel.INFO
        if self._is_running and valid_level:
            self._process_log(LogLevel.INFO, msg.encode())

    cpdef void warning(self, str msg):
        """
        Send a warning-level log message.

        Args:
            msg (str): The log message text.
        """
        cdef bint valid_level = self._config.base_level <= LogLevel.WARNING
        if self._is_running and valid_level:
            self._process_log(LogLevel.WARNING, msg.encode())

    cpdef void error(self, str msg):
        """
        Send an error-level log message.

        Args:
            msg (str): The log message text.
        """
        cdef bint valid_level = self._config.base_level <= LogLevel.ERROR
        if self._is_running and valid_level:
            self._process_log(LogLevel.ERROR, msg.encode())

    cpdef void critical(self, str msg):
        """
        Send a critical-level log message.

        Args:
            msg (str): The log message text.
        """
        cdef bint valid_level = self._config.base_level <= LogLevel.CRITICAL
        if self._is_running and valid_level:
            self._process_log(LogLevel.CRITICAL, msg.encode())

    cpdef void shutdown(self):
        """
        Flush any remaining messages and shuts down the worker logger.

        This method drains both the log buffer and the data buffer, sending any 
        remaining items to the master logger, and then stops the connection.

        Warning:
            After calling `shutdown()`, this logger should not be used again.
        """
        if not self._is_running:
            return
        
        self.warning("Shutting down worker logger; flushing buffers and stopping connection")

        self._is_running = False
        
        cdef:
            list raw_log_buffer = self._log_buffer.acquire_all()
            list raw_data_buffer = self._data_buffer.acquire_all()
        
        if raw_log_buffer:
            self._logs_dump_to_queue_callback(raw_log_buffer)

        if raw_data_buffer:
            self._data_dump_to_queue_callback(raw_data_buffer)
            
        self._master_conn.stop()

    cpdef bint is_running(self):
        """
        Check if the master logger is running.
        """
        return self._is_running
    
    cpdef str get_name(self):
        """
        Get the name of the master logger.
        """
        return self._name.decode()

    cpdef LoggerConfig get_config(self):
        """
        Get the configuration of the master logger.
        """
        return self._config