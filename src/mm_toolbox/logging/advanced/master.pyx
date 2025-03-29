import zmq
import msgspec
from typing import Union

from mm_toolbox.time.time cimport time_us

from .handlers.base import BaseLogHandler
from ..utils.system import _get_system_info
from ..utils.zmq import ZmqConnection

from .config cimport LoggerConfig
from .structs import (
    LogMessage, 
    LogMessageBatch, 
    DataMessage, 
    DataMessageBatch,
    log_level_to_str
)
from .structs cimport LogLevel, MessageBuffer


cdef class MasterLogger:
    """
    The MasterLogger acts as a central aggregator for log messages sent by worker loggers.
    It receives binary, multipart messages, decodes them, and stores them in a buffer until full.
    Once the buffer is full, it flushes them to external handlers.
    """
    def __init__(
        self, 
        LoggerConfig config=None, 
        str name="",
        list log_handlers=None,  
        list data_handlers=None, 
    ):
        self._config = config
        if self._config is None:
            self._config = LoggerConfig()

        self._log_handlers = log_handlers
        if self._log_handlers is None:
            self._log_handlers: list[BaseLogHandler] = []

        self._data_handlers = data_handlers
        if self._data_handlers is None:
            self._data_handlers: list[BaseLogHandler] = []

        # Verify that all handlers are valid 
        for handler in self._log_handlers + self._data_handlers:
            handler_base_class = handler.__class__.__base__
            if not handler_base_class == BaseLogHandler:
                raise TypeError(f"Invalid handler base class; expected BaseLogHandler but got {handler_base_class}")

            # Mainly for forwarding the str_format to the handler for formatting log messages
            # where the final point is not a code environment (eg Discord, Telegram, etc).
            handler.add_primary_config(config)

        self._system_info = _get_system_info(
            machine=True, 
            network=True, 
            op_sys=True
        )
        self._name = name.encode()

        self._batch_encoder = msgspec.msgpack.Encoder()
        self._batch_decoder = msgspec.msgpack.Decoder(Union[LogMessageBatch, DataMessageBatch])

        # PUSH/PULL sockets are more performant for MPSC style queues.
        self._master_conn = ZmqConnection(
            socket_type=zmq.PULL,
            path=config.path,
            bind=True
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
        self._master_conn.listen(self._process_worker_msg)

        self._is_running = True

        self.info(f"Master logger initialized; name: {self._name.decode()}")

    cdef inline void _ensure_running(self):
        """
        Ensure that the master logger is running. 
        If it is not, raise a RuntimeError.
        """
        if not self._is_running:
            raise RuntimeError("Master logger is not running; cannot send/recv messages")

    cpdef void _process_worker_msg(self, bytes msg):
        """
        Process received multipart messages and decode them into appropriate message structs.
        
        Args:
            msg (bytes): The raw message bytes to process.
        """
        # Do not perform actions if not running.
        if not self._is_running:
            return 
            
        try:
            decoded_msg = self._batch_decoder.decode(msg)
        except Exception as e:
            # Log the error but don't crash the master logger
            print(f"CRITICAL: Error decoding message; {e}")
            print(f"CRITICAL: Message: {msg}")
            print(f"CRITICAL: General decoder output: {msgspec.msgpack.decode(msg)}")
            return
            
        decoded_msg_type = type(decoded_msg)

        if decoded_msg_type == LogMessageBatch:
            for handler in self._log_handlers:
                handler.push(decoded_msg)
        elif decoded_msg_type == DataMessageBatch:
            for handler in self._data_handlers:
                handler.push(decoded_msg)

    cdef void _logs_dump_to_queue_callback(self, list raw_log_buffer):
        """
        Internal callback to push batched log messages to the respective 
        handlers.

        Args:
            raw_log_buffer (list[LogMessage]): A list of LogMessage objects to be encoded and sent.
        """
        # Usually the buffer will never be empty, but in .shutdown() it may be.
        # Later, explore doing this in an exception handler to avoid this check.
        if not raw_log_buffer:
            return
            
        cdef Py_ssize_t log_buffer_size = len(raw_log_buffer)

        cdef object log_batch = LogMessageBatch(
            system=self._system_info,
            name=self._name,
            time=time_us(),
            size=log_buffer_size,
            data=raw_log_buffer
        ) 

        for handler in self._log_handlers:
            handler.push(log_batch)

    cdef void _data_dump_to_queue_callback(self, list raw_data_buffer):
        """
        Internal callback to push batched data messages to the respective 
        handlers.

        Args:
            raw_data_buffer (list[DataMessage]): A list of data objects (DataMessage or otherwise) 
                to be encoded and sent.
        """
        # Usually the buffer will never be empty, but in .shutdown() it may be.
        # Later, explore doing this in an exception handler to avoid this check.
        if not raw_data_buffer:
            return
            
        cdef Py_ssize_t data_buffer_size = len(raw_data_buffer)

        cdef object data_batch = DataMessageBatch(
            system=self._system_info,
            name=self._name,
            time=time_us(),
            size=data_buffer_size,
            data=raw_data_buffer
        ) 

        for handler in self._data_handlers:
            handler.push(data_batch)

    cdef inline void _process_log(self, LogLevel level, bytes msg):
        """
        Process a single log message by creating a LogMessage struct and buffering it.

        Args:
            level (LogLevel): The log level, e.g. LogLevel.INFO.
            msg (bytes): The log message text, encoded as bytes.
        """   
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
        cdef object data_msg_struct = DataMessage(
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
        self.debug(f"Changing format string from '{self._config.str_format}' to '{format_string}'")
        self._config.str_format = format_string
        for handlers in self._log_handlers + self._data_handlers:
            handlers.add_primary_config(self._config)

    cpdef void set_log_level(self, int level):
        """
        Modify the logger's base log level at runtime.

        Args:
            level (LogLevel): The new base log level.
        """
        self.debug(f"Changing base log level from '{log_level_to_str(self._config.base_level)}' to '{log_level_to_str(level)}'")
        self._config.base_level = level
        for handlers in self._log_handlers + self._data_handlers:
            handlers.add_primary_config(self._config)

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
            After calling `shutdown()`, this logger cannot not be used again.
        """
        if not self._is_running:
            return
            
        self.warning("Shutting down master logger; flushing buffers and stopping connection")

        # No longer accept new messages from worker loggers from this point on.
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

    cpdef dict get_system_info(self):
        """
        Get the system information of the master logger.
        """
        return self._system_info