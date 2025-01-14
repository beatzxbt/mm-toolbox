import zmq
import asyncio
import msgspec
from typing import Union

from libc.stdint cimport uint8_t, uint16_t

from mm_toolbox.time.time cimport time_ns

from .handlers.base import LogHandler
from ..utils.system import _get_system_info
from ..utils.zmq import ZmqConnection

from .config cimport LoggerConfig
from .structs import (
    LogMessage, 
    LogMessageBatch, 
    DataMessage, 
    DataMessageBatch
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
        LoggerConfig config=LoggerConfig(), 
        str srcfilename="",
        list log_handlers=[],  
        list data_handlers=[], 
    ):
        for handler in log_handlers + data_handlers:
            handler_base_class = handler.__class__.__base__
            if not handler_base_class == LogHandler:
                raise TypeError(f"Invalid handler base class; expected LogHandler but got {handler_base_class}")
        
        self._log_handlers = log_handlers
        self._data_handlers = data_handlers

        self._system_info = _get_system_info(
            machine=True, 
            network=True, 
            op_sys=True
        )
        self._srcfilename = srcfilename.encode()
        
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
            self._logs_dump_to_queue_callback,
            timeout_s=1.0,
        )
        self._data_buffer = MessageBuffer(
            self._data_dump_to_queue_callback,
            timeout_s=2.0,
        )

        self._master_conn.listen(self._process_worker_msg)

        self._is_running = False

    cpdef void _process_worker_msg(self, bytes msg):
        """
        Asynchronous loop that continuously receives multipart messages
        and decodes them into LogMessage structs.
        """
        cdef object decoded_msg = self._batch_decoder.decode(msg)
        cdef object decoded_msg_type = type(decoded_msg)

        # We could directly serialize these into JSON and 
        # forward the bytes only to the handlers, but in 
        # cases it may easier to work with struct directly
        # incase of non-JSON type requirements.
        if decoded_msg_type == LogMessageBatch:
            for handler in self._log_handlers:
                handler.push(decoded_msg)
        elif decoded_msg_type == DataMessageBatch:
            for handler in self._data_handlers:
                handler.push(decoded_msg)
        else:
            raise TypeError(f"Invalid MessageBatch type; expected [LogMessageBatch, DataMessageBatch] but got {decoded_msg_type}")
                
    cdef void _logs_dump_to_queue_callback(self, list raw_log_buffer):
        """
        Internal callback to push batched log messages to the master logger.

        Args:
            raw_log_buffer (list): A list of LogMessage objects to be encoded and sent.
        """
        cdef uint16_t log_buffer_size = len(raw_log_buffer)

        cdef object log_batch = LogMessageBatch(
            system=self._system_info,
            srcfilename=self._srcfilename,
            time=time_ns(),
            size=log_buffer_size,
            msgs=raw_log_buffer
        ) 

        for handler in self._log_handlers:
            handler.push(log_batch)

    cdef void _data_dump_to_queue_callback(self, list raw_data_buffer):
        """
        Internal callback to push batched data messages to the master logger.

        Args:
            raw_data_buffer (list): A list of data objects (DataMessage or otherwise) 
                to be encoded and sent.
        """
        cdef uint16_t data_buffer_size = len(raw_data_buffer)

        cdef object data_batch = DataMessageBatch(
            time=time_ns(),
            size=data_buffer_size,
            data=raw_data_buffer
        ) 

        for handler in self._data_handlers:
            handler.push(data_batch)

    cdef inline void _process_log(self, uint8_t level, bytes msg):
        """
        Process a single log message by creating a LogMessage struct and buffering it.

        Args:
            level (uint8_t): The log level, e.g. LogLevel.INFO.
            msg (bytes): The log message text, encoded as bytes.
        """   
        cdef object log_msg_struct = LogMessage(
            time=time_ns(),
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
        data_msg_struct = DataMessage(
            time=time_ns(),
            data=msg
        )
        self._data_buffer.append(data_msg_struct)

    cpdef void data(self, object data):
        """
        Enqueue data to be sent to the master logger.

        Args:
            data (object): A msgspec.Struct representing the data.

        Raises:
            TypeError: If data is not a msgspec.Struct.
        """
        cdef object data_class_type = type(data).__class__

        if data_class_type == msgspec.Struct.__class__:
            self._process_data(data)
        else:
            raise TypeError(
                f"Invalid data type; expected msgspec.Struct but got '{data_class_type}'"
            )

    cpdef void trace(self, bytes msg):
        """
        Send a trace-level log message.

        Args:
            msg (bytes): The log message text, encoded as bytes.
        """
        self._process_log(LogLevel.TRACE, msg)

    cpdef void debug(self, bytes msg):
        """
        Send a debug-level log message.

        Args:
            msg (bytes): The log message text, encoded as bytes.
        """
        self._process_log(LogLevel.DEBUG, msg)

    cpdef void info(self, bytes msg):
        """
        Send an info-level log message.

        Args:
            msg (bytes): The log message text, encoded as bytes.
        """
        self._process_log(LogLevel.INFO, msg)

    cpdef void warning(self, bytes msg):
        """
        Send a warning-level log message.

        Args:
            msg (bytes): The log message text, encoded as bytes.
        """
        self._process_log(LogLevel.WARNING, msg)

    cpdef void error(self, bytes msg):
        """
        Send an error-level log message.

        Args:
            msg (bytes): The log message text, encoded as bytes.
        """
        self._process_log(LogLevel.ERROR, msg)

    cpdef void critical(self, bytes msg):
        """
        Send a critical-level log message.

        Args:
            msg (bytes): The log message text, encoded as bytes.
        """
        self._process_log(LogLevel.CRITICAL, msg)

    cpdef void close(self):
        """
        Flush any remaining messages and close the worker logger.

        This method drains both the log buffer and the data buffer, sending any 
        remaining items to the master logger, and then stops the connection.

        Warning:
            After calling `close()`, this logger should not be used again.
        """
        cdef:
            list raw_log_buffer = self._log_buffer.acquire_all()
            list raw_data_buffer = self._data_buffer.acquire_all()
        
        if len(raw_log_buffer) > 0:
            self._logs_dump_to_queue_callback(raw_log_buffer)

        if len(raw_data_buffer) > 0:
            self._data_dump_to_queue_callback(raw_data_buffer)

        self._master_conn.cancel_listening()