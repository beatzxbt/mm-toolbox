import zmq
import asyncio
import msgspec
from libc.stdint cimport uint8_t, uint32_t

from mm_toolbox.time.time cimport time_ns

from ..utils._zmq import ZmqConnection
from ..utils._system import _get_system_info

from .config cimport LoggerConfig
from .structs cimport (
    LogLevel, 
    LogMessage, 
    LogMessageBatch, 
    DataMessage, 
    DataMessageBatch, 
    MessageBuffer, 
)

cdef class WorkerLogger:
    """
    A lightweight worker logger that sends log messages and data to the master logger.

    This class manages two internal buffers—one for log messages and one for data messages—
    and pushes them to the master logger when full or upon request.
    """

    def __init__(self, LoggerConfig config, str srcfilename=""):
        """
        Initialize the WorkerLogger.

        Args:
            config (LoggerConfig): Configuration details for connecting to the master logger.
            srcfilename (str, optional): The source filename or identifier to attach to log messages. 
                Defaults to an empty string.
        """
        self._system_info = _get_system_info(
            machine=True, 
            network=True, 
            op_sys=True
        )
        self._ev_loop = asyncio.get_event_loop()
        self._srcfilename = srcfilename.encode()
        
        self._log_batch_encoder = msgspec.msgpack.Encoder(LogMessageBatch)
        self._data_batch_encoder = msgspec.msgpack.Encoder(DataMessageBatch)

        # PUSH/PULL sockets are more performant for MPSC style queues.
        self._master_conn = ZmqConnection(
            socket_type=zmq.PUSH,
            path=config.path,
            mode="PUB",
        )

        self._log_buffer = MessageBuffer(self._logs_dump_to_queue_callback)
        self._data_buffer = MessageBuffer(self._data_dump_to_queue_callback)

        self._is_running = True

    cdef void _logs_dump_to_queue_callback(self, list raw_log_buffer):
        """
        Internal callback to push batched log messages to the master logger.

        Args:
            raw_log_buffer (list): A list of LogMessage objects to be encoded and sent.
        """
        cdef uint32_t log_buffer_size = len(raw_log_buffer)

        cdef object log_batch = LogMessageBatch(
            system=self._system_info,
            srcfilename=self._srcfilename,
            time=time_ns(),
            size=log_buffer_size,
            msgs=raw_log_buffer
        ) 

        cdef bytes log_batch_encoded = self._log_batch_encoder.encode(log_batch)
        self._ev_loop.create_task(self._master_conn.send(log_batch_encoded))

    cdef void _data_dump_to_queue_callback(self, list raw_data_buffer):
        """
        Internal callback to push batched data messages to the master logger.

        Args:
            raw_data_buffer (list): A list of data objects (DataMessage or otherwise) 
                to be encoded and sent.
        """
        cdef uint32_t data_buffer_size = len(raw_data_buffer)

        cdef object data_batch = DataMessageBatch(
            time=time_ns(),
            size=data_buffer_size,
            data=raw_data_buffer
        ) 

        cdef bytes data_batch_encoded = self._data_batch_encoder.encode(data_batch)
        self._ev_loop.create_task(self._master_conn.send(data_batch_encoded))

    cdef inline void _process_log(self, uint8_t level, bytes msg):
        """
        Process a single log message by creating a LogMessage struct and buffering it.

        Args:
            level (uint8_t): The log level, e.g. LogLevel.LL_INFO.
            msg (bytes): The log message text, encoded as bytes.
        """   
        cdef object log_msg_struct = LogMessage(
            time=time_ns(),
            level=level,
            msg=msg
        )
        self._log_buffer.append(log_msg_struct)

    cdef inline void _process_data(self, object data):
        """
        Process a single data item by buffering it for later batch sending.

        Args:
            data (object): The data to buffer. Often a msgspec.Struct or DataMessage.
        """
        self._data_buffer.append(data)

    cpdef void data(self, object data):
        """
        Enqueue data to be sent to the master logger.

        Args:
            data (object): Either a msgspec.Struct or a dict. If dict, it is wrapped 
                in a DataMessage object.

        Raises:
            TypeError: If data is neither a msgspec.Struct nor a dict.
        """
        cdef object data_type = type(data)

        if data_type == msgspec.Struct:
            self._process_data(data)
        elif data_type == dict:
            data_msg_struct = DataMessage(
                time=time_ns(),
                data=data
            )
            self._process_data(data_msg_struct)
        else:
            raise TypeError(
                f"Invalid data type; expected [msgspec.Struct, dict] but got '{data_type}'"
            )

    cpdef void trace(self, bytes msg):
        """
        Send a trace-level log message.

        Args:
            msg (bytes): The log message text, encoded as bytes.
        """
        self._process_log(LogLevel.LL_TRACE, msg)

    cpdef void debug(self, bytes msg):
        """
        Send a debug-level log message.

        Args:
            msg (bytes): The log message text, encoded as bytes.
        """
        self._process_log(LogLevel.LL_DEBUG, msg)

    cpdef void info(self, bytes msg):
        """
        Send an info-level log message.

        Args:
            msg (bytes): The log message text, encoded as bytes.
        """
        self._process_log(LogLevel.LL_INFO, msg)

    cpdef void warning(self, bytes msg):
        """
        Send a warning-level log message.

        Args:
            msg (bytes): The log message text, encoded as bytes.
        """
        self._process_log(LogLevel.LL_WARNING, msg)

    cpdef void error(self, bytes msg):
        """
        Send an error-level log message.

        Args:
            msg (bytes): The log message text, encoded as bytes.
        """
        self._process_log(LogLevel.LL_ERROR, msg)

    cpdef void critical(self, bytes msg):
        """
        Send a critical-level log message.

        Args:
            msg (bytes): The log message text, encoded as bytes.
        """
        self._process_log(LogLevel.LL_CRITICAL, msg)

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

        self._ev_loop.create_task(self._master_conn.stop())
