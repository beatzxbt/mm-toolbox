import msgspec
from typing import Union, T

from libc.stdint cimport uint8_t
from mm_toolbox.time.time cimport time_s

cpdef enum LogLevel:
    TRACE = 0x0
    DEBUG = 0x1
    INFO = 0x2
    WARNING = 0x3
    ERROR = 0x4
    CRITICAL = 0x5

# Future optimization is converting these msgspec.Struct structs
# to native Cython structs. Current penalty is in hundreds of nanos
# in the buffer due to object mode overhead. 
#
# Use binary packing so _buffer goes from list -> C array, will 
# put individual append speeds into sub 50ns.
class LogMessage(msgspec.Struct, kw_only=True, tag=True, gc=False):
    time: Union[int, float]
    level: int
    msg: LogMessage

class LogMessageBatch(msgspec.Struct, kw_only=True, tag=True, gc=False):
    system: dict[str, str]
    name: bytes
    time: Union[int, float]
    size: int
    data: list[LogMessage]

# Special types for handling data recordings. These MUST always be 
# msgspec.Structs, but can be either be written to DataMessage or 
# used in their native struct (eg OrderbookUpdate, TradeUpdate, etc). 
#
# In general, the batches here will be much larger than the logs 
# buffer, to save on time blocked by encoding the messages. 
class DataMessage(msgspec.Struct, kw_only=True, tag=True, gc=False):
    time: Union[int, float]
    msg: msgspec.Struct

class DataMessageBatch(msgspec.Struct, kw_only=True, tag=True, gc=False):
    system: dict[str, str]
    name: bytes
    time: Union[int, float]
    size: int

    # We cant have msgspec.Struct here as it fails to correctly deserialize
    # in the master logger to the original struct type. So it is kept as 
    # type T temporarily.
    data: list[T]

# Useful later to map LogLevel's back to their readable form 
# without the use of log_level_to_str().
LogLevelMap: dict[int, str] = {
    0: "TRACE",
    1: "DEBUG",
    2: "INFO",
    3: "WARNING",
    4: "ERROR",
    5: "CRITICAL"
}

cpdef str log_level_to_str(uint8_t level):
    """
    Convert a LogLevel enum value to its corresponding string representation.

    Rather than using LogLevelMap, this function is faster as comparing 
    int<>int in Cython compiles to switch-case statements under the hood.

    Args:
        level (int): The LogLevel enum value to convert.

    Returns:
        str: The string representation of the LogLevel enum value.
    """
    if level == 0:
        return "TRACE"
    elif level == 1:
        return "DEBUG"
    elif level == 2:
        return "INFO"
    elif level == 3:
        return "WARNING"
    elif level == 4:
        return "ERROR"
    elif level == 5:
        return "CRITICAL"
    else:
        raise ValueError(f"Invalid LogLevel; expected [{', '.join(LogLevelMap.values())}] but got {level}")

cdef class MessageBuffer:
    """
    A fixed-size buffer of LogMessage structs for efficient batch storage and encoding.
    """

    def __init__(
        self, 
        object dump_to_queue_callback, 
        Py_ssize_t capacity=1000, 
        double timeout_s=1.0
    ):
        """
        Initialize the MessageBuffer with a given capacity and optional timeout.

        Args:
            dump_to_queue_callback (callable): The callback function that handles dumping messages to a queue.
            capacity (int): The maximum number of LogMessage entries this buffer can hold.
            timeout_s (float, optional): The maximum number of seconds that can elapse before
                the buffer is considered full, even if not at capacity. If 0, no timeout is applied.
        """
        self._capacity = capacity
        self._size = 0
        self._timeout_s = timeout_s
        self._start_time = time_s()

        # Defaulting to 1000 LogMessage structs is quite overkill, though 
        # it's a safe default option incase of high log rates.
        self._buffer = [None] * self._capacity 

        self._dump_to_queue_callback = dump_to_queue_callback

    cdef inline bint _is_full(self):
        """
        Check whether the buffer is currently full due to capacity and timeout.

        Returns:
            bool: True if the buffer has reached capacity and the timeout has expired;
                  False otherwise.
        """
        cdef bint expired = (time_s() - self._start_time) >= self._timeout_s
        cdef bint full = self._size == self._capacity
        return expired or full

    cdef void append(self, object msg):
        """
        Append a single LogMessage to the buffer.

        Args:
            msg (LogMessage): A fully-initialized LogMessage struct to store in the buffer.

        Raises:
            IndexError: If the buffer is full and cannot accept more messages.
        """
        if self._size == 0:
            self._start_time = time_s()

        self._buffer[self._size] = msg
        self._size += 1

        if self._is_full():
            self._dump_to_queue_callback(self._buffer[:self._size])
            self._size = 0

    cdef list acquire_all(self):
        """
        Retrieve and clear all messages currently stored in the buffer.

        Returns:
            list: The list of LogMessage objects that were in the buffer before clearing.
        """
        cdef list buffer = self._buffer[:self._size]
        self._size = 0
        return buffer