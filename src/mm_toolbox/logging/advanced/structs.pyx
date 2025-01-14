import msgspec
from typing import Union

from libc.stdint cimport uint16_t, UINT16_MAX
from mm_toolbox.time.time cimport time_s

# Future optimization is converting these msgspec.Struct structs
# to native Cython structs. Current penalty is in hundreds of nanos
# in the buffer due to object mode overhead. 
#
# Use binary packing so _buffer goes from list -> C array, will 
# put individual append speeds into sub 50ns.
class LogMessage(msgspec.Struct, tag=True):
    time: int
    level: int
    msg: bytes

class LogMessageBatch(msgspec.Struct, tag=True):
    system: dict[str, str]
    srcfilename: bytes
    time: int
    size: int
    data: list[LogMessage]

# Special types for handling data recordings. These MUST always be 
# msgspec.Structs, but can be either be written to DataMessage or 
# used in their native struct (eg OrderbookUpdate, TradeUpdate, etc). 
#
# In general, the batches here will be much larger than the logs 
# buffer, to save on time blocked by encoding the messages. 
class DataMessage(msgspec.Struct, tag=True):
    time: int
    msg: msgspec.Struct

class DataMessageBatch(msgspec.Struct, tag=True):
    # system: dict[str, str]
    # srcfilename: bytes
    time: int
    size: int
    data: list[DataMessage]

cdef class MessageBuffer:
    """
    A fixed-size buffer of LogMessage structs for efficient batch storage and encoding.
    """

    def __init__(self, object dump_to_queue_callback, uint16_t capacity=UINT16_MAX, double timeout_s=1.0):
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
        cdef uint16_t _size = self._size
        self._size = 0
        return self._buffer[:_size]
