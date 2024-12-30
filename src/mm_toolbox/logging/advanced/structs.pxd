import msgspec
from libc.stdint cimport uint32_t

cdef enum LogLevel:
    LL_TRACE = 1
    LL_DEBUG = 2
    LL_INFO = 3
    LL_WARNING = 4
    LL_ERROR = 5
    LL_CRITICAL = 6

# Future optimization is converting these msgspec.Struct structs
# to native cython structs. Current penalty is in hundreds of nanos
# in the buffer due to object mode overhead. 
#
# Use binary packing so _buffer goes from list -> C array, will 
# put individual append speeds into sub 50ns.
class LogMessage(msgspec.Struct):
    time: int
    level: int
    msg: bytes

class LogMessageBatch(msgspec.Struct):
    system: dict[str, str]
    srcfilename: bytes
    time: int
    size: int
    msgs: list[LogMessage]

# Special types for handling data recordings. These MUST always be 
# msgspec.Structs, but can be either be written to DataMessage or 
# used in their native struct (eg OrderbookUpdate, TradeUpdate, etc). 
#
# In general, the batches here will be much larger than the logs 
# buffer, to save on time blocked by encoding the messages. 
class DataMessage(msgspec.Struct):
    time: int
    data: msgspec.Struct

class DataMessageBatch(msgspec.Struct):
    # system: dict[str, str]
    # srcfilename: bytes
    time: int
    size: int
    data: list[DataMessage]

cdef class MessageBuffer:
    cdef:
        uint32_t        _capacity
        uint32_t        _size
        double          _timeout_s
        double          _start_time
        list            _buffer
        object          _dump_to_queue_callback

    # def __init__(self, callable dump_to_queue_callback, uint32_t capacity=UINT32_MAX, double timeout_s=2.5)
    cdef inline bint    _is_full(self)
    cdef void           append(self, object msg) # msg: msgspec.Struct
    cdef list           acquire_all(self) # return: list[msgspec.Struct]