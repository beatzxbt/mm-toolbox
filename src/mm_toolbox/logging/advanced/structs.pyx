from enum import IntEnum
from libc.stdint cimport (
    uint8_t as u8,
    uint32_t as u32,
    uint64_t as u64,
)
from libc.string cimport memcpy

from mm_toolbox.time.time cimport time_ns
from mm_toolbox.logging.advanced.structs cimport (
    BufMsgType,
    CLogLevel,
)

cdef int clog_level_to_int(CLogLevel clog_level):
    """Maps a CLogLevel to an int."""
    if clog_level == CLogLevel.TRACE:
        return 0
    elif clog_level == CLogLevel.DEBUG:
        return 1
    elif clog_level == CLogLevel.INFO:
        return 2
    elif clog_level == CLogLevel.WARNING:
        return 3
    elif clog_level == CLogLevel.ERROR:
        return 4

cdef bytes int_to_log_name(int log_level):
    """Maps an int (derived from CLogLevel) to its corresponding log name"""
    if log_level == 0:
        return b"TRACE"
    elif log_level == 1:
        return b"DEBUG"
    elif log_level == 2:
        return b"INFO"
    elif log_level == 3:
        return b"WARNING"
    elif log_level == 4:
        return b"ERROR"

cdef class LogBatch:
    def __cinit__(self, bytes name):
        self.name = name
        self.name_len = len(name)

        # If you somehow hit the 1024 ** 2 log limit in 1s, god help you
        self.num_logs_in_batch = 0
        self.log_batch = [b""] * 1024 ** 2

    cdef void add_log(self, CLogLevel level, bytes msg):
        """
        Format: {headers: {time: u64, level: u8, msg_len: u32}, msg: bytes}

        0:7 = Time in nanoseconds
        8 = Level
        9:12 = Message length
        13:(Message length) = Message
        """ 
        cdef: 
            u64 time = time_ns()
            u32 msg_len = len(msg)
            bytes buffer = bytes(sizeof(u64) + sizeof(u8) + sizeof(u32) + msg_len)

        memcpy(buffer, &time, sizeof(u64))
        memcpy(buffer + sizeof(u64), &level, sizeof(u8))
        memcpy(buffer + sizeof(u64) + sizeof(u8), &msg_len, sizeof(u32))
        memcpy(buffer + sizeof(u64) + sizeof(u8) + sizeof(u32), msg, len(msg))

        self.log_batch[self.num_logs_in_batch] = buffer
        self.num_logs_in_batch += 1

    cdef bytes to_bytes(self, bint reset=True):
        """
        Format: {headers: {buf_msg_type: u8, len_name: u8, name: bytes, num_logs: u32}, logs: bytes[]}

        Args:
            reset (bool): Whether to reset the number of logs in the batch.
        """
        cdef:
            BufMsgType buf_msg_type = BufMsgType.LOG
            u32        num_logs = self.num_logs_in_batch
        
        # Calculate total size of all logs
        cdef u32 i, total_logs_size = 0
        for i in range(num_logs):
            total_logs_size += len(self.log_batch[i])
        
        # Header size: buf_msg_type + len_name + name + num_logs
        cdef u32 header_size = sizeof(u8) + sizeof(u8) + self.name_len + sizeof(u32)
        cdef bytes buffer = bytes(header_size + total_logs_size)

        # Write header
        memcpy(buffer, &buf_msg_type, sizeof(u8))
        memcpy(buffer + sizeof(u8), &self.name_len, sizeof(u8))
        memcpy(buffer + sizeof(u8) + sizeof(u8), self.name, self.name_len)
        memcpy(buffer + sizeof(u8) + sizeof(u8) + self.name_len, &num_logs, sizeof(u32))
        
        # Write logs
        cdef u32 pos = header_size
        for i in range(num_logs):
            cdef bytes log_data = self.log_batch[i]
            memcpy(buffer + pos, log_data, len(log_data))
            pos += len(log_data)
            
        if reset:
            self.num_logs_in_batch = 0

        return buffer

    cdef list[tuple] get_all_logs(self, bint reset=True):
        """
        For use in the MasterLogger's own generated logs.

        Logs returned in the format: [(time_ns: int, level: LogLevel, msg: bytes)]

        Args:
            reset (bool): Whether to reset the number of logs in the batch.
        """
        cdef list[tuple] logs = []

        # Individual log components
        cdef:
            bytes log_data
            u64 time
            u8 level
            u32 msg_len
            bytes msg

        cdef u32 i, pos
        for i in range(self.num_logs_in_batch):
            log_data = self.log_batch[i]
            pos = 0

            # Extract time (u64)
            memcpy(&time, log_data + pos, sizeof(u64))
            pos += sizeof(u64)
            
            # Extract level (u8)
            memcpy(&level, log_data + pos, sizeof(u8))
            pos += sizeof(u8)
            
            # Extract message length (u32)
            memcpy(&msg_len, log_data + pos, sizeof(u32))
            pos += sizeof(u32)
            
            # Extract message (bytes)
            msg = log_data[pos:pos + msg_len]
            pos += msg_len

            logs.append((time, int_to_log_name(level), msg))

        if reset:
            self.num_logs_in_batch = 0

        return logs

cdef tuple[bytes, list[tuple]] log_batch_from_bytes(bytes buffer):
    """
    Buffer must be the exact output given by .to_bytes()

    Logs returned in the format: (name: bytes, [(time_ns: int, level: LogLevel, msg: bytes)])

    This is only ever called within the MasterLogger, so speed is not a priority.
    """
    # Name 
    cdef u8 len_name = buffer[1]
    cdef bytes name = buffer[2:2 + len_name]

    # Batch of logs
    cdef list[tuple] logs = []
    cdef u32 num_logs
    memcpy(&num_logs, buffer + 2 + len_name, sizeof(u32))

    # Individual log components
    cdef:
        u64 time
        u8  level
        u32 msg_len
        bytes msg

    cdef u32 pos = 2 + len_name + sizeof(u32)
    cdef u32 buffer_len = len(buffer)
    
    for i in range(num_logs):
        # Check bounds before reading
        if pos + sizeof(u64) + sizeof(u8) + sizeof(u32) > buffer_len:
            break
            
        # Extract time (u64)
        memcpy(&time, buffer + pos, sizeof(u64))
        pos += sizeof(u64)
        
        # Extract level (u8)
        memcpy(&level, buffer + pos, sizeof(u8))
        pos += sizeof(u8)
        
        # Extract message length (u32)
        memcpy(&msg_len, buffer + pos, sizeof(u32))
        pos += sizeof(u32)
        
        # Check bounds for message
        if pos + msg_len > buffer_len:
            break  # Avoid buffer overrun
            
        # Extract message (bytes)
        msg = buffer[pos:pos + msg_len]
        pos += msg_len

        logs.append((time, int_to_log_name(level), msg))

    return (name, logs)

cdef bytes heartbeat_to_bytes(bytes name, u64 time, u64 next_checkin_time):
    """
    {buf_msg_type: u8, len_name: u8, name: bytes, time: u64, next_checkin_time: u64}
    """
    cdef:
        const u8 buf_msg_type = BufMsgType.HEARTBEAT
        u8 len_name = len(name)

    cdef bytes buffer = bytes(sizeof(u8) + sizeof(u8) + len_name + sizeof(u64) + sizeof(u64))

    memcpy(buffer, &buf_msg_type, sizeof(u8))
    memcpy(buffer + sizeof(u8), &len_name, sizeof(u8))
    memcpy(buffer + sizeof(u8) + sizeof(u8), name, len_name)
    memcpy(buffer + sizeof(u8) + sizeof(u8) + len_name, &time, sizeof(u64))
    memcpy(buffer + sizeof(u8) + sizeof(u8) + len_name + sizeof(u64), &next_checkin_time, sizeof(u64))
    return buffer

cdef tuple[bytes, u64, u64] heartbeat_from_bytes(bytes buffer):
    """
    {buf_msg_type: u8, len_name: u8, name: bytes, time: u64, next_checkin_time: u64}

    Returns:
        tuple[bytes, u64, u64]: A tuple of (name, time, next_checkin_time)
    """
    cdef:
        u8 len_name = buffer[1]
        bytes name = buffer[2:2 + len_name]

    cdef u64 time = buffer[2 + len_name:2 + len_name + sizeof(u64)]
    cdef u64 next_checkin_time = buffer[2 + len_name + sizeof(u64):2 + len_name + sizeof(u64) + sizeof(u64)]
    return name, time, next_checkin_time
