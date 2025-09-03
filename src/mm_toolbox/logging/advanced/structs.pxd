from libc.stdint cimport (
    uint8_t as u8,
    uint32_t as u32,
    uint64_t as u64,
)


cdef enum BufMsgType:
    LOG
    DATA # Reserved for future use
    HEARTBEAT

cdef enum CLogLevel:
    TRACE
    DEBUG
    INFO
    WARNING
    ERROR

cdef int    clog_level_to_int(CLogLevel clog_level)
cdef bytes  int_to_log_name(int log_level)

cdef class LogBatch:
    cdef u8             name_len
    cdef bytes          name
    cdef u32            num_logs_in_batch
    cdef list[bytes]    log_batch
    
    cdef void           add_log(self, CLogLevel level, bytes msg)
    cdef bytes          to_bytes(self, bint reset=*)
    cdef list[tuple]    get_all_logs(self, bint reset=*)

cdef tuple[bytes, list[tuple]]  log_batch_from_bytes(bytes buffer)
cdef bytes                      heartbeat_to_bytes(bytes name, u64 time, u64 next_checkin_time)
cdef tuple[bytes, u64, u64]     heartbeat_from_bytes(bytes buffer)