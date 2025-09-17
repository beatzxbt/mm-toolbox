from libc.stdint cimport uint64_t as u64, uint32_t as u32

cdef enum CLogLevel:
    TRACE
    DEBUG
    INFO
    WARNING
    ERROR

cdef struct CLog:
    u64 timestamp_ns
    u32 name_len
    char* name
    CLogLevel clevel
    u32 message_len
    char* message