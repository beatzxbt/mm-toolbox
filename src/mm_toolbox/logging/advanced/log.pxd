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
    unsigned char* name
    CLogLevel clevel
    u32 message_len
    unsigned char* message