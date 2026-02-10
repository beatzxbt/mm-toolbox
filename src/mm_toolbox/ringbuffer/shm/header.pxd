from libc.stdint cimport uint64_t as u64


cdef extern from "c/shm_types.h":
    ctypedef struct ShmHeader:
        u64 magic
        u64 capacity
        u64 mask
        u64 write_pos
        u64 read_pos
        u64 msg_count
        u64 latest_insert_time_ns
        u64 latest_consume_time_ns
