from libc.stdint cimport uint64_t as u64

cdef extern from "c/shm_types.h":
    ctypedef struct ShmHeader:
        u64 magic
        u64 capacity
        u64 mask
        u64 write_pos
        u64 read_pos
        u64 producer_claim
        u64 latest_insert_time_ns
        u64 latest_consume_time_ns
        u64 dropped_count
        u64 reserved[7]

    ctypedef struct ShmMpscGlobalHeader:
        u64 magic
        u64 num_rings
        u64 ring_capacity
        u64 next_producer_ring
        u64 reserved[4]

    ctypedef ShmHeader ShmSubRingHeader
