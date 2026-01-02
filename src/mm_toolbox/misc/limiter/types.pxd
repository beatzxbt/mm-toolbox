from libc.stdint cimport int64_t as i64

cpdef enum RateLimitState:
    NORMAL
    WARNING
    BLOCKED
    OVERRIDE

cdef struct ConsumeResult:
    bint allowed
    RateLimitState state
    i64 remaining
    double usage

cdef struct SubEventTokenState:
    i64 allocated_tokens
    i64 used_tokens
    i64 prev_refill_time_ms
    i64 next_refill_time_ms

cdef struct EventTokenState:
    i64 allocated_tokens
    i64 used_tokens
    i64 prev_refill_time_ms
    i64 next_refill_time_ms
    i64 burst_attempts_used
    i64 num_sub_event_states
    SubEventTokenState* sub_event_states


