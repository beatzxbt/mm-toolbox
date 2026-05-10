"""Rate limiter class declarations."""
from libc.stdint cimport int64_t as i64


cdef class RateLimiter:
    cdef:
        object _config
        i64    _capacity
        i64    _window_ms
        bint   _sub_enabled
        i64    _num_sub_buckets
        double _warn_threshold
        double _block_threshold
        bint   _state_enabled
        bint   _burst_enabled
        i64    _max_burst_tokens
        i64    _max_burst_attempts
        i64    _used_tokens
        i64    _window_start_ms
        i64    _burst_used
        i64*   _sub_allocations
        i64*   _sub_used

    cdef inline void _maybe_refill(self, i64 now)
    cdef inline i64 _sub_index(self, i64 now)
    cpdef void refill(self)
    cpdef object try_consume(self, bint force=*)
    cpdef object try_consume_multiple(self, i64 num_tokens, bint force=*)
    cpdef i64 tokens_remaining(self)
    cpdef double usage(self)
