"""Rate limiter class declarations."""
from libc.stdint cimport int64_t as i64

from .types cimport EventTokenState


cdef class RateLimiter:
    cdef:
        object _config
        EventTokenState _state

    cdef void _maybe_refill(self)
    cdef inline EventTokenState get_state(self)
    cpdef void refill(self)
    cpdef object try_consume(self, bint force=*)
    cpdef object try_consume_multiple(self, i64 num_tokens, bint force=*)
    cpdef i64 tokens_remaining(self)
    cpdef double usage(self)
