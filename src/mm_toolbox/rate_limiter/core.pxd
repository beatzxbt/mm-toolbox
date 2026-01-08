from libc.stdint cimport int64_t as i64
from .types cimport EventTokenState


cdef class RateLimiter:
    cdef:
        object _config
        EventTokenState _state

    cdef inline i64         _active_sub_index(self, i64 now)
    cdef inline void        _refresh_sub_bucket(self, i64 now, i64 idx)
    cdef void               _maybe_refill(self)
    cdef void               _reset_state(self, i64 now)
    cdef inline EventTokenState get_state(self)
    cpdef object            try_consume(self, bint force=*)
    cpdef object            try_consume_multiple(self, i64 num_tokens, bint force=*)
    cpdef void              refill(self)
    cpdef i64               tokens_remaining(self)
    cpdef double            usage(self)

