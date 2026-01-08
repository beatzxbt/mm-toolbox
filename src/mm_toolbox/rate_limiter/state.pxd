"""Event token state factory declarations."""
from libc.stdint cimport int64_t as i64

from .types cimport EventTokenState


cdef EventTokenState make_event_token_state(i64 capacity, i64 window_s, object strategy)
