"""Sub-bucket management declarations."""
from libc.stdint cimport int64_t as i64

from .types cimport EventTokenState


cdef i64 active_sub_index(EventTokenState* state, i64 now)
cdef void refresh_sub_bucket(EventTokenState* state, i64 now, i64 idx)
cdef void reset_state(EventTokenState* state, object config, i64 now)
