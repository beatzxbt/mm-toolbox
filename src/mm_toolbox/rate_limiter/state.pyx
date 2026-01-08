"""
Event token state factory.

Provides initialization logic for creating EventTokenState structures with
optional per-second sub-buckets.
"""
from __future__ import annotations

from mm_toolbox.time.time cimport time_ms

from libc.stdlib cimport malloc
from libc.stdint cimport int64_t as i64
from libc.string cimport memset

from .types cimport SubEventTokenState, EventTokenState
from .config import SubBucketStrategy


cdef EventTokenState make_event_token_state(i64 capacity, i64 window_s, object strategy):
    """Create and initialize an event token state.

    The window is optionally subdivided into per-second buckets depending on
    `strategy`. Capacity is distributed exactly across buckets so that the sum
    of sub-bucket allocations equals `capacity`.

    Args:
        capacity: Total token capacity for the window.
        window_s: Window duration in seconds.
        strategy: SubBucketStrategy determining bucket subdivision.

    Returns:
        Initialized EventTokenState structure.
    """
    cdef i64 now = time_ms()
    cdef EventTokenState state
    memset(<void*>&state, 0, sizeof(EventTokenState))

    state.allocated_tokens = capacity
    state.used_tokens = 0
    state.prev_refill_time_ms = now
    state.next_refill_time_ms = now + window_s * 1000
    state.burst_attempts_used = 0

    cdef i64 bucket_count = 1
    cdef i64 i = 0
    cdef i64 q = 0
    cdef i64 r = 0

    # Strategy: DISABLED -> 1 bucket; PER_SECOND -> window_s buckets
    if strategy is not None and strategy == SubBucketStrategy.PER_SECOND:
        bucket_count = window_s if window_s > 0 else 1
    if bucket_count < 1:
        bucket_count = 1

    state.num_sub_event_states = bucket_count
    state.sub_event_states = <SubEventTokenState*> malloc(sizeof(SubEventTokenState) * bucket_count)
    if state.sub_event_states != NULL:
        q = capacity // bucket_count
        r = capacity - q * bucket_count
        for i in range(bucket_count):
            # Distribute the remainder to the first r buckets
            state.sub_event_states[i].allocated_tokens = q + (1 if i < r else 0)
            state.sub_event_states[i].used_tokens = 0
            state.sub_event_states[i].prev_refill_time_ms = now if bucket_count == 1 else now + i * 1000
            state.sub_event_states[i].next_refill_time_ms = state.sub_event_states[i].prev_refill_time_ms + 1000
    else:
        # Allocation failed; degrade gracefully to no sub-events
        state.num_sub_event_states = 0

    return state
