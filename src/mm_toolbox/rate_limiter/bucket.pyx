"""
Sub-bucket management operations.

Provides functions for managing per-second token buckets within a rate limiter
window, including index calculation, refresh, and state reset.
"""
from __future__ import annotations

from libc.stdint cimport int64_t as i64

from .types cimport SubEventTokenState, EventTokenState


cdef inline i64 active_sub_index(EventTokenState* state, i64 now):
    """Calculate the active per-second bucket index.

    Args:
        state: Pointer to the event token state.
        now: Current time in milliseconds.

    Returns:
        Index of the currently active sub-bucket (0 if only one bucket).
    """
    cdef i64 elapsed = 0
    if state.num_sub_event_states <= 1:
        return 0
    elapsed = now - state.prev_refill_time_ms
    if elapsed < 0:
        elapsed = 0
    return (elapsed // 1000) % state.num_sub_event_states


cdef inline void refresh_sub_bucket(EventTokenState* state, i64 now, i64 idx):
    """Refill a specific sub-bucket if its window has expired.

    Aligns the bucket to the current second boundary and resets used tokens.

    Args:
        state: Pointer to the event token state.
        now: Current time in milliseconds.
        idx: Index of the sub-bucket to refresh.
    """
    cdef i64 start = 0
    if state.num_sub_event_states <= 0 or state.sub_event_states == NULL:
        return
    if state.sub_event_states[idx].next_refill_time_ms <= now:
        # Align to current second boundary
        start = (now // 1000) * 1000
        state.sub_event_states[idx].prev_refill_time_ms = start
        state.sub_event_states[idx].next_refill_time_ms = start + 1000
        state.sub_event_states[idx].used_tokens = 0


cdef void reset_state(EventTokenState* state, object config, i64 now):
    """Reset overall and per-second buckets to the start of a new window.

    Args:
        state: Pointer to the event token state.
        config: RateLimiterConfig with capacity and window_s.
        now: Current time in milliseconds.
    """
    cdef i64 n = 0
    cdef i64 i = 0

    state.allocated_tokens = config.capacity
    state.used_tokens = 0
    state.burst_attempts_used = 0
    state.prev_refill_time_ms = now
    state.next_refill_time_ms = now + config.window_s * 1000

    if state.num_sub_event_states > 0 and state.sub_event_states != NULL:
        n = state.num_sub_event_states
        for i in range(n):
            state.sub_event_states[i].used_tokens = 0
            if n == 1:
                state.sub_event_states[i].prev_refill_time_ms = now
            else:
                state.sub_event_states[i].prev_refill_time_ms = now + i * 1000
            state.sub_event_states[i].next_refill_time_ms = state.sub_event_states[i].prev_refill_time_ms + 1000
