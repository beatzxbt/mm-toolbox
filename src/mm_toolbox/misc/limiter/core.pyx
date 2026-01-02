"""
Token bucket rate limiter with optional per-second sub-buckets.

Model
-----
- A single token bucket of size `capacity` that refills every `window_s` seconds.
- Optionally, the window is subdivided into `window_s` one-second buckets, each
  with a share of the capacity, capping instantaneous burst usage evenly.
- Threshold policy annotates state (NORMAL/WARNING/BLOCKED) based on utilization.
- Burst policy (optional) permits limited overage within the active second.

This module is single-threaded by design. No explicit locking is used and no
nogil sections are required. Time alignment is based on the current wall clock
at creation/refill time; per-second buckets align to second boundaries.
"""
from mm_toolbox.time.time cimport time_ms

from libc.stdlib cimport malloc, free
from libc.stdint cimport int64_t as i64
from libc.string cimport memset
from msgspec import Struct

from .types cimport RateLimitState, SubEventTokenState, EventTokenState
from .config import (
    RateLimiterConfig,
    RateLimitBurstConfig,
    RateLimitStateConfig,
    SubBucketStrategy,
)

# Python-visible result
class ConsumeResult(Struct):
    """Result of a consumption attempt."""
    allowed: bool
    state: RateLimitState
    remaining: int
    usage: float


cdef EventTokenState make_event_token_state(i64 capacity, i64 window_s, object strategy):
    """Create and initialize an event token state.

    The window is optionally subdivided into per-second buckets depending on
    `strategy`. Capacity is distributed exactly across buckets so that the sum
    of sub-bucket allocations equals `capacity`.
    """
    cdef i64 now = time_ms()
    cdef EventTokenState state
    memset(<void*>&state, 0, sizeof(EventTokenState))

    state.allocated_tokens = capacity
    state.used_tokens = 0
    state.prev_refill_time_ms = now
    state.next_refill_time_ms = now + window_s * 1000
    state.burst_attempts_used = 0

    # Strategy: DISABLED -> 1 bucket; PER_SECOND -> window_s buckets
    cdef i64 bucket_count = 1
    if strategy is not None and strategy == SubBucketStrategy.PER_SECOND:
        bucket_count = window_s if window_s > 0 else 1
    if bucket_count < 1:
        bucket_count = 1

    state.num_sub_event_states = bucket_count
    state.sub_event_states = <SubEventTokenState*> malloc(sizeof(SubEventTokenState) * bucket_count)
    if state.sub_event_states != NULL:
        cdef i64 i
        cdef i64 q = 0
        cdef i64 r = 0
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


cdef class RateLimiter:
    """Token-bucket limiter with optional per-second sub-buckets.

    The limiter divides time into fixed windows of length `window_s`. Each window
    has `capacity` tokens. Optionally, it caps per-second usage by splitting the
    window into equally allocated per-second buckets. A threshold policy marks
    high utilization, and an optional burst policy allows limited overage.
    """

    def __cinit__(self, RateLimiterConfig config):
        """Construct a limiter with a given configuration."""
        self._config = config
        self._state = make_event_token_state(
            config.capacity, config.window_s, config.sub_bucket_strategy
        )

    def __dealloc__(self):
        if self._state.sub_event_states != NULL:
            free(self._state.sub_event_states)
            self._state.sub_event_states = NULL

    cdef inline i64 _active_sub_index(self, i64 now):
        if self._state.num_sub_event_states <= 1:
            return 0
        cdef i64 elapsed = now - self._state.prev_refill_time_ms
        if elapsed < 0:
            elapsed = 0
        return (elapsed // 1000) % self._state.num_sub_event_states

    cdef inline void _refresh_sub_bucket(self, i64 now, i64 idx):
        if self._state.num_sub_event_states <= 0 or self._state.sub_event_states == NULL:
            return
        if self._state.sub_event_states[idx].next_refill_time_ms <= now:
            # Align to current second boundary
            cdef i64 start = (now // 1000) * 1000
            self._state.sub_event_states[idx].prev_refill_time_ms = start
            self._state.sub_event_states[idx].next_refill_time_ms = start + 1000
            self._state.sub_event_states[idx].used_tokens = 0

    cdef void _reset_state(self, i64 now):
        """Reset overall and per-second buckets to the start of a new window."""
        self._state.allocated_tokens = self._config.capacity
        self._state.used_tokens = 0
        self._state.burst_attempts_used = 0
        self._state.prev_refill_time_ms = now
        self._state.next_refill_time_ms = now + self._config.window_s * 1000

        if self._state.num_sub_event_states > 0 and self._state.sub_event_states != NULL:
            cdef i64 n = self._state.num_sub_event_states
            cdef i64 i
            for i in range(n):
                self._state.sub_event_states[i].used_tokens = 0
                if n == 1:
                    self._state.sub_event_states[i].prev_refill_time_ms = now
                else:
                    self._state.sub_event_states[i].prev_refill_time_ms = now + i * 1000
                self._state.sub_event_states[i].next_refill_time_ms = self._state.sub_event_states[i].prev_refill_time_ms + 1000

    cdef void _maybe_refill(self):
        cdef i64 now = time_ms()
        cdef bint time_to_refill = now >= self._state.next_refill_time_ms

        if not time_to_refill:
            return

        self._reset_state(now)

    cpdef void refill(self):
        """Force a refill cycle immediately."""
        cdef i64 now = time_ms()
        self._reset_state(now)

    cdef inline EventTokenState get_state(self):
        return self._state

    cpdef object try_consume(self, bint force=False):
        """Consume a single token and return a ConsumeResult.

        Args:
            force: When True, bypass checks and allow consumption, returning OVERRIDE.

        Returns:
            ConsumeResult: Allowed flag, state, remaining, and usage.
        """
        self._maybe_refill()
        return self.try_consume_multiple(1, force)

    cpdef object try_consume_multiple(self, i64 num_tokens, bint force=False):
        """Consume multiple tokens and return a ConsumeResult.

        Args:
            num_tokens: Token count to consume.
            force: When True, bypass checks and allow consumption, returning OVERRIDE.

        Returns:
            ConsumeResult: Allowed flag, state, remaining, and usage.
        """
        self._maybe_refill()

        if num_tokens <= 0:
            return ConsumeResult(
                allowed=True,
                state=RateLimitState.NORMAL,
                remaining=<int>(self._state.allocated_tokens - self._state.used_tokens),
                usage=float(
                    1.0 if self._state.allocated_tokens <= 0
                    else (<double>self._state.used_tokens / <double>self._state.allocated_tokens)
                ),
            )

        if force:
            # Apply usage without enforcing capacity, annotate as OVERRIDE.
            cdef i64 now_force = time_ms()
            cdef bint sub_enabled_force = self._state.num_sub_event_states > 0 and self._state.sub_event_states != NULL
            cdef i64 idx_force = 0
            if sub_enabled_force:
                idx_force = self._active_sub_index(now_force)
                self._refresh_sub_bucket(now_force, idx_force)
                self._state.sub_event_states[idx_force].used_tokens += num_tokens
            self._state.used_tokens += num_tokens
            return ConsumeResult(
                allowed=True,
                state=RateLimitState.OVERRIDE,
                remaining=<int>(self._state.allocated_tokens - self._state.used_tokens),
                usage=float(
                    1.0 if self._state.allocated_tokens <= 0
                    else (<double>self._state.used_tokens / <double>self._state.allocated_tokens)
                ),
            )

        cdef i64 capacity = self._state.allocated_tokens
        cdef i64 new_used = self._state.used_tokens + num_tokens

        cdef bint sub_enabled = self._state.num_sub_event_states > 0 and self._state.sub_event_states != NULL
        cdef i64 now = time_ms()
        cdef i64 idx = 0
        cdef i64 sub_new_used = 0
        cdef bint sub_ok = 1
        if sub_enabled:
            idx = self._active_sub_index(now)
            self._refresh_sub_bucket(now, idx)
            sub_new_used = self._state.sub_event_states[idx].used_tokens + num_tokens
            sub_ok = sub_new_used <= self._state.sub_event_states[idx].allocated_tokens

        cdef bint overall_ok = new_used <= capacity
        if overall_ok and sub_ok:
            cdef double usage = <double>new_used / <double>capacity
            if self._config.state_config.is_enabled and usage > self._config.state_config.block_threshold:
                return ConsumeResult(
                    allowed=False,
                    state=RateLimitState.BLOCKED,
                    remaining=<int>(capacity - self._state.used_tokens),
                    usage=float(<double>self._state.used_tokens / <double>capacity),
                )
            # apply
            self._state.used_tokens = new_used
            if sub_enabled:
                self._state.sub_event_states[idx].used_tokens = sub_new_used
            if self._config.state_config.is_enabled and usage > self._config.state_config.warning_threshold:
                return ConsumeResult(
                    allowed=True,
                    state=RateLimitState.WARNING,
                    remaining=<int>(capacity - self._state.used_tokens),
                    usage=float(<double>self._state.used_tokens / <double>capacity),
                )
            return ConsumeResult(
                allowed=True,
                state=RateLimitState.NORMAL,
                remaining=<int>(capacity - self._state.used_tokens),
                usage=float(<double>self._state.used_tokens / <double>capacity),
            )

        # Handle burst allowance
        if self._config.burst_config.is_enabled and sub_enabled:
            if num_tokens > self._config.burst_config.max_tokens:
                return ConsumeResult(
                    allowed=False,
                    state=RateLimitState.BLOCKED,
                    remaining=<int>(capacity - self._state.used_tokens),
                    usage=float(<double>self._state.used_tokens / <double>capacity),
                )
            if self._state.burst_attempts_used < self._config.burst_config.max_burst_attempts:
                self._state.burst_attempts_used += 1
                # Cap usage at capacity/sub-capacity
                self._state.used_tokens = new_used if new_used <= capacity else capacity
                if sub_enabled:
                    self._state.sub_event_states[idx].used_tokens = (
                        sub_new_used if sub_new_used <= self._state.sub_event_states[idx].allocated_tokens
                        else self._state.sub_event_states[idx].allocated_tokens
                    )
                return ConsumeResult(
                    allowed=True,
                    state=RateLimitState.NORMAL,
                    remaining=<int>(capacity - self._state.used_tokens),
                    usage=float(<double>self._state.used_tokens / <double>capacity),
                )
            return ConsumeResult(
                allowed=False,
                state=RateLimitState.WARNING,
                remaining=<int>(capacity - self._state.used_tokens),
                usage=float(<double>self._state.used_tokens / <double>capacity),
            )

        return ConsumeResult(
            allowed=False,
            state=RateLimitState.BLOCKED,
            remaining=<int>(capacity - self._state.used_tokens),
            usage=float(<double>self._state.used_tokens / <double>capacity),
        )

    cpdef i64 tokens_remaining(self):
        """Return remaining tokens in the overall bucket."""
        self._maybe_refill()
        return self._state.allocated_tokens - self._state.used_tokens

    cpdef double usage(self):
        """Return the fraction of tokens used in the overall bucket."""
        self._maybe_refill()
        if self._state.allocated_tokens <= 0:
            return 1.0
        return <double>self._state.used_tokens / <double>self._state.allocated_tokens

    @classmethod
    def per_window(cls, int capacity, int window_s):
        """Create a limiter with given capacity and window size (seconds)."""
        return cls(RateLimiterConfig.default(capacity=capacity, window_s=window_s))

    @classmethod
    def per_second(cls, int capacity):
        """Create a per-second limiter."""
        return cls(RateLimiterConfig.default(capacity=capacity, window_s=1))

    @classmethod
    def per_minute(cls, int capacity):
        """Create a per-minute limiter."""
        return cls(RateLimiterConfig.default(capacity=capacity, window_s=60))
