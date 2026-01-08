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
from __future__ import annotations

from mm_toolbox.time.time cimport time_ms

from libc.stdlib cimport free
from libc.stdint cimport int64_t as i64

from .types cimport EventTokenState
from .result import RateLimitState, ConsumeResult
from .state cimport make_event_token_state
from .bucket cimport active_sub_index, refresh_sub_bucket, reset_state
from .config import RateLimiterConfig


cdef class RateLimiter:
    """Token-bucket limiter with optional per-second sub-buckets.

    The limiter divides time into fixed windows of length `window_s`. Each window
    has `capacity` tokens. Optionally, it caps per-second usage by splitting the
    window into equally allocated per-second buckets. A threshold policy marks
    high utilization, and an optional burst policy allows limited overage.
    """

    def __cinit__(self, object config):
        """Construct a limiter with a given configuration.

        Args:
            config: RateLimiterConfig specifying capacity, window, and policies.
        """
        self._config = config
        self._state = make_event_token_state(
            config.capacity, config.window_s, config.sub_bucket_strategy
        )

    def __dealloc__(self):
        """Free allocated sub-bucket memory."""
        if self._state.sub_event_states != NULL:
            free(self._state.sub_event_states)
            self._state.sub_event_states = NULL

    cdef void _maybe_refill(self):
        """Trigger a refill if the overall window has expired."""
        cdef i64 now = time_ms()
        cdef bint time_to_refill = now >= self._state.next_refill_time_ms

        if not time_to_refill:
            return

        reset_state(&self._state, self._config, now)

    cdef inline EventTokenState get_state(self):
        """Return the internal state structure (for testing/debugging)."""
        return self._state

    cpdef void refill(self):
        """Force a refill cycle immediately."""
        cdef i64 now = time_ms()
        reset_state(&self._state, self._config, now)

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
        cdef i64 now_force = 0
        cdef bint sub_enabled_force = 0
        cdef i64 idx_force = 0
        cdef i64 capacity = 0
        cdef i64 new_used = 0
        cdef bint sub_enabled = 0
        cdef i64 now = 0
        cdef i64 idx = 0
        cdef i64 sub_new_used = 0
        cdef bint sub_ok = 1
        cdef bint overall_ok = 0
        cdef double usage = 0.0

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
            # Apply usage without enforcing capacity, annotate as OVERRIDE
            now_force = time_ms()
            sub_enabled_force = (
                self._state.num_sub_event_states > 0
                and self._state.sub_event_states != NULL
            )
            if sub_enabled_force:
                idx_force = active_sub_index(&self._state, now_force)
                refresh_sub_bucket(&self._state, now_force, idx_force)
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

        capacity = self._state.allocated_tokens
        new_used = self._state.used_tokens + num_tokens

        sub_enabled = (
            self._state.num_sub_event_states > 0
            and self._state.sub_event_states != NULL
        )
        now = time_ms()
        if sub_enabled:
            idx = active_sub_index(&self._state, now)
            refresh_sub_bucket(&self._state, now, idx)
            sub_new_used = self._state.sub_event_states[idx].used_tokens + num_tokens
            sub_ok = sub_new_used <= self._state.sub_event_states[idx].allocated_tokens

        overall_ok = new_used <= capacity
        if overall_ok and sub_ok:
            usage = <double>new_used / <double>capacity
            if self._config.state_config.is_enabled and usage > self._config.state_config.block_threshold:
                return ConsumeResult(
                    allowed=False,
                    state=RateLimitState.BLOCKED,
                    remaining=<int>(capacity - self._state.used_tokens),
                    usage=float(<double>self._state.used_tokens / <double>capacity),
                )
            # Apply consumption
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
        """Create a limiter with given capacity and window size (seconds).

        Args:
            capacity: Total token capacity.
            window_s: Window duration in seconds.

        Returns:
            RateLimiter: Configured limiter instance.
        """
        return cls(RateLimiterConfig.default(capacity=capacity, window_s=window_s))

    @classmethod
    def per_second(cls, int capacity):
        """Create a per-second limiter.

        Args:
            capacity: Token capacity per second.

        Returns:
            RateLimiter: Configured limiter instance.
        """
        return cls(RateLimiterConfig.default(capacity=capacity, window_s=1))

    @classmethod
    def per_minute(cls, int capacity):
        """Create a per-minute limiter.

        Args:
            capacity: Token capacity per minute.

        Returns:
            RateLimiter: Configured limiter instance.
        """
        return cls(RateLimiterConfig.default(capacity=capacity, window_s=60))
