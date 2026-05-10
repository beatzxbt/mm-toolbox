"""
Token bucket rate limiter with optional per-second sub-buckets.

Model
-----
- A single token bucket of size ``capacity`` that refills every ``window_s``
  seconds.
- Optionally, the window is subdivided into ``window_s`` one-second buckets,
  each with a share of the capacity, capping instantaneous burst usage evenly.
- Threshold policy annotates state (NORMAL / WARNING / BLOCKED) based on
  utilization.
- Burst policy (optional) permits limited overage within the active window.

This module is **single-threaded by design**. No explicit locking is used.
Time alignment is based on monotonic time; per-second buckets are indexed
relative to the start of the current window.
"""
from __future__ import annotations

from mm_toolbox.time.time cimport time_monotonic_ms

from libc.stdlib cimport malloc, free
from libc.stdint cimport int64_t as i64

from .result import RateLimitState, ConsumeResult
from .config import RateLimiterConfig, SubBucketStrategy


cdef class RateLimiter:
    """Token-bucket limiter with optional per-second sub-buckets.

    The limiter divides time into fixed windows of length ``window_s``. Each
    window has ``capacity`` tokens. Optionally, it caps per-second usage by
    splitting the window into equally allocated per-second buckets. A threshold
    policy marks high utilization, and an optional burst policy allows limited
    overage.

    This class is **not thread-safe**. It must be used from a single thread
    or protected by external synchronization.
    """

    def __cinit__(self, object config):
        """Construct a limiter with a given configuration.

        Args:
            config: :class:`RateLimiterConfig` specifying capacity, window, and
                policies.

        Raises:
            TypeError: If *config* is not a :class:`RateLimiterConfig`.
        """
        cdef i64 n
        cdef i64 q
        cdef i64 r
        cdef i64 i

        if not isinstance(config, RateLimiterConfig):
            raise TypeError(
                f"config must be RateLimiterConfig, got {type(config).__name__}"
            )

        self._config = config
        self._capacity = config.capacity
        self._window_ms = config.window_s * 1000
        self._sub_enabled = (
            config.sub_bucket_strategy == SubBucketStrategy.PER_SECOND
        )
        self._num_sub_buckets = config.window_s if self._sub_enabled else 1
        self._warn_threshold = config.state_config.warning_threshold
        self._block_threshold = config.state_config.block_threshold
        self._state_enabled = config.state_config.is_enabled
        self._burst_enabled = config.burst_config.is_enabled
        self._max_burst_tokens = config.burst_config.max_tokens
        self._max_burst_attempts = config.burst_config.max_burst_attempts

        self._used_tokens = 0
        self._window_start_ms = time_monotonic_ms()
        self._burst_used = 0

        self._sub_allocations = NULL
        self._sub_used = NULL

        n = self._num_sub_buckets
        self._sub_allocations = <i64*>malloc(sizeof(i64) * n)
        self._sub_used = <i64*>malloc(sizeof(i64) * n)

        if self._sub_allocations == NULL or self._sub_used == NULL:
            raise MemoryError("Failed to allocate sub-bucket arrays")

        q = self._capacity // n
        r = self._capacity - q * n
        for i in range(n):
            self._sub_allocations[i] = q + (1 if i < r else 0)
            self._sub_used[i] = 0

    def __dealloc__(self):
        """Free allocated sub-bucket memory."""
        if self._sub_allocations != NULL:
            free(self._sub_allocations)
            self._sub_allocations = NULL
        if self._sub_used != NULL:
            free(self._sub_used)
            self._sub_used = NULL

    cdef inline void _maybe_refill(self, i64 now):
        """Trigger a refill if the overall window has expired.

        Args:
            now: Current monotonic time in milliseconds.
        """
        cdef i64 i
        if now - self._window_start_ms >= self._window_ms:
            self._window_start_ms = now
            self._used_tokens = 0
            self._burst_used = 0
            for i in range(self._num_sub_buckets):
                self._sub_used[i] = 0

    cdef inline i64 _sub_index(self, i64 now):
        """Calculate the active per-second bucket index.

        Args:
            now: Current monotonic time in milliseconds.

        Returns:
            Index of the currently active sub-bucket (0 if sub-buckets are
            disabled).
        """
        cdef i64 elapsed
        if not self._sub_enabled:
            return 0
        elapsed = now - self._window_start_ms
        if elapsed < 0:
            elapsed = 0
        return (elapsed // 1000) % self._num_sub_buckets

    cpdef void refill(self):
        """Force a refill cycle immediately."""
        cdef i64 now
        cdef i64 i
        now = time_monotonic_ms()
        self._window_start_ms = now
        self._used_tokens = 0
        self._burst_used = 0
        for i in range(self._num_sub_buckets):
            self._sub_used[i] = 0

    cpdef object try_consume(self, bint force=False):
        """Consume a single token and return a :class:`ConsumeResult`.

        Args:
            force: When ``True``, bypass checks and allow consumption,
                returning :attr:`RateLimitState.OVERRIDE`.

        Returns:
            ConsumeResult: Allowed flag, state, remaining, and usage.
        """
        return self.try_consume_multiple(1, force)

    cpdef object try_consume_multiple(self, i64 num_tokens, bint force=False):
        """Consume multiple tokens and return a :class:`ConsumeResult`.

        Args:
            num_tokens: Token count to consume. Values less than or equal to
                zero return the current state without consuming any tokens,
                which can be used as a lightweight state query.
            force: When ``True``, bypass checks and allow consumption,
                returning :attr:`RateLimitState.OVERRIDE`.

        Returns:
            ConsumeResult: Allowed flag, state, remaining, and usage.
        """
        cdef i64 now
        cdef i64 remaining
        cdef double usage
        cdef i64 new_used
        cdef bint overall_ok
        cdef bint sub_ok
        cdef i64 sub_idx
        cdef i64 sub_new_used
        cdef double new_usage
        cdef double post_usage
        cdef object state

        now = time_monotonic_ms()
        self._maybe_refill(now)

        remaining = self._capacity - self._used_tokens
        usage = (
            1.0
            if self._capacity <= 0
            else (<double>self._used_tokens / <double>self._capacity)
        )

        if num_tokens <= 0:
            return ConsumeResult(
                allowed=True,
                state=RateLimitState.NORMAL,
                remaining=<int>remaining,
                usage=usage,
            )

        if force:
            self._used_tokens += num_tokens
            if self._sub_enabled:
                self._sub_used[self._sub_index(now)] += num_tokens
            return ConsumeResult(
                allowed=True,
                state=RateLimitState.OVERRIDE,
                remaining=<int>(self._capacity - self._used_tokens),
                usage=(
                    1.0
                    if self._capacity <= 0
                    else (
                        <double>self._used_tokens / <double>self._capacity
                    )
                ),
            )

        new_used = self._used_tokens + num_tokens
        overall_ok = new_used <= self._capacity
        sub_ok = True
        sub_idx = 0
        sub_new_used = 0

        if self._sub_enabled:
            sub_idx = self._sub_index(now)
            sub_new_used = self._sub_used[sub_idx] + num_tokens
            sub_ok = sub_new_used <= self._sub_allocations[sub_idx]

        if overall_ok and sub_ok:
            new_usage = <double>new_used / <double>self._capacity
            if self._state_enabled and new_usage > self._block_threshold:
                return ConsumeResult(
                    allowed=False,
                    state=RateLimitState.BLOCKED,
                    remaining=<int>remaining,
                    usage=usage,
                )
            self._used_tokens = new_used
            if self._sub_enabled:
                self._sub_used[sub_idx] = sub_new_used
            post_usage = (
                <double>self._used_tokens / <double>self._capacity
            )
            state = RateLimitState.NORMAL
            if self._state_enabled and post_usage > self._warn_threshold:
                state = RateLimitState.WARNING
            return ConsumeResult(
                allowed=True,
                state=state,
                remaining=<int>(self._capacity - self._used_tokens),
                usage=post_usage,
            )

        if self._burst_enabled:
            if num_tokens > self._max_burst_tokens:
                return ConsumeResult(
                    allowed=False,
                    state=RateLimitState.BLOCKED,
                    remaining=<int>remaining,
                    usage=usage,
                )
            if self._burst_used < self._max_burst_attempts:
                self._burst_used += 1
                self._used_tokens = (
                    new_used if new_used <= self._capacity else self._capacity
                )
                if self._sub_enabled:
                    self._sub_used[sub_idx] = (
                        sub_new_used
                        if sub_new_used <= self._sub_allocations[sub_idx]
                        else self._sub_allocations[sub_idx]
                    )
                return ConsumeResult(
                    allowed=True,
                    state=RateLimitState.NORMAL,
                    remaining=<int>(self._capacity - self._used_tokens),
                    usage=(
                        1.0
                        if self._capacity <= 0
                        else (
                            <double>self._used_tokens
                            / <double>self._capacity
                        )
                    ),
                )
            return ConsumeResult(
                allowed=False,
                state=RateLimitState.WARNING,
                remaining=<int>remaining,
                usage=usage,
            )

        return ConsumeResult(
            allowed=False,
            state=RateLimitState.BLOCKED,
            remaining=<int>remaining,
            usage=usage,
        )

    cpdef i64 tokens_remaining(self):
        """Return remaining tokens in the overall bucket."""
        cdef i64 now
        now = time_monotonic_ms()
        self._maybe_refill(now)
        return self._capacity - self._used_tokens

    cpdef double usage(self):
        """Return the fraction of tokens used in the overall bucket."""
        cdef i64 now
        now = time_monotonic_ms()
        self._maybe_refill(now)
        if self._capacity <= 0:
            return 1.0
        return <double>self._used_tokens / <double>self._capacity

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
