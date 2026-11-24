"""Native limiter using a manually controlled monotonic clock."""

from libc.stdint cimport int64_t as i64

from mm_toolbox.rate_limiter.limiter cimport RateLimiter


cdef class ClockedRateLimiter(RateLimiter):
    cdef public i64 now_ms

    def __init__(self, config):
        # The base C constructor runs before this subclass's virtual clock is active.
        self.refill()

    cdef i64 _now_ms(self) noexcept:
        return self.now_ms
