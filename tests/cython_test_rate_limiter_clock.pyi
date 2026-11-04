"""Native limiter with a manually controlled clock, initially zero."""

from mm_toolbox.rate_limiter.limiter import RateLimiter

class ClockedRateLimiter(RateLimiter):
    now_ms: int
