from __future__ import annotations

from enum import Enum

from .config import RateLimiterConfig

class RateLimitState(Enum):
    NORMAL: int
    WARNING: int
    BLOCKED: int
    OVERRIDE: int

class ConsumeResult:
    allowed: bool
    state: RateLimitState
    remaining: int
    usage: float

class RateLimiter:
    def __init__(self, config: RateLimiterConfig) -> None: ...
    def refill(self) -> None: ...
    def try_consume(self, force: bool = ...) -> ConsumeResult: ...
    def try_consume_multiple(
        self, num_tokens: int, force: bool = ...
    ) -> ConsumeResult: ...
    def tokens_remaining(self) -> int: ...
    def usage(self) -> float: ...
    # Cython-only: get_state() returns EventTokenState for internal use
    # (not available from pure Python callers)
    # Static constructors
    @classmethod
    def per_second(cls, capacity: int) -> RateLimiter: ...
    @classmethod
    def per_minute(cls, capacity: int) -> RateLimiter: ...
    @classmethod
    def per_window(cls, capacity: int, window_s: int) -> RateLimiter: ...
