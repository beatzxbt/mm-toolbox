from __future__ import annotations

from enum import Enum

class RateLimitState(Enum):
    """State of rate limiter after a consumption attempt."""

    NORMAL: int
    WARNING: int
    BLOCKED: int
    OVERRIDE: int

class ConsumeResult:
    """Result of a token consumption attempt."""

    allowed: bool
    state: RateLimitState
    remaining: int
    usage: float
