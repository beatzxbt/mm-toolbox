"""
Rate limiter result types.

Defines the RateLimitState enum for consumption outcomes and the ConsumeResult
struct returned by consumption attempts.
"""
from __future__ import annotations

from msgspec import Struct


cpdef enum RateLimitState:
    """State of rate limiter after a consumption attempt."""
    NORMAL
    WARNING
    BLOCKED
    OVERRIDE


class ConsumeResult(Struct):
    """Result of a token consumption attempt.

    Attributes:
        allowed: Whether the consumption was permitted.
        state: The rate limiter state after the attempt.
        remaining: Tokens remaining in the overall bucket.
        usage: Fraction of capacity used (0.0 to 1.0+).
    """

    allowed: bool
    state: RateLimitState
    remaining: int
    usage: float
