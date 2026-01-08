from typing import Self

from msgspec import Struct
from enum import StrEnum


class SubBucketStrategy(StrEnum):
    """How per-second buckets apply within the window."""

    DISABLED = "disabled"
    PER_SECOND = "per_second"


class RateLimitStateConfig(Struct):
    """State thresholds for warning and blocking."""

    is_enabled: bool
    warning_threshold: float
    block_threshold: float

    def __post_init__(self):
        """Validate that thresholds are in (0, 1) and ordered."""
        if not (0.0 < self.warning_threshold < 1.0):
            raise ValueError("warning_threshold must be in (0, 1)")
        if not (0.0 < self.block_threshold < 1.0):
            raise ValueError("block_threshold must be in (0, 1)")
        if self.warning_threshold >= self.block_threshold:
            raise ValueError("warning_threshold must be less than block_threshold")

    @classmethod
    def default(cls) -> Self:
        """Return enabled thresholds: warn at 75%, block at 95%."""
        return cls(
            is_enabled=True,
            warning_threshold=0.75,
            block_threshold=0.95,
        )


class RateLimitBurstConfig(Struct):
    """Optional burst allowance control."""

    is_enabled: bool
    max_tokens: int
    max_burst_attempts: int

    def __post_init__(self):
        """Validate burst settings when enabled."""
        if self.is_enabled:
            if self.max_tokens <= 0:
                raise ValueError(
                    "Invalid max_tokens; must be greater than 0 when burst handling is enabled"
                )
            if self.max_burst_attempts <= 0:
                raise ValueError(
                    "Invalid max_burst_attempts; must be greater than 0 when burst handling is enabled"
                )

    @classmethod
    def default(cls) -> Self:
        """Return disabled burst handling."""
        return cls(
            is_enabled=False,
            max_tokens=0,
            max_burst_attempts=0,
        )


class RateLimiterConfig(Struct):
    """Top-level configuration for a token-bucket limiter with optional sub-second buckets."""

    capacity: int
    window_s: int
    state_config: RateLimitStateConfig
    burst_config: RateLimitBurstConfig
    sub_bucket_strategy: SubBucketStrategy

    def __post_init__(self):
        """Validate core capacities and duration."""
        if self.capacity <= 0:
            raise ValueError("Invalid capacity; must be greater than 0")
        if self.window_s <= 0:
            raise ValueError("Invalid window_s; must be greater than 0")

    @classmethod
    def default(cls, capacity: int, window_s: int) -> Self:
        """Return config with default state, burst settings, and per-second bucketing."""
        return cls(
            capacity=capacity,
            window_s=window_s,
            state_config=RateLimitStateConfig.default(),
            burst_config=RateLimitBurstConfig.default(),
            sub_bucket_strategy=SubBucketStrategy.PER_SECOND,
        )
