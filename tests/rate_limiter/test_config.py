"""Configuration boundaries and documented default policies."""

from typing import cast

import pytest

from mm_toolbox.rate_limiter import (
    RateLimitBurstConfig,
    RateLimiter,
    RateLimiterConfig,
    RateLimitStateConfig,
    SubBucketStrategy,
)


def test_default_policies() -> None:
    config = RateLimiterConfig.default(capacity=5, window_s=2)
    assert config.state_config == RateLimitStateConfig(True, 0.75, 0.95)
    assert config.burst_config == RateLimitBurstConfig(False, 0, 0)
    assert config.sub_bucket_strategy is SubBucketStrategy.PER_SECOND


@pytest.mark.parametrize(
    "warning,block",
    [
        (0.0, 0.5),
        (1.0, 0.5),
        (-0.1, 0.5),
        (0.5, 0.0),
        (0.5, 1.0),
        (0.5, 1.1),
        (0.75, 0.75),
        (0.8, 0.7),
        (float("nan"), 0.9),
        (0.5, float("nan")),
        (float("inf"), 0.9),
        (0.5, -float("inf")),
    ],
)
def test_invalid_thresholds(warning: float, block: float) -> None:
    with pytest.raises(ValueError):
        RateLimitStateConfig(True, warning, block)


@pytest.mark.parametrize("max_tokens,max_attempts", [(0, 1), (1, 0), (-1, 2), (2, -1)])
def test_enabled_burst_requires_positive_limits(
    max_tokens: int, max_attempts: int
) -> None:
    with pytest.raises(ValueError):
        RateLimitBurstConfig(True, max_tokens, max_attempts)


@pytest.mark.parametrize("capacity,window_s", [(0, 1), (-1, 1), (1, 0), (1, -2)])
def test_invalid_capacity_or_window(capacity: int, window_s: int) -> None:
    with pytest.raises(ValueError):
        RateLimiterConfig.default(capacity, window_s)


@pytest.mark.parametrize("invalid", [None, {}, object()])
def test_native_constructor_rejects_wrong_config_type(invalid: object) -> None:
    # This test deliberately bypasses static typing at the public native boundary.
    with pytest.raises(TypeError, match="config must be RateLimiterConfig"):
        RateLimiter(cast(RateLimiterConfig, invalid))
