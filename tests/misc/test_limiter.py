"""Tests for rate limiter utilities."""

from __future__ import annotations

import time

import pytest

from mm_toolbox.misc.limiter import (
    RateLimitBurstConfig,
    RateLimiter,
    RateLimiterConfig,
    RateLimitState,
    RateLimitStateConfig,
    SubBucketStrategy,
)


def _avoid_second_boundary(buffer_s: float = 0.05) -> None:
    """Sleep briefly to avoid crossing a second boundary mid-test."""
    if time.time() % 1.0 > 1.0 - buffer_s:
        time.sleep(buffer_s * 2.0)


def make_limiter(
    capacity: int,
    window_s: int,
    *,
    warn: float = 0.75,
    block: float = 0.95,
    state_enabled: bool = True,
    burst_enabled: bool = False,
    max_tokens: int = 0,
    max_attempts: int = 0,
    sub_bucket_strategy: SubBucketStrategy = SubBucketStrategy.PER_SECOND,
) -> RateLimiter:
    """Create a limiter with explicit configuration controls."""
    state_cfg = RateLimitStateConfig(
        is_enabled=state_enabled,
        warning_threshold=warn,
        block_threshold=block,
    )
    burst_cfg = RateLimitBurstConfig(
        is_enabled=burst_enabled,
        max_tokens=max_tokens,
        max_burst_attempts=max_attempts,
    )
    cfg = RateLimiterConfig(
        capacity=capacity,
        window_s=window_s,
        state_config=state_cfg,
        burst_config=burst_cfg,
        sub_bucket_strategy=sub_bucket_strategy,
    )
    return RateLimiter(cfg)


class TestRateLimitStateConfig:
    """Test RateLimitStateConfig validation and defaults."""

    def test_default_thresholds(self) -> None:
        """Default thresholds are enabled and ordered."""
        cfg = RateLimitStateConfig.default()
        assert cfg.is_enabled is True
        assert cfg.warning_threshold == 0.75
        assert cfg.block_threshold == 0.95

    @pytest.mark.parametrize(
        "warning_threshold, block_threshold",
        [
            (0.0, 0.5),
            (1.0, 0.5),
            (0.5, 0.0),
            (0.5, 1.0),
            (0.8, 0.7),
        ],
    )
    def test_invalid_thresholds(
        self, warning_threshold: float, block_threshold: float
    ) -> None:
        """Thresholds must be in (0, 1) and strictly increasing."""
        with pytest.raises(ValueError):
            RateLimitStateConfig(
                is_enabled=True,
                warning_threshold=warning_threshold,
                block_threshold=block_threshold,
            )


class TestRateLimitBurstConfig:
    """Test RateLimitBurstConfig validation and defaults."""

    def test_default_config(self) -> None:
        """Default burst config is disabled."""
        cfg = RateLimitBurstConfig.default()
        assert cfg.is_enabled is False
        assert cfg.max_tokens == 0
        assert cfg.max_burst_attempts == 0

    @pytest.mark.parametrize("max_tokens, max_attempts", [(0, 1), (1, 0), (-1, 2)])
    def test_invalid_enabled_settings(
        self, max_tokens: int, max_attempts: int
    ) -> None:
        """Enabled burst config requires positive limits."""
        with pytest.raises(ValueError):
            RateLimitBurstConfig(
                is_enabled=True,
                max_tokens=max_tokens,
                max_burst_attempts=max_attempts,
            )

    def test_valid_enabled_settings(self) -> None:
        """Enabled burst config accepts positive limits."""
        cfg = RateLimitBurstConfig(
            is_enabled=True,
            max_tokens=2,
            max_burst_attempts=1,
        )
        assert cfg.is_enabled is True
        assert cfg.max_tokens == 2
        assert cfg.max_burst_attempts == 1


class TestRateLimiterConfig:
    """Test RateLimiterConfig validation and defaults."""

    def test_default_config(self) -> None:
        """Default config wires state and burst policies."""
        cfg = RateLimiterConfig.default(capacity=5, window_s=2)
        assert cfg.capacity == 5
        assert cfg.window_s == 2
        assert cfg.state_config.is_enabled is True
        assert cfg.burst_config.is_enabled is False
        assert cfg.sub_bucket_strategy is SubBucketStrategy.PER_SECOND

    @pytest.mark.parametrize("capacity, window_s", [(0, 1), (-1, 1), (1, 0), (1, -2)])
    def test_invalid_capacity_or_window(self, capacity: int, window_s: int) -> None:
        """Capacity and window duration must be positive."""
        with pytest.raises(ValueError):
            RateLimiterConfig.default(capacity=capacity, window_s=window_s)


class TestRateLimiterBasicOperations:
    """Test core consume and accounting behavior."""

    def test_basic_consumption_and_usage(self) -> None:
        """Consuming tokens updates remaining count and usage."""
        rl = make_limiter(
            capacity=4,
            window_s=1,
            state_enabled=False,
            sub_bucket_strategy=SubBucketStrategy.DISABLED,
        )

        first = rl.try_consume()
        assert first.allowed is True
        assert first.state == RateLimitState.NORMAL
        assert first.remaining == 3
        assert first.usage == pytest.approx(0.25)

        no_change = rl.try_consume_multiple(0)
        assert no_change.allowed is True
        assert no_change.state == RateLimitState.NORMAL
        assert no_change.remaining == 3
        assert no_change.usage == pytest.approx(0.25)

        negative = rl.try_consume_multiple(-2)
        assert negative.allowed is True
        assert negative.remaining == 3
        assert negative.usage == pytest.approx(0.25)

        assert rl.tokens_remaining() == 3
        assert rl.usage() == pytest.approx(0.25)

    def test_force_consumption_over_capacity(self) -> None:
        """Force mode bypasses checks and marks OVERRIDE."""
        rl = make_limiter(
            capacity=2,
            window_s=1,
            state_enabled=False,
            sub_bucket_strategy=SubBucketStrategy.DISABLED,
        )

        forced = rl.try_consume_multiple(3, force=True)
        assert forced.allowed is True
        assert forced.state == RateLimitState.OVERRIDE
        assert forced.remaining == -1
        assert forced.usage == pytest.approx(1.5)
        assert rl.tokens_remaining() == -1


class TestRateLimiterThresholds:
    """Test warning and blocking thresholds."""

    def test_warning_and_block_transitions(self) -> None:
        """Warning triggers at > warn and block triggers at > block."""
        rl = make_limiter(
            capacity=4,
            window_s=1,
            warn=0.5,
            block=0.75,
            state_enabled=True,
            sub_bucket_strategy=SubBucketStrategy.DISABLED,
        )

        normal = rl.try_consume_multiple(2)
        assert normal.allowed is True
        assert normal.state == RateLimitState.NORMAL
        assert normal.usage == pytest.approx(0.5)

        warning = rl.try_consume()
        assert warning.allowed is True
        assert warning.state == RateLimitState.WARNING
        assert warning.usage == pytest.approx(0.75)

        blocked = rl.try_consume()
        assert blocked.allowed is False
        assert blocked.state == RateLimitState.BLOCKED
        assert blocked.remaining == 1
        assert blocked.usage == pytest.approx(0.75)
        assert rl.tokens_remaining() == 1


class TestRateLimiterSubBuckets:
    """Test per-second sub-bucket behavior."""

    def test_sub_bucket_limits_same_second(self) -> None:
        """Sub-buckets cap per-second usage even when capacity remains."""
        _avoid_second_boundary()
        rl = make_limiter(
            capacity=4,
            window_s=2,
            state_enabled=False,
            sub_bucket_strategy=SubBucketStrategy.PER_SECOND,
        )

        allowed = rl.try_consume_multiple(2)
        assert allowed.allowed is True
        assert allowed.remaining == 2

        blocked = rl.try_consume()
        assert blocked.allowed is False
        assert blocked.state == RateLimitState.BLOCKED
        assert blocked.remaining == 2

        time.sleep(1.05)
        next_bucket = rl.try_consume_multiple(2)
        assert next_bucket.allowed is True
        assert next_bucket.remaining == 0


class TestRateLimiterBurst:
    """Test burst allowance behavior."""

    def test_burst_allows_limited_overage(self) -> None:
        """Burst allows one extra attempt within configured limits."""
        rl = make_limiter(
            capacity=2,
            window_s=2,
            state_enabled=False,
            burst_enabled=True,
            max_tokens=2,
            max_attempts=1,
            sub_bucket_strategy=SubBucketStrategy.PER_SECOND,
        )

        burst_ok = rl.try_consume_multiple(2)
        assert burst_ok.allowed is True
        assert burst_ok.state == RateLimitState.NORMAL
        assert burst_ok.remaining == 0

        burst_exhausted = rl.try_consume()
        assert burst_exhausted.allowed is False
        assert burst_exhausted.state == RateLimitState.WARNING
        assert burst_exhausted.remaining == 0


class TestRateLimiterFactories:
    """Test RateLimiter factory constructors."""

    def test_factory_constructors(self) -> None:
        """Factory methods return functional limiters with expected capacity."""
        per_second = RateLimiter.per_second(3)
        per_minute = RateLimiter.per_minute(4)
        per_window = RateLimiter.per_window(5, 2)

        assert per_second.tokens_remaining() == 3
        assert per_minute.tokens_remaining() == 4
        assert per_window.tokens_remaining() == 5


class TestRateLimiterIntegration:
    """Test refill behavior in a realistic loop."""

    def test_refill_resets_after_window(self, wait_for) -> None:
        """Tokens are restored after the window elapses."""
        rl = make_limiter(
            capacity=2,
            window_s=1,
            state_enabled=False,
            sub_bucket_strategy=SubBucketStrategy.DISABLED,
        )

        consumed = rl.try_consume_multiple(2)
        assert consumed.allowed is True
        assert rl.tokens_remaining() == 0

        assert wait_for(lambda: rl.tokens_remaining() == 2, timeout_s=2.0)
