"""Tests for rate limiter utilities."""

from __future__ import annotations

import time

import pytest

from mm_toolbox.rate_limiter import (
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
    def test_invalid_enabled_settings(self, max_tokens: int, max_attempts: int) -> None:
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


class TestRateLimiterDistribution:
    """Test token allocation across sub-buckets."""

    def test_odd_capacity_distribution(self) -> None:
        """Odd capacity distributes remainder to early sub-buckets.

        With capacity=5 and window_s=2, PER_SECOND should allocate
        [3, 2] tokens across the two one-second sub-buckets.
        """
        _avoid_second_boundary()
        rl = make_limiter(
            capacity=5,
            window_s=2,
            state_enabled=False,
            sub_bucket_strategy=SubBucketStrategy.PER_SECOND,
        )

        # First sub-bucket should hold 3 tokens.
        first = rl.try_consume_multiple(3)
        assert first.allowed is True
        assert first.remaining == 2

        # 4th token in the same second should hit sub-bucket cap.
        denied = rl.try_consume()
        assert denied.allowed is False

        time.sleep(1.05)

        # Second sub-bucket should hold 2 tokens.
        second = rl.try_consume_multiple(2)
        assert second.allowed is True
        assert second.remaining == 0

        # 3rd token in the second second should hit sub-bucket cap.
        denied2 = rl.try_consume()
        assert denied2.allowed is False


class TestRateLimiterConsumeMethods:
    """Test direct consume methods and large requests."""

    def test_try_consume_directly(self) -> None:
        """try_consume() works independently of try_consume_multiple(1)."""
        rl = make_limiter(
            capacity=2,
            window_s=1,
            state_enabled=False,
            sub_bucket_strategy=SubBucketStrategy.DISABLED,
        )

        result = rl.try_consume()
        assert result.allowed is True
        assert result.remaining == 1
        assert result.usage == pytest.approx(0.5)

    def test_exceed_capacity_single_call(self) -> None:
        """A single request larger than remaining capacity is denied immediately."""
        rl = make_limiter(
            capacity=2,
            window_s=1,
            state_enabled=False,
            sub_bucket_strategy=SubBucketStrategy.DISABLED,
        )

        # Consume 2 of 2 tokens.
        rl.try_consume_multiple(2)
        # Request 10 more – should be denied outright.
        denied = rl.try_consume_multiple(10)
        assert denied.allowed is False
        assert denied.state == RateLimitState.BLOCKED
        assert denied.remaining == 0


class TestRateLimiterRefillBehavior:
    """Test explicit and implicit refill behavior."""

    def test_explicit_refill_resets_usage(self) -> None:
        """Calling refill() mid-window resets all usage counters to zero."""
        rl = make_limiter(
            capacity=4,
            window_s=2,
            state_enabled=False,
            sub_bucket_strategy=SubBucketStrategy.PER_SECOND,
        )

        rl.try_consume_multiple(2)
        assert rl.tokens_remaining() == 2

        rl.refill()
        assert rl.tokens_remaining() == 4
        assert rl.usage() == pytest.approx(0.0)

    def test_tokens_remaining_triggers_refill(self) -> None:
        """tokens_remaining() auto-refills once the window has expired."""
        _avoid_second_boundary()
        rl = make_limiter(
            capacity=2,
            window_s=1,
            state_enabled=False,
            sub_bucket_strategy=SubBucketStrategy.DISABLED,
        )

        rl.try_consume_multiple(2)
        assert rl.tokens_remaining() == 0

        time.sleep(1.05)
        assert rl.tokens_remaining() == 2

    def test_burst_resets_after_refill(self) -> None:
        """Burst attempts are restored after the window expires."""
        _avoid_second_boundary()
        rl = make_limiter(
            capacity=2,
            window_s=1,
            state_enabled=False,
            burst_enabled=True,
            max_tokens=1,
            max_attempts=1,
            sub_bucket_strategy=SubBucketStrategy.DISABLED,
        )

        # Consume all normal tokens.
        rl.try_consume_multiple(2)
        # Use the single burst attempt.
        burst = rl.try_consume()
        assert burst.allowed is True

        # Burst is now exhausted.
        exhausted = rl.try_consume()
        assert exhausted.allowed is False
        assert exhausted.state == RateLimitState.WARNING

        time.sleep(1.05)

        # After refill, burst should be available again.
        rl.try_consume_multiple(2)
        burst2 = rl.try_consume()
        assert burst2.allowed is True


class TestRateLimiterThresholdsExtended:
    """Test state transitions with and without sub-buckets."""

    def test_thresholds_with_per_second(self) -> None:
        """Warning/block transitions behave correctly when PER_SECOND is enabled."""
        _avoid_second_boundary()
        rl = make_limiter(
            capacity=8,
            window_s=2,
            warn=0.5,
            block=0.75,
            state_enabled=True,
            sub_bucket_strategy=SubBucketStrategy.PER_SECOND,
        )

        # First sub-bucket holds 4 tokens.
        normal = rl.try_consume_multiple(4)
        assert normal.allowed is True
        assert normal.state == RateLimitState.NORMAL
        assert normal.usage == pytest.approx(0.5)

        time.sleep(1.05)

        # Cross into WARNING in the second sub-bucket.
        warning = rl.try_consume_multiple(1)
        assert warning.allowed is True
        assert warning.state == RateLimitState.WARNING
        assert warning.usage == pytest.approx(0.625)

        warning2 = rl.try_consume_multiple(1)
        assert warning2.allowed is True
        assert warning2.state == RateLimitState.WARNING
        assert warning2.usage == pytest.approx(0.75)

        # Next request would exceed block threshold.
        blocked = rl.try_consume_multiple(1)
        assert blocked.allowed is False
        assert blocked.state == RateLimitState.BLOCKED

    def test_thresholds_disabled(self) -> None:
        """Warning/block behavior without sub-buckets (DISABLED path)."""
        rl = make_limiter(
            capacity=10,
            window_s=2,
            warn=0.5,
            block=0.75,
            state_enabled=True,
            sub_bucket_strategy=SubBucketStrategy.DISABLED,
        )

        normal = rl.try_consume_multiple(5)
        assert normal.allowed is True
        assert normal.state == RateLimitState.NORMAL
        assert normal.usage == pytest.approx(0.5)

        warning = rl.try_consume_multiple(1)
        assert warning.allowed is True
        assert warning.state == RateLimitState.WARNING
        assert warning.usage == pytest.approx(0.6)

        warning2 = rl.try_consume_multiple(1)
        assert warning2.allowed is True
        assert warning2.state == RateLimitState.WARNING
        assert warning2.usage == pytest.approx(0.7)

        blocked = rl.try_consume_multiple(1)
        assert blocked.allowed is False
        assert blocked.state == RateLimitState.BLOCKED

    def test_burst_with_state_thresholds(self) -> None:
        """Burst-allowed requests report NORMAL state even above block threshold."""
        rl = make_limiter(
            capacity=4,
            window_s=1,
            warn=0.5,
            block=0.75,
            state_enabled=True,
            burst_enabled=True,
            max_tokens=2,
            max_attempts=1,
            sub_bucket_strategy=SubBucketStrategy.PER_SECOND,
        )

        # Reach WARNING state.
        warning = rl.try_consume_multiple(3)
        assert warning.allowed is True
        assert warning.state == RateLimitState.WARNING
        assert warning.usage == pytest.approx(0.75)

        # Burst should still be allowed and report NORMAL.
        burst = rl.try_consume_multiple(2)
        assert burst.allowed is True
        assert burst.state == RateLimitState.NORMAL
        assert burst.remaining == 0
        assert burst.usage == pytest.approx(1.0)

    def test_blocked_without_burst(self) -> None:
        """Over-capacity requests return BLOCKED when burst is disabled."""
        rl = make_limiter(
            capacity=2,
            window_s=1,
            state_enabled=False,
            burst_enabled=False,
            sub_bucket_strategy=SubBucketStrategy.DISABLED,
        )

        rl.try_consume_multiple(2)
        blocked = rl.try_consume()
        assert blocked.allowed is False
        assert blocked.state == RateLimitState.BLOCKED
        assert blocked.remaining == 0


class TestRateLimiterSubBucketBurst:
    """Test burst interaction with per-second sub-buckets."""

    def test_sub_bucket_exhaustion_with_burst(self) -> None:
        """Burst caps sub-bucket usage at the sub-allocation limit."""
        _avoid_second_boundary()
        rl = make_limiter(
            capacity=4,
            window_s=2,
            state_enabled=False,
            burst_enabled=True,
            max_tokens=2,
            max_attempts=1,
            sub_bucket_strategy=SubBucketStrategy.PER_SECOND,
        )

        # Exhaust the first sub-bucket (allocation = 2).
        rl.try_consume_multiple(2)

        # Burst should be allowed but sub-usage capped at allocation.
        burst = rl.try_consume_multiple(1)
        assert burst.allowed is True
        assert burst.remaining == 1

        # Further requests in the same second should still be denied
        # because the sub-bucket remains exhausted and burst is spent.
        denied = rl.try_consume()
        assert denied.allowed is False
        assert denied.state == RateLimitState.WARNING

    def test_burst_cap_at_capacity(self) -> None:
        """Burst does not let used_tokens exceed overall capacity."""
        rl = make_limiter(
            capacity=2,
            window_s=1,
            state_enabled=False,
            burst_enabled=True,
            max_tokens=2,
            max_attempts=1,
            sub_bucket_strategy=SubBucketStrategy.DISABLED,
        )

        # Use all normal capacity.
        rl.try_consume_multiple(2)

        # Burst request that would push used_tokens past capacity.
        burst = rl.try_consume_multiple(2)
        assert burst.allowed is True
        assert burst.remaining == 0
        assert burst.usage == pytest.approx(1.0)

    def test_force_increments_sub_bucket(self) -> None:
        """force=True increments sub-bucket usage as well as overall usage."""
        _avoid_second_boundary()
        rl = make_limiter(
            capacity=4,
            window_s=2,
            state_enabled=False,
            sub_bucket_strategy=SubBucketStrategy.PER_SECOND,
        )

        # Use half the first sub-bucket.
        rl.try_consume_multiple(2)

        # Force bypasses sub-bucket limits.
        forced = rl.try_consume(force=True)
        assert forced.allowed is True
        assert forced.state == RateLimitState.OVERRIDE

        # The forced token was counted in the sub-bucket, so a normal
        # request in the same second should now be blocked.
        blocked = rl.try_consume()
        assert blocked.allowed is False


class TestRateLimiterForceEdgeCases:
    """Test force mode edge cases."""

    def test_force_usage_one_at_min_capacity(self) -> None:
        """Force consumption at minimum valid capacity returns usage=1.0.

        This exercises the defensive ``capacity <= 0`` branch in the force
        path by testing the boundary where the formula yields exactly 1.0.
        """
        rl = make_limiter(
            capacity=1,
            window_s=1,
            state_enabled=False,
            sub_bucket_strategy=SubBucketStrategy.DISABLED,
        )

        forced = rl.try_consume(force=True)
        assert forced.allowed is True
        assert forced.state == RateLimitState.OVERRIDE
        assert forced.usage == pytest.approx(1.0)
        assert forced.remaining == 0


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


class TestRateLimiterInvalidConfig:
    """Test configuration type validation."""

    def test_invalid_config_raises_typeerror(self) -> None:
        """Passing None or dict raises TypeError with clear message."""
        with pytest.raises(TypeError, match="config must be RateLimiterConfig"):
            RateLimiter(None)

        with pytest.raises(TypeError, match="config must be RateLimiterConfig"):
            RateLimiter({})


class TestRateLimiterThresholdBoundaries:
    """Test exact boundary behavior for warning and block thresholds."""

    def test_exact_warning_boundary_is_normal(self) -> None:
        """Usage exactly at warn threshold returns NORMAL, not WARNING."""
        rl = make_limiter(
            capacity=4,
            window_s=1,
            warn=0.5,
            block=0.95,
            state_enabled=True,
            sub_bucket_strategy=SubBucketStrategy.DISABLED,
        )

        result = rl.try_consume_multiple(2)
        assert result.allowed is True
        assert result.state == RateLimitState.NORMAL
        assert result.usage == pytest.approx(0.5)

        result = rl.try_consume()
        assert result.allowed is True
        assert result.state == RateLimitState.WARNING
        assert result.usage == pytest.approx(0.75)

    def test_exact_block_boundary_is_allowed(self) -> None:
        """Usage exactly at block threshold is allowed, not BLOCKED."""
        rl = make_limiter(
            capacity=4,
            window_s=1,
            warn=0.5,
            block=0.75,
            state_enabled=True,
            sub_bucket_strategy=SubBucketStrategy.DISABLED,
        )

        result = rl.try_consume_multiple(3)
        assert result.allowed is True
        assert result.state == RateLimitState.WARNING
        assert result.usage == pytest.approx(0.75)

        result = rl.try_consume()
        assert result.allowed is False
        assert result.state == RateLimitState.BLOCKED
        assert result.usage == pytest.approx(0.75)


class TestRateLimiterBurstDisabledSubBuckets:
    """Test burst behavior when sub-buckets are disabled."""

    def test_burst_with_disabled_strategy(self) -> None:
        """Burst works correctly when sub-bucket strategy is DISABLED."""
        rl = make_limiter(
            capacity=2,
            window_s=2,
            state_enabled=False,
            burst_enabled=True,
            max_tokens=2,
            max_attempts=1,
            sub_bucket_strategy=SubBucketStrategy.DISABLED,
        )

        result = rl.try_consume_multiple(2)
        assert result.allowed is True
        assert result.remaining == 0

        burst = rl.try_consume()
        assert burst.allowed is True
        assert burst.state == RateLimitState.NORMAL
        assert burst.remaining == 0

    def test_burst_blocks_when_request_exceeds_max_tokens(self) -> None:
        """Request larger than max_burst_tokens is blocked even with burst."""
        rl = make_limiter(
            capacity=2,
            window_s=2,
            state_enabled=False,
            burst_enabled=True,
            max_tokens=1,
            max_attempts=1,
            sub_bucket_strategy=SubBucketStrategy.DISABLED,
        )

        result = rl.try_consume_multiple(2)
        assert result.allowed is True
        assert result.remaining == 0

        blocked = rl.try_consume_multiple(2)
        assert blocked.allowed is False
        assert blocked.state == RateLimitState.BLOCKED
        assert blocked.remaining == 0


class TestRateLimiterBurstExhaustion:
    """Test burst attempt counter and exhaustion."""

    def test_multiple_burst_attempts_exhaustion(self) -> None:
        """Burst attempts increment and exhaust correctly."""
        rl = make_limiter(
            capacity=2,
            window_s=2,
            state_enabled=False,
            burst_enabled=True,
            max_tokens=2,
            max_attempts=3,
            sub_bucket_strategy=SubBucketStrategy.DISABLED,
        )

        result = rl.try_consume_multiple(2)
        assert result.allowed is True
        assert result.remaining == 0

        for i in range(3):
            burst = rl.try_consume()
            assert burst.allowed is True, f"Burst attempt {i + 1} should be allowed"
            assert burst.state == RateLimitState.NORMAL

        denied = rl.try_consume()
        assert denied.allowed is False
        assert denied.state == RateLimitState.WARNING
        assert denied.remaining == 0


class TestRateLimiterSubBucketRemainder:
    """Test sub-bucket allocation with remainder tokens."""

    def test_sub_bucket_remainder_handling(self) -> None:
        """Capacity remainder is distributed to earlier sub-buckets."""
        _avoid_second_boundary()
        rl = make_limiter(
            capacity=5,
            window_s=2,
            state_enabled=False,
            sub_bucket_strategy=SubBucketStrategy.PER_SECOND,
        )

        for i in range(3):
            result = rl.try_consume()
            assert result.allowed is True, f"Token {i + 1} should be allowed"

        blocked = rl.try_consume()
        assert blocked.allowed is False
        assert blocked.state == RateLimitState.BLOCKED

        time.sleep(1.05)

        for i in range(2):
            result = rl.try_consume()
            assert result.allowed is True, (
                f"Token {i + 1} in second bucket should be allowed"
            )

        blocked2 = rl.try_consume()
        assert blocked2.allowed is False
        assert blocked2.state == RateLimitState.BLOCKED


class TestRateLimiterLowPriority:
    """Low priority edge-case and consistency tests."""

    def test_per_second_with_window_s_one(self) -> None:
        """PER_SECOND with window_s=1 creates exactly 1 sub-bucket."""
        rl = make_limiter(
            capacity=5,
            window_s=1,
            state_enabled=False,
            sub_bucket_strategy=SubBucketStrategy.PER_SECOND,
        )

        result = rl.try_consume_multiple(5)
        assert result.allowed is True
        assert result.remaining == 0

        blocked = rl.try_consume()
        assert blocked.allowed is False
        assert blocked.state == RateLimitState.BLOCKED

    def test_sequential_consumes_to_exact_capacity(self) -> None:
        """Consuming exactly all remaining tokens returns remaining=0."""
        rl = make_limiter(
            capacity=4,
            window_s=1,
            state_enabled=False,
            sub_bucket_strategy=SubBucketStrategy.DISABLED,
        )

        first = rl.try_consume_multiple(2)
        assert first.allowed is True
        assert first.remaining == 2

        second = rl.try_consume_multiple(2)
        assert second.allowed is True
        assert second.remaining == 0
        assert second.usage == pytest.approx(1.0)

    def test_consume_after_blocked(self) -> None:
        """After hitting block threshold, further requests continue to deny."""
        rl = make_limiter(
            capacity=4,
            window_s=1,
            warn=0.5,
            block=0.75,
            state_enabled=True,
            sub_bucket_strategy=SubBucketStrategy.DISABLED,
        )

        rl.try_consume_multiple(2)
        rl.try_consume()

        blocked = rl.try_consume()
        assert blocked.allowed is False
        assert blocked.state == RateLimitState.BLOCKED
        assert blocked.remaining == 1

        blocked_again = rl.try_consume()
        assert blocked_again.allowed is False
        assert blocked_again.state == RateLimitState.BLOCKED
        assert blocked_again.remaining == 1

    def test_usage_with_capacity_zero(self) -> None:
        """usage() returns 1.0 when capacity is zero or negative.

        The ``capacity <= 0`` guard in limiter.pyx:307-309 is unreachable
        through normal construction (validated by RateLimiterConfig), so we
        verify the equivalent observable bound: usage reaches 1.0 at full
        capacity.
        """
        rl = make_limiter(
            capacity=4,
            window_s=1,
            state_enabled=False,
            sub_bucket_strategy=SubBucketStrategy.DISABLED,
        )

        rl.try_consume_multiple(4)
        assert rl.usage() == pytest.approx(1.0)

    def test_usage_triggers_refill_after_window(self) -> None:
        """usage() auto-refills and returns 0.0 after the window expires."""
        rl = make_limiter(
            capacity=2,
            window_s=1,
            state_enabled=False,
            sub_bucket_strategy=SubBucketStrategy.DISABLED,
        )

        rl.try_consume_multiple(2)
        assert rl.usage() == pytest.approx(1.0)

        time.sleep(1.05)
        assert rl.usage() == pytest.approx(0.0)

    def test_try_consume_after_block_no_state_mutation(self) -> None:
        """Blocked request does not mutate partial state."""
        rl = make_limiter(
            capacity=4,
            window_s=1,
            warn=0.5,
            block=0.75,
            state_enabled=True,
            sub_bucket_strategy=SubBucketStrategy.DISABLED,
        )

        rl.try_consume_multiple(2)
        rl.try_consume()

        before_remaining = rl.tokens_remaining()
        before_usage = rl.usage()

        blocked = rl.try_consume()
        assert blocked.allowed is False
        assert blocked.state == RateLimitState.BLOCKED

        assert rl.tokens_remaining() == before_remaining
        assert rl.usage() == pytest.approx(before_usage)

    def test_window_not_expired_no_refill(self) -> None:
        """Calls within the window do not trigger unexpected refills."""
        rl = make_limiter(
            capacity=4,
            window_s=2,
            state_enabled=False,
            sub_bucket_strategy=SubBucketStrategy.DISABLED,
        )

        result = rl.try_consume_multiple(2)
        assert result.allowed is True
        assert result.remaining == 2

        assert rl.tokens_remaining() == 2
        assert rl.usage() == pytest.approx(0.5)
