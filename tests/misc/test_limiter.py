"""Layer 1–2 — Primitives and component tests for rate limiter utilities.

Covers ``RateLimitStateConfig`` and ``RateLimitBurstConfig`` validation,
``RateLimiterConfig`` defaults, token consumption, sub-bucket allocation,
burst allowances, state transitions (NORMAL → WARNING → BLOCKED), refill
behaviour, and exact boundary conditions.
"""

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
    """Sleep briefly to avoid crossing a second boundary mid-test.

    Args:
        buffer_s: Seconds before the boundary that trigger a sleep.
    """
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
    """Create a ``RateLimiter`` with explicit configuration controls.

    Args:
        capacity: Token capacity per window.
        window_s: Window duration in seconds.
        warn: Warning threshold as a fraction of capacity.
        block: Block threshold as a fraction of capacity.
        state_enabled: Whether state transitions are active.
        burst_enabled: Whether burst tokens are allowed.
        max_tokens: Maximum burst tokens.
        max_attempts: Maximum burst attempts.
        sub_bucket_strategy: Sub-bucket allocation strategy.

    Returns:
        Configured ``RateLimiter`` instance.
    """
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
    """Layer 1 — ``RateLimitStateConfig`` primitive validation and defaults."""

    def test_default_thresholds(self) -> None:
        """Given no arguments, defaults are enabled with warn=0.75 and block=0.95."""
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
        """Given thresholds outside (0, 1) or non-monotonic, construction raises ``ValueError``."""
        with pytest.raises(ValueError):
            RateLimitStateConfig(
                is_enabled=True,
                warning_threshold=warning_threshold,
                block_threshold=block_threshold,
            )


class TestRateLimitBurstConfig:
    """Layer 1 — ``RateLimitBurstConfig`` primitive validation and defaults."""

    def test_default_config(self) -> None:
        """Given no arguments, burst is disabled with zero limits."""
        cfg = RateLimitBurstConfig.default()
        assert cfg.is_enabled is False
        assert cfg.max_tokens == 0
        assert cfg.max_burst_attempts == 0

    @pytest.mark.parametrize("max_tokens, max_attempts", [(0, 1), (1, 0), (-1, 2)])
    def test_invalid_enabled_settings(self, max_tokens: int, max_attempts: int) -> None:
        """Given enabled burst with non-positive limits, construction raises ``ValueError``."""
        with pytest.raises(ValueError):
            RateLimitBurstConfig(
                is_enabled=True,
                max_tokens=max_tokens,
                max_burst_attempts=max_attempts,
            )

    def test_valid_enabled_settings(self) -> None:
        """Given enabled burst with positive limits, construction succeeds."""
        cfg = RateLimitBurstConfig(
            is_enabled=True,
            max_tokens=2,
            max_burst_attempts=1,
        )
        assert cfg.is_enabled is True
        assert cfg.max_tokens == 2
        assert cfg.max_burst_attempts == 1


class TestRateLimiterConfig:
    """Layer 1 — ``RateLimiterConfig`` primitive validation and defaults."""

    def test_default_config(self) -> None:
        """Given ``default()``, state is enabled, burst is disabled, and strategy is PER_SECOND."""
        cfg = RateLimiterConfig.default(capacity=5, window_s=2)
        assert cfg.capacity == 5
        assert cfg.window_s == 2
        assert cfg.state_config.is_enabled is True
        assert cfg.burst_config.is_enabled is False
        assert cfg.sub_bucket_strategy is SubBucketStrategy.PER_SECOND

    @pytest.mark.parametrize("capacity, window_s", [(0, 1), (-1, 1), (1, 0), (1, -2)])
    def test_invalid_capacity_or_window(self, capacity: int, window_s: int) -> None:
        """Given non-positive capacity or window, ``default()`` raises ``ValueError``."""
        with pytest.raises(ValueError):
            RateLimiterConfig.default(capacity=capacity, window_s=window_s)


class TestRateLimiterBasicOperations:
    """Layer 2 — Core consume and accounting behaviour."""

    def test_basic_consumption_and_usage(self) -> None:
        """Given a capacity of 4, consuming 1 token leaves 3 and usage=0.25."""
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
        """Given ``force=True``, consumption bypasses limits and marks ``OVERRIDE``."""
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
    """Layer 2 — Warning and blocking threshold behaviour."""

    def test_warning_and_block_transitions(self) -> None:
        """Given warn=0.5 and block=0.75, state transitions from NORMAL to WARNING to BLOCKED."""
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
    """Layer 2 — Per-second sub-bucket behaviour."""

    def test_sub_bucket_limits_same_second(self) -> None:
        """Given PER_SECOND with capacity=4 and window=2, only 2 tokens are available in the first second."""
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
    """Layer 2 — Burst allowance behaviour."""

    def test_burst_allows_limited_overage(self) -> None:
        """Given burst with max_tokens=2 and max_attempts=1, one extra request succeeds."""
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
    """Layer 1 — ``RateLimiter`` factory constructors."""

    def test_factory_constructors(self) -> None:
        """Given factory methods, each returns a limiter with the expected initial capacity."""
        per_second = RateLimiter.per_second(3)
        per_minute = RateLimiter.per_minute(4)
        per_window = RateLimiter.per_window(5, 2)

        assert per_second.tokens_remaining() == 3
        assert per_minute.tokens_remaining() == 4
        assert per_window.tokens_remaining() == 5


class TestRateLimiterDistribution:
    """Layer 2 — Token allocation across sub-buckets."""

    def test_odd_capacity_distribution(self) -> None:
        """Given capacity=5 and window=2, PER_SECOND allocates [3, 2] tokens.

        Odd capacity must distribute the remainder to early sub-buckets;
        otherwise a 5-token limiter would allow 2+2=4 tokens instead of 5.
        """
        _avoid_second_boundary()
        rl = make_limiter(
            capacity=5,
            window_s=2,
            state_enabled=False,
            sub_bucket_strategy=SubBucketStrategy.PER_SECOND,
        )

        first = rl.try_consume_multiple(3)
        assert first.allowed is True
        assert first.remaining == 2

        denied = rl.try_consume()
        assert denied.allowed is False

        time.sleep(1.05)

        second = rl.try_consume_multiple(2)
        assert second.allowed is True
        assert second.remaining == 0

        denied2 = rl.try_consume()
        assert denied2.allowed is False


class TestRateLimiterConsumeMethods:
    """Layer 2 — Direct consume methods and large requests."""

    def test_try_consume_directly(self) -> None:
        """Given ``try_consume()``, it behaves like ``try_consume_multiple(1)``."""
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
        """Given a request larger than remaining capacity, it is denied immediately."""
        rl = make_limiter(
            capacity=2,
            window_s=1,
            state_enabled=False,
            sub_bucket_strategy=SubBucketStrategy.DISABLED,
        )

        rl.try_consume_multiple(2)
        denied = rl.try_consume_multiple(10)
        assert denied.allowed is False
        assert denied.state == RateLimitState.BLOCKED
        assert denied.remaining == 0


class TestRateLimiterRefillBehavior:
    """Layer 2 — Explicit and implicit refill behaviour."""

    def test_explicit_refill_resets_usage(self) -> None:
        """Given ``refill()``, all usage counters reset to zero mid-window."""
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
        """Given an expired window, ``tokens_remaining()`` auto-refills."""
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
        """Given an expired window, burst attempts are restored."""
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

        rl.try_consume_multiple(2)
        burst = rl.try_consume()
        assert burst.allowed is True

        exhausted = rl.try_consume()
        assert exhausted.allowed is False
        assert exhausted.state == RateLimitState.WARNING

        time.sleep(1.05)

        rl.try_consume_multiple(2)
        burst2 = rl.try_consume()
        assert burst2.allowed is True


class TestRateLimiterThresholdsExtended:
    """Layer 2 — State transitions with and without sub-buckets."""

    def test_thresholds_with_per_second(self) -> None:
        """Given PER_SECOND enabled, warning and block thresholds behave correctly across buckets."""
        _avoid_second_boundary()
        rl = make_limiter(
            capacity=8,
            window_s=2,
            warn=0.5,
            block=0.75,
            state_enabled=True,
            sub_bucket_strategy=SubBucketStrategy.PER_SECOND,
        )

        normal = rl.try_consume_multiple(4)
        assert normal.allowed is True
        assert normal.state == RateLimitState.NORMAL
        assert normal.usage == pytest.approx(0.5)

        time.sleep(1.05)

        warning = rl.try_consume_multiple(1)
        assert warning.allowed is True
        assert warning.state == RateLimitState.WARNING
        assert warning.usage == pytest.approx(0.625)

        warning2 = rl.try_consume_multiple(1)
        assert warning2.allowed is True
        assert warning2.state == RateLimitState.WARNING
        assert warning2.usage == pytest.approx(0.75)

        blocked = rl.try_consume_multiple(1)
        assert blocked.allowed is False
        assert blocked.state == RateLimitState.BLOCKED

    def test_thresholds_disabled(self) -> None:
        """Given DISABLED strategy, state transitions behave linearly across the whole window."""
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
        """Given burst above the block threshold, the state reports ``NORMAL`` instead of ``BLOCKED``."""
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

        warning = rl.try_consume_multiple(3)
        assert warning.allowed is True
        assert warning.state == RateLimitState.WARNING
        assert warning.usage == pytest.approx(0.75)

        burst = rl.try_consume_multiple(2)
        assert burst.allowed is True
        assert burst.state == RateLimitState.NORMAL
        assert burst.remaining == 0
        assert burst.usage == pytest.approx(1.0)

    def test_blocked_without_burst(self) -> None:
        """Given burst disabled, over-capacity requests return ``BLOCKED``."""
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
    """Layer 2 — Burst interaction with per-second sub-buckets."""

    def test_sub_bucket_exhaustion_with_burst(self) -> None:
        """Given burst, sub-bucket usage is still capped at the sub-allocation limit."""
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

        rl.try_consume_multiple(2)

        burst = rl.try_consume_multiple(1)
        assert burst.allowed is True
        assert burst.remaining == 1

        denied = rl.try_consume()
        assert denied.allowed is False
        assert denied.state == RateLimitState.WARNING

    def test_burst_cap_at_capacity(self) -> None:
        """Given burst, ``used_tokens`` never exceeds overall capacity."""
        rl = make_limiter(
            capacity=2,
            window_s=1,
            state_enabled=False,
            burst_enabled=True,
            max_tokens=2,
            max_attempts=1,
            sub_bucket_strategy=SubBucketStrategy.DISABLED,
        )

        rl.try_consume_multiple(2)

        burst = rl.try_consume_multiple(2)
        assert burst.allowed is True
        assert burst.remaining == 0
        assert burst.usage == pytest.approx(1.0)

    def test_force_increments_sub_bucket(self) -> None:
        """Given ``force=True``, sub-bucket usage is incremented even though the limit was bypassed."""
        _avoid_second_boundary()
        rl = make_limiter(
            capacity=4,
            window_s=2,
            state_enabled=False,
            sub_bucket_strategy=SubBucketStrategy.PER_SECOND,
        )

        rl.try_consume_multiple(2)

        forced = rl.try_consume(force=True)
        assert forced.allowed is True
        assert forced.state == RateLimitState.OVERRIDE

        blocked = rl.try_consume()
        assert blocked.allowed is False


class TestRateLimiterForceEdgeCases:
    """Layer 2 — Force mode edge cases."""

    def test_force_usage_one_at_min_capacity(self) -> None:
        """Given capacity=1 and ``force=True``, usage returns exactly 1.0.

        This exercises the defensive ``capacity <= 0`` branch boundary.
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
    """Layer 3 — Refill behaviour in a realistic loop."""

    def test_refill_resets_after_window(self, wait_for) -> None:
        """Given a full-window consumption, tokens are restored after the window elapses."""
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
    """Layer 1 — Configuration type validation."""

    def test_invalid_config_raises_typeerror(self) -> None:
        """Given ``None`` or a ``dict``, construction raises ``TypeError``."""
        with pytest.raises(TypeError, match="config must be RateLimiterConfig"):
            RateLimiter(None)

        with pytest.raises(TypeError, match="config must be RateLimiterConfig"):
            RateLimiter({})


class TestRateLimiterThresholdBoundaries:
    """Layer 2 — Exact boundary behaviour for warning and block thresholds."""

    def test_exact_warning_boundary_is_normal(self) -> None:
        """Given usage exactly at warn=0.5, the state is ``NORMAL``, not ``WARNING``."""
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
        """Given usage exactly at block=0.75, the request is allowed, not ``BLOCKED``."""
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
    """Layer 2 — Burst behaviour when sub-buckets are disabled."""

    def test_burst_with_disabled_strategy(self) -> None:
        """Given DISABLED strategy, burst still functions correctly."""
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
        """Given a request larger than ``max_burst_tokens``, it is blocked even with burst enabled."""
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
    """Layer 2 — Burst attempt counter and exhaustion."""

    def test_multiple_burst_attempts_exhaustion(self) -> None:
        """Given 3 burst attempts, the 4th request is denied."""
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
    """Layer 2 — Sub-bucket allocation with remainder tokens."""

    def test_sub_bucket_remainder_handling(self) -> None:
        """Given capacity=5 and window=2, 3 tokens are in the first bucket and 2 in the second."""
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
        """Given PER_SECOND with window_s=1, exactly 1 sub-bucket is created."""
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
        """Given sequential consumes, exactly all remaining tokens can be consumed."""
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
        """Given a blocked state, further requests continue to be denied."""
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
        """Given full consumption, usage reaches 1.0.

        The ``capacity <= 0`` guard in limiter.pyx is unreachable through
        normal construction (validated by ``RateLimiterConfig``), so we
        verify the observable bound: usage reaches 1.0 at full capacity.
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
        """Given an expired window, ``usage()`` auto-refills and returns 0.0."""
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
        """Given a blocked request, partial state is not mutated."""
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
        """Given calls within the window, no unexpected refill occurs."""
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
