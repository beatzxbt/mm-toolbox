"""Accounting and policy scenarios with exact monotonic timestamps."""

from __future__ import annotations

from time import sleep
from typing import Literal

import pytest

from mm_toolbox.rate_limiter import (
    RateLimitBurstConfig,
    RateLimiter,
    RateLimiterConfig,
    RateLimitState,
    RateLimitStateConfig,
    SubBucketStrategy,
)
from mm_toolbox.time import time_monotonic_ms
from tests.cython_test_rate_limiter_clock import ClockedRateLimiter

STRATEGIES = tuple(SubBucketStrategy)
I64_MAX = (1 << 63) - 1


def make_limiter(
    capacity: int = 4,
    window_s: int = 2,
    *,
    strategy: SubBucketStrategy = SubBucketStrategy.DISABLED,
    state_enabled: bool = False,
    warning: float = 0.5,
    block: float = 0.75,
    burst_attempts: int = 0,
    burst_tokens: int = 2,
) -> ClockedRateLimiter:
    return ClockedRateLimiter(
        RateLimiterConfig(
            capacity=capacity,
            window_s=window_s,
            state_config=RateLimitStateConfig(state_enabled, warning, block),
            burst_config=RateLimitBurstConfig(
                burst_attempts > 0, burst_tokens, burst_attempts
            ),
            sub_bucket_strategy=strategy,
        )
    )


@pytest.mark.parametrize("requests", [(1, 1, 2), (2, 2), (4,)])
def test_consumption_accounting_and_exhaustion(requests: tuple[int, ...]) -> None:
    limiter = make_limiter()
    assert limiter.tokens_remaining() == 4
    assert limiter.usage() == 0.0
    consumed = 0
    for tokens in requests:
        result = (
            limiter.try_consume()
            if tokens == 1
            else limiter.try_consume_multiple(tokens)
        )
        consumed += tokens
        assert result.allowed
        assert result.state is RateLimitState.NORMAL
        assert result.remaining == limiter.tokens_remaining() == 4 - consumed
        assert result.usage == limiter.usage() == pytest.approx(consumed / 4)

    for _ in range(2):
        denied = limiter.try_consume()
        assert not denied.allowed
        assert denied.state is RateLimitState.BLOCKED
        assert denied.remaining == limiter.tokens_remaining() == 0
        assert denied.usage == limiter.usage() == 1.0


@pytest.mark.parametrize("tokens", [0, -1, -(1 << 63)])
@pytest.mark.parametrize("force", [False, True])
def test_nonpositive_queries_preserve_usage(tokens: int, force: bool) -> None:
    limiter = make_limiter(capacity=8, state_enabled=True)
    limiter.try_consume_multiple(5)
    result = limiter.try_consume_multiple(tokens, force=force)
    assert result.allowed
    assert result.state is RateLimitState.NORMAL
    assert result.remaining == limiter.tokens_remaining() == 3
    assert result.usage == limiter.usage() == 0.625


@pytest.mark.parametrize("state_enabled", [False, True])
def test_denied_request_does_not_prevent_smaller_valid_request(
    state_enabled: bool,
) -> None:
    limiter = make_limiter(capacity=10, state_enabled=state_enabled)
    limiter.try_consume_multiple(5)
    denied = limiter.try_consume_multiple(3 if state_enabled else 6)
    assert not denied.allowed
    assert denied.state is RateLimitState.BLOCKED
    assert denied.remaining == limiter.tokens_remaining() == 5
    assert denied.usage == limiter.usage() == 0.5
    accepted = limiter.try_consume()
    assert accepted.allowed
    assert accepted.remaining == 4
    assert accepted.usage == 0.6


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize(
    "tokens,allowed,state,usage",
    [
        (3, True, RateLimitState.NORMAL, 0.375),
        (4, True, RateLimitState.NORMAL, 0.5),
        (5, True, RateLimitState.WARNING, 0.625),
        (6, True, RateLimitState.WARNING, 0.75),
        (7, False, RateLimitState.BLOCKED, 0.0),
    ],
)
def test_threshold_boundaries(
    strategy: SubBucketStrategy,
    tokens: int,
    allowed: bool,
    state: RateLimitState,
    usage: float,
) -> None:
    limiter = make_limiter(
        capacity=8, window_s=1, strategy=strategy, state_enabled=True
    )
    result = limiter.try_consume_multiple(tokens)
    assert result.allowed is allowed
    assert result.state is state
    assert result.usage == limiter.usage() == usage
    assert (
        result.remaining == limiter.tokens_remaining() == (8 - tokens if allowed else 8)
    )


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_disabled_thresholds_allow_full_capacity(strategy: SubBucketStrategy) -> None:
    limiter = make_limiter(window_s=1, strategy=strategy)
    result = limiter.try_consume_multiple(4)
    assert result.allowed
    assert result.state is RateLimitState.NORMAL
    assert result.usage == 1.0


def test_threshold_usage_accumulates_across_sub_buckets() -> None:
    limiter = make_limiter(
        capacity=8, strategy=SubBucketStrategy.PER_SECOND, state_enabled=True
    )
    assert limiter.try_consume_multiple(4).state is RateLimitState.NORMAL
    limiter.now_ms = 1000
    for expected_usage in (0.625, 0.75):
        result = limiter.try_consume()
        assert result.allowed
        assert result.state is RateLimitState.WARNING
        assert result.usage == expected_usage
    denied = limiter.try_consume()
    assert not denied.allowed
    assert denied.remaining == limiter.tokens_remaining() == 2
    assert denied.usage == limiter.usage() == 0.75


@pytest.mark.parametrize(
    "capacity,allocations",
    [(1, (1,)), (4, (2, 2)), (5, (3, 2)), (10, (4, 3, 3)), (2, (1, 1, 0, 0, 0))],
)
def test_sub_bucket_allocations(capacity: int, allocations: tuple[int, ...]) -> None:
    limiter = make_limiter(
        capacity, len(allocations), strategy=SubBucketStrategy.PER_SECOND
    )
    consumed = 0
    for second, allocation in enumerate(allocations):
        limiter.now_ms = second * 1000
        if allocation:
            result = limiter.try_consume_multiple(allocation)
            assert result.allowed
            consumed += allocation
        denied = limiter.try_consume()
        assert not denied.allowed
        assert denied.remaining == limiter.tokens_remaining() == capacity - consumed
    assert consumed == capacity


@pytest.mark.parametrize("origin_ms", [0, 137, 10_123])
@pytest.mark.parametrize("elapsed_ms", [999, 1000, 1001])
def test_sub_bucket_boundary_is_relative_to_window(
    origin_ms: int, elapsed_ms: int
) -> None:
    limiter = make_limiter(strategy=SubBucketStrategy.PER_SECOND)
    limiter.now_ms = origin_ms
    limiter.refill()
    assert limiter.try_consume_multiple(2).allowed
    limiter.now_ms = origin_ms + elapsed_ms
    result = limiter.try_consume_multiple(2)
    assert result.allowed is (elapsed_ms >= 1000)
    assert result.remaining == (0 if elapsed_ms >= 1000 else 2)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_unused_sub_bucket_capacity_does_not_carry_forward(
    strategy: SubBucketStrategy,
) -> None:
    limiter = make_limiter(capacity=5, strategy=strategy)
    limiter.now_ms = 1000
    request = 2 if strategy is SubBucketStrategy.PER_SECOND else 5
    assert limiter.try_consume_multiple(request).allowed
    assert not limiter.try_consume().allowed
    assert limiter.tokens_remaining() == 5 - request


@pytest.mark.parametrize("elapsed_ms", [1999, 2000, 2001, 10_999])
@pytest.mark.parametrize("trigger", ["single", "multiple", "remaining", "usage"])
def test_every_public_time_reader_refills_at_window_boundary(
    elapsed_ms: int,
    trigger: Literal["single", "multiple", "remaining", "usage"],
) -> None:
    limiter = make_limiter()
    limiter.try_consume_multiple(4)
    limiter.now_ms = elapsed_ms
    expired = elapsed_ms >= 2000
    consumed = trigger in ("single", "multiple")
    if trigger == "single":
        assert limiter.try_consume().allowed is expired
    elif trigger == "multiple":
        assert limiter.try_consume_multiple(1).allowed is expired
    elif trigger == "remaining":
        assert limiter.tokens_remaining() == (4 if expired else 0)
    else:
        assert limiter.usage() == (0.0 if expired else 1.0)
    remaining = (3 if consumed else 4) if expired else 0
    assert limiter.tokens_remaining() == remaining
    assert limiter.usage() == (4 - remaining) / 4


def test_idle_refill_starts_one_new_window_at_observation_time() -> None:
    limiter = make_limiter(strategy=SubBucketStrategy.PER_SECOND)
    limiter.try_consume_multiple(2)
    limiter.now_ms = 10_999
    assert limiter.tokens_remaining() == 4
    assert limiter.try_consume_multiple(2).allowed
    limiter.now_ms = 11_000
    assert not limiter.try_consume().allowed
    limiter.now_ms = 11_999
    assert limiter.try_consume_multiple(2).allowed


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_explicit_refill_resets_all_budgets_and_origin(
    strategy: SubBucketStrategy,
) -> None:
    limiter = make_limiter(strategy=strategy, burst_attempts=1)
    normal = 2 if strategy is SubBucketStrategy.PER_SECOND else 4
    assert limiter.try_consume_multiple(normal).allowed
    assert limiter.try_consume_multiple(2).allowed
    assert not limiter.try_consume().allowed
    limiter.now_ms = 137
    limiter.refill()
    assert limiter.tokens_remaining() == 4
    assert limiter.usage() == 0.0
    assert limiter.try_consume_multiple(normal).allowed
    assert limiter.try_consume_multiple(2).allowed
    assert not limiter.try_consume().allowed


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("attempts", [1, 3])
def test_burst_budget_lasts_the_entire_window(
    strategy: SubBucketStrategy,
    attempts: int,
) -> None:
    limiter = make_limiter(strategy=strategy, burst_attempts=attempts)
    normal = 2 if strategy is SubBucketStrategy.PER_SECOND else 4
    assert limiter.try_consume_multiple(normal).allowed
    for _ in range(attempts):
        result = limiter.try_consume_multiple(2)
        assert result.allowed
        assert result.state is RateLimitState.NORMAL
        assert result.remaining == 0
        assert result.usage == 1.0
    limiter.now_ms = 1000
    denied = limiter.try_consume()
    assert not denied.allowed
    assert denied.state is RateLimitState.WARNING
    limiter.now_ms = 1999
    assert not limiter.try_consume().allowed
    limiter.now_ms = 2000
    assert limiter.try_consume_multiple(normal).allowed
    assert limiter.try_consume_multiple(2).allowed


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_oversized_burst_does_not_spend_attempt(strategy: SubBucketStrategy) -> None:
    limiter = make_limiter(window_s=1, strategy=strategy, burst_attempts=1)
    limiter.try_consume_multiple(4)
    denied = limiter.try_consume_multiple(3)
    assert not denied.allowed
    assert denied.state is RateLimitState.BLOCKED
    assert limiter.try_consume_multiple(2).allowed
    assert not limiter.try_consume().allowed


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("tokens", [1, 2])
def test_burst_applies_to_threshold_and_capacity_rejections_consistently(
    strategy: SubBucketStrategy,
    tokens: int,
) -> None:
    limiter = make_limiter(
        window_s=1, strategy=strategy, state_enabled=True, burst_attempts=1
    )
    assert limiter.try_consume_multiple(3).state is RateLimitState.WARNING
    burst = limiter.try_consume_multiple(tokens)
    assert burst.allowed
    assert burst.state is RateLimitState.NORMAL
    assert burst.remaining == 0
    assert burst.usage == 1.0
    exhausted = limiter.try_consume()
    assert not exhausted.allowed
    assert exhausted.state is RateLimitState.WARNING


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_normal_requests_do_not_spend_burst_attempts(
    strategy: SubBucketStrategy,
) -> None:
    limiter = make_limiter(window_s=1, strategy=strategy, burst_attempts=1)
    for _ in range(4):
        assert limiter.try_consume().allowed
    assert limiter.try_consume_multiple(2).allowed
    assert not limiter.try_consume().allowed


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_force_accounts_for_overage_and_burst_does_not_erase_it(
    strategy: SubBucketStrategy,
) -> None:
    limiter = make_limiter(strategy=strategy, state_enabled=True, burst_attempts=1)
    forced = limiter.try_consume_multiple(10, force=True)
    assert forced.allowed
    assert forced.state is RateLimitState.OVERRIDE
    assert forced.remaining == -6
    assert forced.usage == 2.5
    burst = limiter.try_consume_multiple(2)
    assert burst.allowed
    assert burst.remaining == limiter.tokens_remaining() == -6
    assert burst.usage == limiter.usage() == 2.5
    assert not limiter.try_consume().allowed


def test_forced_tokens_are_charged_to_the_active_sub_bucket() -> None:
    limiter = make_limiter(strategy=SubBucketStrategy.PER_SECOND)
    assert limiter.try_consume(force=True).state is RateLimitState.OVERRIDE
    assert limiter.try_consume().allowed
    assert not limiter.try_consume().allowed
    assert limiter.tokens_remaining() == 2
    limiter.now_ms = 1000
    assert limiter.try_consume_multiple(2).allowed
    assert limiter.tokens_remaining() == 0


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_wide_counts_and_overflow_rejection_are_atomic(
    strategy: SubBucketStrategy,
) -> None:
    capacity = 1 << 40
    limiter = make_limiter(capacity, strategy=strategy)
    wide = limiter.try_consume_multiple(1 << 31, force=True)
    assert wide.remaining == capacity - (1 << 31)
    assert limiter.tokens_remaining() == wide.remaining
    denied = limiter.try_consume_multiple(I64_MAX)
    assert not denied.allowed
    assert denied.remaining == wide.remaining
    assert limiter.usage() == wide.usage
    forced = limiter.try_consume_multiple(I64_MAX - (1 << 31), force=True)
    assert forced.allowed
    assert forced.state is RateLimitState.OVERRIDE
    assert forced.remaining == capacity - I64_MAX
    overflow = limiter.try_consume(force=True)
    assert not overflow.allowed
    assert overflow.state is RateLimitState.BLOCKED
    assert overflow.remaining == limiter.tokens_remaining() == forced.remaining
    assert overflow.usage == limiter.usage() == forced.usage


@pytest.mark.parametrize("tokens", [1 << 63, -(1 << 63) - 1])
def test_out_of_range_native_argument_preserves_state(tokens: int) -> None:
    limiter = make_limiter()
    limiter.try_consume()
    with pytest.raises(OverflowError):
        limiter.try_consume_multiple(tokens)
    assert limiter.tokens_remaining() == 3
    assert limiter.usage() == 0.25


def test_huge_burst_request_cannot_overflow_counters() -> None:
    limiter = make_limiter(
        strategy=SubBucketStrategy.PER_SECOND, burst_attempts=1, burst_tokens=I64_MAX
    )
    limiter.try_consume_multiple(4, force=True)
    result = limiter.try_consume_multiple(I64_MAX)
    assert result.allowed
    assert result.remaining == limiter.tokens_remaining() == 0
    assert result.usage == limiter.usage() == 1.0


@pytest.mark.parametrize("factory", ["second", "minute", "window"])
def test_factory_duration_and_per_second_policy(factory: str) -> None:
    if factory == "second":
        limiter = ClockedRateLimiter.per_second(4)
        capacity, window_s = 4, 1
    elif factory == "minute":
        limiter = ClockedRateLimiter.per_minute(120)
        capacity, window_s = 120, 60
    else:
        limiter = ClockedRateLimiter.per_window(6, 3)
        capacity, window_s = 6, 3
    first_bucket = 3 if window_s == 1 else capacity // window_s
    assert limiter.try_consume_multiple(first_bucket).allowed
    assert not limiter.try_consume().allowed
    limiter.try_consume_multiple(capacity - first_bucket, force=True)
    limiter.now_ms = window_s * 1000 - 1
    assert limiter.tokens_remaining() == 0
    limiter.now_ms += 1
    assert limiter.tokens_remaining() == capacity
    assert limiter.try_consume().allowed


def test_production_monotonic_clock_refills() -> None:
    """One integration smoke test keeps the real native clock connected."""
    config = RateLimiterConfig(
        capacity=2,
        window_s=1,
        state_config=RateLimitStateConfig(False, 0.5, 0.75),
        burst_config=RateLimitBurstConfig.default(),
        sub_bucket_strategy=SubBucketStrategy.DISABLED,
    )
    limiter = RateLimiter(config)
    assert limiter.try_consume_multiple(2).allowed
    deadline = time_monotonic_ms() + 3000
    while limiter.tokens_remaining() == 0 and time_monotonic_ms() < deadline:
        sleep(0.01)
    assert limiter.tokens_remaining() == 2
    assert limiter.usage() == 0.0
