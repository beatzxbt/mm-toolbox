"""Benchmark labels must describe the operation that was actually timed."""

import pytest

from benchmarks.core import BenchmarkStatistics
from benchmarks.rate_limiter.benchmark_rate_limiter import (
    RateLimiterBenchmark,
    RateLimiterBenchmarkConfig,
)
from mm_toolbox.rate_limiter import (
    RateLimitBurstConfig,
    RateLimiter,
    RateLimiterConfig,
    RateLimitState,
    RateLimitStateConfig,
    SubBucketStrategy,
)
from mm_toolbox.rate_limiter.result import ConsumeResult


@pytest.mark.parametrize("strategy", ["disabled", "per_second"])
@pytest.mark.parametrize("state_enabled", [False, True])
@pytest.mark.parametrize("attempts", [1, 5])
def test_burst_benchmark_times_admitted_bursts(
    monkeypatch: pytest.MonkeyPatch,
    strategy: str,
    state_enabled: bool,
    attempts: int,
) -> None:
    observed: list[ConsumeResult] = []

    class ObservedLimiter(RateLimiter):
        def try_consume(self, force: bool = False) -> ConsumeResult:
            result = super().try_consume(force)
            observed.append(result)
            return result

    benchmark = RateLimiterBenchmark(
        RateLimiterBenchmarkConfig(
            num_operations=3,
            warmup_operations=1,
            capacity=1000,
            window_s=60,
            sub_bucket_strategy=strategy,
            state_enabled=state_enabled,
            burst_enabled=True,
            max_burst_tokens=10,
            max_burst_attempts=attempts,
        )
    )
    # Preserve the real native limiter while observing each single-token result.
    limiter = ObservedLimiter(
        RateLimiterConfig(
            capacity=1000,
            window_s=60,
            state_config=RateLimitStateConfig(state_enabled, 0.75, 0.95),
            burst_config=RateLimitBurstConfig(True, 10, attempts),
            sub_bucket_strategy=SubBucketStrategy(strategy),
        )
    )
    monkeypatch.setattr(benchmark, "_create_limiter", lambda: limiter)
    benchmark.stats = BenchmarkStatistics()
    benchmark._benchmark_try_consume_burst()

    assert len(observed) == 4
    assert all(
        result.allowed and result.state is RateLimitState.NORMAL for result in observed
    )
    assert all(result.remaining == 0 and result.usage == 1.0 for result in observed)
