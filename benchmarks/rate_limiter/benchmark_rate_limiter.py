"""Benchmarks for RateLimiter overhead.

Usage:
    uv run python benchmarks/rate_limiter/benchmark_rate_limiter.py [ARGS]

Measures:
- try_consume_allowed: single-token consume when bucket has capacity
- try_consume_denied: single-token consume when bucket is saturated
- try_consume_multiple: multi-token consume when allowed
- tokens_remaining: lightweight state query
- try_consume_burst: burst consume when bucket is exhausted (if burst enabled)
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

try:
    from benchmarks.core import (
        BaseBenchmarkConfig,
        BenchmarkCLI,
        BenchmarkReporter,
        BenchmarkRunner,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from benchmarks.core import (
        BaseBenchmarkConfig,
        BenchmarkCLI,
        BenchmarkReporter,
        BenchmarkRunner,
    )

from mm_toolbox.rate_limiter.config import (
    RateLimiterConfig,
    RateLimitStateConfig,
    RateLimitBurstConfig,
    SubBucketStrategy,
)
from mm_toolbox.rate_limiter.limiter import RateLimiter


@dataclass
class RateLimiterBenchmarkConfig(BaseBenchmarkConfig):
    """Configuration for RateLimiter benchmarking."""

    capacity: int = 1000
    window_s: int = 60
    sub_bucket_strategy: str = "disabled"
    state_enabled: bool = False
    burst_enabled: bool = False
    max_burst_tokens: int = 10
    max_burst_attempts: int = 5
    consume_tokens: int = 1


class RateLimiterBenchmark(BenchmarkRunner[RateLimiterBenchmarkConfig]):
    """Benchmark runner for RateLimiter operations."""

    def _create_subject(self) -> None:
        """No persistent benchmark subject is required."""
        return None

    def _create_limiter(self) -> RateLimiter:
        """Create a RateLimiter with the benchmark configuration."""
        sub_strategy = (
            SubBucketStrategy.PER_SECOND
            if self.config.sub_bucket_strategy == "per_second"
            else SubBucketStrategy.DISABLED
        )

        state_config = RateLimitStateConfig(
            is_enabled=self.config.state_enabled,
            warning_threshold=0.75,
            block_threshold=0.95,
        )

        burst_config = RateLimitBurstConfig(
            is_enabled=self.config.burst_enabled,
            max_tokens=self.config.max_burst_tokens,
            max_burst_attempts=self.config.max_burst_attempts,
        )

        config = RateLimiterConfig(
            capacity=self.config.capacity,
            window_s=self.config.window_s,
            state_config=state_config,
            burst_config=burst_config,
            sub_bucket_strategy=sub_strategy,
        )

        return RateLimiter(config)

    def _run_benchmark_suite(self, _subject: None) -> None:
        """Run all benchmark operations."""
        self._benchmark_try_consume_allowed()
        self._benchmark_try_consume_denied()
        self._benchmark_try_consume_multiple()
        self._benchmark_tokens_remaining()
        if self.config.burst_enabled:
            self._benchmark_try_consume_burst()

    def _benchmark_try_consume_allowed(self) -> None:
        """Measure single-token consume when bucket has capacity."""
        if self.stats is None:
            raise RuntimeError("Statistics not initialized")

        limiter = self._create_limiter()
        warmup = self.config.warmup_operations
        measure = self.config.num_operations

        for _ in range(warmup):
            limiter.refill()
            limiter.try_consume()

        metrics = self.stats.add_operation("try_consume_allowed")
        for _ in range(measure):
            limiter.refill()
            start = time.perf_counter_ns()
            limiter.try_consume()
            elapsed = time.perf_counter_ns() - start
            metrics.add_latency(elapsed)

    def _benchmark_try_consume_denied(self) -> None:
        """Measure single-token consume when bucket is saturated."""
        if self.stats is None:
            raise RuntimeError("Statistics not initialized")

        limiter = self._create_limiter()
        warmup = self.config.warmup_operations
        measure = self.config.num_operations

        # Exhaust the bucket (and burst attempts if enabled)
        while limiter.try_consume().allowed:
            pass

        for _ in range(warmup):
            limiter.try_consume()

        metrics = self.stats.add_operation("try_consume_denied")
        for _ in range(measure):
            start = time.perf_counter_ns()
            limiter.try_consume()
            elapsed = time.perf_counter_ns() - start
            metrics.add_latency(elapsed)

    def _benchmark_try_consume_multiple(self) -> None:
        """Measure multi-token consume when allowed."""
        if self.stats is None:
            raise RuntimeError("Statistics not initialized")

        limiter = self._create_limiter()
        tokens = self.config.consume_tokens
        warmup = self.config.warmup_operations
        measure = self.config.num_operations

        for _ in range(warmup):
            limiter.refill()
            limiter.try_consume_multiple(tokens)

        metrics = self.stats.add_operation("try_consume_multiple")
        for _ in range(measure):
            limiter.refill()
            start = time.perf_counter_ns()
            limiter.try_consume_multiple(tokens)
            elapsed = time.perf_counter_ns() - start
            metrics.add_latency(elapsed)

    def _benchmark_tokens_remaining(self) -> None:
        """Measure lightweight state query."""
        if self.stats is None:
            raise RuntimeError("Statistics not initialized")

        limiter = self._create_limiter()
        warmup = self.config.warmup_operations
        measure = self.config.num_operations

        for _ in range(warmup):
            limiter.tokens_remaining()

        metrics = self.stats.add_operation("tokens_remaining")
        for _ in range(measure):
            start = time.perf_counter_ns()
            limiter.tokens_remaining()
            elapsed = time.perf_counter_ns() - start
            metrics.add_latency(elapsed)

    def _benchmark_try_consume_burst(self) -> None:
        """Measure burst consumes when bucket is exhausted."""
        if self.stats is None:
            raise RuntimeError("Statistics not initialized")

        if not self.config.burst_enabled:
            return

        limiter = self._create_limiter()
        warmup = self.config.warmup_operations
        measure = self.config.num_operations

        for _ in range(warmup):
            limiter.refill()
            for _ in range(self.config.capacity):
                if not limiter.try_consume().allowed:
                    break
            limiter.try_consume()

        metrics = self.stats.add_operation("try_consume_burst")
        for _ in range(measure):
            limiter.refill()
            for _ in range(self.config.capacity):
                if not limiter.try_consume().allowed:
                    break
            start = time.perf_counter_ns()
            limiter.try_consume()
            elapsed = time.perf_counter_ns() - start
            metrics.add_latency(elapsed)


def _build_config(args: argparse.Namespace) -> RateLimiterBenchmarkConfig:
    """Build benchmark config from CLI args."""
    return RateLimiterBenchmarkConfig(
        num_operations=max(1, int(args.num_operations)),
        warmup_operations=max(0, int(args.warmup)),
        capacity=max(1, int(args.capacity)),
        window_s=max(1, int(args.window_s)),
        sub_bucket_strategy=args.sub_bucket,
        state_enabled=args.state,
        burst_enabled=args.burst,
        max_burst_tokens=max(1, int(args.max_burst_tokens)),
        max_burst_attempts=max(1, int(args.max_burst_attempts)),
        consume_tokens=max(1, int(args.consume_tokens)),
    )


def run_single(config: RateLimiterBenchmarkConfig) -> None:
    """Run one benchmark configuration and print a report."""
    benchmark = RateLimiterBenchmark(config)
    stats = benchmark.run()

    reporter = BenchmarkReporter(
        "RateLimiter Benchmark Results",
        {
            "Capacity": config.capacity,
            "Window (s)": config.window_s,
            "Sub-bucket strategy": config.sub_bucket_strategy,
            "State enabled": config.state_enabled,
            "Burst enabled": config.burst_enabled,
            "Max burst tokens": config.max_burst_tokens,
            "Max burst attempts": config.max_burst_attempts,
            "Consume tokens": config.consume_tokens,
        },
    )
    reporter.print_full_report(stats, warmup=config.warmup_operations)


def main() -> None:
    """Main entry point."""
    cli = BenchmarkCLI("Benchmark RateLimiter overhead")
    cli.parser.add_argument(
        "--capacity",
        type=int,
        default=1000,
        help="Token bucket capacity (default: 1000)",
    )
    cli.parser.add_argument(
        "--window-s",
        type=int,
        default=60,
        help="Window duration in seconds (default: 60)",
    )
    cli.parser.add_argument(
        "--sub-bucket",
        choices=["disabled", "per_second"],
        default="disabled",
        help="Sub-bucket strategy (default: disabled)",
    )
    cli.parser.add_argument(
        "--state",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable state thresholds (default: False)",
    )
    cli.parser.add_argument(
        "--burst",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable burst handling (default: False)",
    )
    cli.parser.add_argument(
        "--max-burst-tokens",
        type=int,
        default=10,
        help="Maximum tokens per burst attempt (default: 10)",
    )
    cli.parser.add_argument(
        "--max-burst-attempts",
        type=int,
        default=5,
        help="Maximum burst attempts (default: 5)",
    )
    cli.parser.add_argument(
        "--consume-tokens",
        type=int,
        default=1,
        help="Number of tokens to consume in multi-token benchmark (default: 1)",
    )
    cli.parser.add_argument(
        "--num-operations",
        type=int,
        default=100_000,
        help="Number of operations to benchmark (default: 100,000)",
    )

    args = cli.parse()
    config = _build_config(args)
    run_single(config)


if __name__ == "__main__":
    main()
