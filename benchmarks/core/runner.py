"""Base benchmark runner classes.

Provides template method pattern for benchmark execution with common patterns
like warmup phases, timing, and cleanup.
"""

from __future__ import annotations

import gc
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

from .stats import BenchmarkStatistics

TConfig = TypeVar("TConfig")


@dataclass
class BaseBenchmarkConfig(ABC):
    """Base configuration for all benchmarks.

    Args:
        num_operations: Number of operations to benchmark.
        warmup_operations: Number of warmup operations before measurement.
    """

    num_operations: int = 100_000
    warmup_operations: int = 1_000


class BenchmarkRunner(ABC, Generic[TConfig]):
    """Base benchmark runner with common execution patterns.

    Uses template method pattern where subclasses implement the specific
    benchmark logic while the framework handles common patterns like warmup,
    timing, and statistics collection.

    Subclasses must implement:
        _create_subject(): Create the object to benchmark.
        _run_benchmark_suite(): Run all benchmark operations.

    Args:
        config: Benchmark configuration.
    """

    def __init__(self, config: TConfig) -> None:
        """Initialize benchmark runner.

        Args:
            config: Benchmark configuration.
        """
        self.config = config
        self.stats: BenchmarkStatistics | None = None

    @abstractmethod
    def _create_subject(self):
        """Create the object/system under test.

        Returns:
            Object to benchmark (e.g., ringbuffer, orderbook).
        """

    @abstractmethod
    def _run_benchmark_suite(self, subject) -> None:
        """Run all benchmark operations.

        Args:
            subject: Object returned by _create_subject().
        """

    def measure_operation(
        self,
        operation_name: str,
        operation_callable: Callable[[], None],
        warmup_count: int | None = None,
        measure_count: int | None = None,
        **metadata,
    ) -> None:
        """Measure a single operation with warmup.

        Encapsulates the common pattern of warmup + measurement with nanosecond
        timing. Automatically tracks results in benchmark statistics.

        Args:
            operation_name: Name for tracking (e.g., "insert(size=128)").
            operation_callable: Callable that performs one operation.
            warmup_count: Number of warmup iterations (defaults to config).
            measure_count: Number of measurement iterations (defaults to config).
            **metadata: Optional metadata to track with each measurement.
        """
        if self.stats is None:
            raise RuntimeError("Statistics not initialized")

        warmup = warmup_count or self.config.warmup_operations
        measure = measure_count or self.config.num_operations

        # Warmup phase
        for _ in range(warmup):
            operation_callable()

        # Measurement phase
        metrics = self.stats.add_operation(operation_name)
        for _ in range(measure):
            start = time.perf_counter_ns()
            operation_callable()
            elapsed = time.perf_counter_ns() - start
            metrics.add_latency(elapsed, **metadata)

    def run(self) -> BenchmarkStatistics:
        """Run the full benchmark suite.

        Creates the subject, runs benchmarks, and collects statistics.

        Returns:
            Collected statistics.
        """
        self.stats = BenchmarkStatistics()
        subject = self._create_subject()

        start = time.perf_counter_ns()
        self._run_benchmark_suite(subject)
        end = time.perf_counter_ns()

        self.stats.total_time_ns = end - start
        return self.stats


class MultiSizeBenchmarkRunner(ABC, Generic[TConfig]):
    """Runner for benchmarks across multiple sizes/configs.

    Handles running benchmarks across different sizes with proper cleanup
    and progress reporting.

    Subclasses must implement:
        _create_runner(size): Create a BenchmarkRunner for a specific size.
        _get_key_operations(): Return list of operation names for comparison.

    Args:
        base_config: Base configuration to use.
        sizes: List of sizes to test.
    """

    def __init__(self, base_config: TConfig, sizes: list[int]) -> None:
        """Initialize multi-size runner.

        Args:
            base_config: Base configuration to clone/modify.
            sizes: List of sizes to test.
        """
        self.base_config = base_config
        self.sizes = sizes
        self.results: list[tuple[int, BenchmarkStatistics]] = []

    @abstractmethod
    def _create_runner(self, size: int) -> BenchmarkRunner:
        """Create a benchmark runner for a specific size.

        Args:
            size: Size parameter (e.g., buffer size, orderbook levels).

        Returns:
            Configured benchmark runner.
        """

    @abstractmethod
    def _get_key_operations(self) -> list[str]:
        """Get list of key operation names for comparative reporting.

        Returns:
            List of operation names (e.g., ["insert", "consume"]).
        """

    def run(self, print_progress: bool = True) -> list[tuple[int, BenchmarkStatistics]]:
        """Run benchmarks across all sizes.

        Args:
            print_progress: Whether to print progress messages.

        Returns:
            List of (size, statistics) tuples.
        """
        for size in self.sizes:
            if print_progress:
                print(f"\n{'=' * 100}")
                print(f"Testing size: {size}")
                print(f"{'=' * 100}")

            runner = self._create_runner(size)
            stats = runner.run()
            self.results.append((size, stats))

            # Clean up between runs
            del runner
            gc.collect()

        return self.results
