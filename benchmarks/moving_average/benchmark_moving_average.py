"""Benchmarks for moving average implementations.

Usage:
    uv run python benchmarks/moving_average/benchmark_moving_average.py [options]

Measures:
- initialize: warm-start cost from an ndarray of length = window
- update: hot-path - ingest one tick and update the MA value
- next: peek-forward - calculate next value without mutating state
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from benchmarks.core import (
        BaseBenchmarkConfig,
        BenchmarkCLI,
        BenchmarkReporter,
        BenchmarkRunner,
        BenchmarkStatistics,
        ComparativeReporter,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from benchmarks.core import (
        BaseBenchmarkConfig,
        BenchmarkCLI,
        BenchmarkReporter,
        BenchmarkRunner,
        BenchmarkStatistics,
        ComparativeReporter,
    )

from mm_toolbox.moving_average.ema import ExponentialMovingAverage
from mm_toolbox.moving_average.sma import SimpleMovingAverage
from mm_toolbox.moving_average.tema import TimeExponentialMovingAverage
from mm_toolbox.moving_average.wma import WeightedMovingAverage


MA_TYPES = [
    ("EMA", ExponentialMovingAverage),
    ("SMA", SimpleMovingAverage),
    ("TEMA", TimeExponentialMovingAverage),
    ("WMA", WeightedMovingAverage),
]


@dataclass
class MovingAverageBenchmarkConfig(BaseBenchmarkConfig):
    """Configuration for moving average benchmarking.

    Args:
        window: Number of samples in the moving average window.
        is_fast: If True, skip storing historical values.
        half_life_s: TEMA half life in seconds.
        num_operations: Number of operations to benchmark.
        warmup_operations: Number of warmup operations before measurement.
    """

    window: int = 100
    is_fast: bool = False
    half_life_s: float = 1.0


class MovingAverageBenchmark(BenchmarkRunner[MovingAverageBenchmarkConfig]):
    """Benchmark runner for a single moving average type.

    Args:
        config: Benchmark configuration.
        ma_class: Moving average class to benchmark.
        ma_name: Display name for the moving average type.
        data: Pre-generated synthetic tick data.
    """

    def __init__(
        self,
        config: MovingAverageBenchmarkConfig,
        ma_class: type,
        ma_name: str,
        data: np.ndarray,
    ) -> None:
        """Initialize benchmark runner.

        Args:
            config: Benchmark configuration.
            ma_class: Moving average class to benchmark.
            ma_name: Display name for the moving average type.
            data: Pre-generated synthetic tick data.
        """
        super().__init__(config)
        self.ma_class = ma_class
        self.ma_name = ma_name
        self.data = data

    def _create_subject(self):
        """Create the moving average instance under test.

        Returns:
            Configured moving average instance.
        """
        if self.ma_class is TimeExponentialMovingAverage:
            return self.ma_class(
                self.config.window,
                self.config.is_fast,
                self.config.half_life_s,
            )
        return self.ma_class(self.config.window, self.config.is_fast)

    def _create_new_subject(self):
        """Create a fresh moving average instance.

        Returns:
            New configured moving average instance.
        """
        return self._create_subject()

    def _run_benchmark_suite(self, subject) -> None:
        """Run all benchmark operations.

        Args:
            subject: Moving average instance returned by _create_subject().
        """
        window = self.config.window
        num_ops = self.config.num_operations

        init_data = self.data[:window]
        tick_data = self.data[window : window + num_ops]

        # Pre-initialize subject for update/next benchmarks
        subject.initialize(init_data)

        self._benchmark_initialize(init_data)
        self._benchmark_update(subject, tick_data)
        self._benchmark_next(subject, tick_data)

    def _benchmark_initialize(self, init_data: np.ndarray) -> None:
        """Measure warm-start initialization cost.

        Args:
            init_data: Array of length window used for initialization.
        """

        def op() -> None:
            ma = self._create_new_subject()
            ma.initialize(init_data)

        self.measure_operation("initialize", op)

    def _benchmark_update(self, subject, tick_data: np.ndarray) -> None:
        """Measure hot-path update latency.

        Args:
            subject: Pre-initialized moving average instance.
            tick_data: Array of tick values to ingest.
        """
        idx = [0]

        def op() -> None:
            value = tick_data[idx[0] % len(tick_data)]
            subject.update(value)
            idx[0] += 1

        self.measure_operation("update", op)

    def _benchmark_next(self, subject, tick_data: np.ndarray) -> None:
        """Measure peek-forward next latency.

        Args:
            subject: Pre-initialized moving average instance.
            tick_data: Array of tick values to evaluate.
        """
        idx = [0]

        def op() -> None:
            value = tick_data[idx[0] % len(tick_data)]
            subject.next(value)
            idx[0] += 1

        self.measure_operation("next", op)


def _build_config(args: argparse.Namespace) -> MovingAverageBenchmarkConfig:
    """Build benchmark config from CLI args.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Configured MovingAverageBenchmarkConfig.
    """
    num_ops = (
        args.num_operations if args.num_operations is not None else args.operations
    )
    return MovingAverageBenchmarkConfig(
        window=max(2, int(args.window)),
        is_fast=args.fast,
        half_life_s=max(0.001, float(args.half_life_s)),
        num_operations=max(1, int(num_ops)),
        warmup_operations=max(0, int(args.warmup)),
    )


def _generate_data(num_operations: int, window: int) -> np.ndarray:
    """Generate synthetic random walk tick data.

    Args:
        num_operations: Number of benchmark operations.
        window: Moving average window size.

    Returns:
        Float64 ndarray of length num_operations + window.
    """
    rng = np.random.default_rng(42)
    return np.cumsum(rng.standard_normal(num_operations + window))


def run_all_types(config: MovingAverageBenchmarkConfig) -> None:
    """Run benchmarks for all MA types and print comparative report.

    Args:
        config: Benchmark configuration.
    """
    data = _generate_data(config.num_operations, config.window)
    results: list[tuple[str, BenchmarkStatistics]] = []

    for ma_name, ma_class in MA_TYPES:
        benchmark = MovingAverageBenchmark(config, ma_class, ma_name, data)
        stats = benchmark.run()
        results.append((ma_name, stats))

        reporter = BenchmarkReporter(
            f"{ma_name} Moving Average Benchmark Results",
            {
                "Window": config.window,
                "Fast mode": config.is_fast,
                "Half life (s)": config.half_life_s,
            },
        )
        reporter.print_full_report(stats, warmup=config.warmup_operations)
        print()

    comp_reporter = ComparativeReporter(
        "Moving Average Comparative Summary (mean latency ns)",
        size_label="MA Type",
    )
    comp_reporter.print_comparative_table(
        results,
        ["initialize", "update", "next"],
    )


def main() -> None:
    """Main entry point."""
    cli = BenchmarkCLI("Benchmark moving average implementations")
    cli.parser.add_argument(
        "--window",
        type=int,
        default=100,
        help="Moving average window size (default: 100)",
    )
    cli.parser.add_argument(
        "--fast",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable fast mode (default: False)",
    )
    cli.parser.add_argument(
        "--half-life-s",
        type=float,
        default=1.0,
        help="TEMA half life in seconds (default: 1.0)",
    )
    cli.parser.add_argument(
        "--num-operations",
        type=int,
        default=None,
        help="Number of operations to benchmark (default: 100000)",
    )

    args = cli.parse()
    config = _build_config(args)
    run_all_types(config)


if __name__ == "__main__":
    main()
