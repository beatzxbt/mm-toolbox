"""Benchmarks Rounder scalar and vectorized rounding operations.

Usage:
    uv run python benchmarks/rounding/benchmark_rounding.py
    uv run python benchmarks/rounding/benchmark_rounding.py --multi-tick
    uv run python benchmarks/rounding/benchmark_rounding.py --multi-array

Measures:
- Scalar bid, ask, and size rounding
- Vectorized bids, asks, and sizes rounding (numpy arrays)
"""

from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from benchmarks.core import (
        BaseBenchmarkConfig,
        BenchmarkCLI,
        BenchmarkReporter,
        BenchmarkRunner,
        ComparativeReporter,
    )
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from benchmarks.core import (
        BaseBenchmarkConfig,
        BenchmarkCLI,
        BenchmarkReporter,
        BenchmarkRunner,
        ComparativeReporter,
    )

from mm_toolbox.rounding.rounder import Rounder, RounderConfig


@dataclass
class RoundingBenchmarkConfig(BaseBenchmarkConfig):
    """Configuration for Rounder benchmarking."""

    tick_size: float = 0.01
    lot_size: float = 0.001
    array_size: int = 64
    round_bids_down: bool = True
    round_asks_up: bool = True
    round_size_up: bool = True


class RoundingBenchmark(BenchmarkRunner[RoundingBenchmarkConfig]):
    """Benchmark runner for Rounder operations."""

    def __init__(self, config: RoundingBenchmarkConfig) -> None:
        super().__init__(config)
        self._scalar_prices: np.ndarray | None = None
        self._scalar_sizes: np.ndarray | None = None
        self._vector_prices: np.ndarray | None = None
        self._vector_sizes: np.ndarray | None = None
        self._price_index: int = 0
        self._size_index: int = 0

    def _create_subject(self) -> Rounder:
        """Create a Rounder instance with the benchmark configuration."""
        rounder_config = RounderConfig(
            tick_size=self.config.tick_size,
            lot_size=self.config.lot_size,
            round_bids_down=self.config.round_bids_down,
            round_asks_up=self.config.round_asks_up,
            round_size_up=self.config.round_size_up,
        )
        return Rounder(rounder_config)

    def _pre_generate_data(self) -> None:
        """Pre-generate random prices and sizes for benchmarks."""
        rng = np.random.default_rng(seed=42)

        self._scalar_prices = rng.uniform(1.0, 1000.0, size=self.config.num_operations)
        self._scalar_sizes = rng.uniform(0.001, 100.0, size=self.config.num_operations)

        self._vector_prices = rng.uniform(1.0, 1000.0, size=self.config.array_size)
        self._vector_sizes = rng.uniform(0.001, 100.0, size=self.config.array_size)

        self._price_index = 0
        self._size_index = 0

    def _next_price(self) -> float:
        """Get next scalar price value."""
        price = float(self._scalar_prices[self._price_index])
        self._price_index = (self._price_index + 1) % len(self._scalar_prices)
        return price

    def _next_size(self) -> float:
        """Get next scalar size value."""
        size = float(self._scalar_sizes[self._size_index])
        self._size_index = (self._size_index + 1) % len(self._scalar_sizes)
        return size

    def _run_benchmark_suite(self, subject: Rounder) -> None:
        """Run all benchmark operations."""
        self._pre_generate_data()

        self.measure_operation(
            "bid_scalar",
            lambda: subject.bid(self._next_price()),
        )

        self.measure_operation(
            "ask_scalar",
            lambda: subject.ask(self._next_price()),
        )

        self.measure_operation(
            "size_scalar",
            lambda: subject.size(self._next_size()),
        )

        prices_arr = self._vector_prices
        sizes_arr = self._vector_sizes

        self.measure_operation(
            "bids_vector",
            lambda: subject.bids(prices_arr),
        )

        self.measure_operation(
            "asks_vector",
            lambda: subject.asks(prices_arr),
        )

        self.measure_operation(
            "sizes_vector",
            lambda: subject.sizes(sizes_arr),
        )


def run_single(config: RoundingBenchmarkConfig) -> None:
    """Run one benchmark configuration and print a report."""
    benchmark = RoundingBenchmark(config)
    stats = benchmark.run()

    reporter = BenchmarkReporter(
        "Rounder Benchmark Results",
        {
            "Tick size": config.tick_size,
            "Lot size": config.lot_size,
            "Array size": config.array_size,
            "Round bids down": config.round_bids_down,
            "Round asks up": config.round_asks_up,
            "Round size up": config.round_size_up,
        },
    )
    reporter.print_full_report(stats, warmup=config.warmup_operations)


def run_multi_tick(args: argparse.Namespace) -> None:
    """Run tick-size comparison benchmark and print comparative summary."""
    tick_sizes = [1.0, 0.5, 0.01, 0.0001]
    results = []

    print("=" * 100)
    print("Rounder Tick Size Comparison")
    print("=" * 100)

    for tick_size in tick_sizes:
        print(f"\nTesting tick size: {tick_size}")
        config = RoundingBenchmarkConfig(
            tick_size=tick_size,
            lot_size=args.lot_size,
            array_size=args.array_size,
            round_bids_down=args.round_bids_down,
            round_asks_up=args.round_asks_up,
            round_size_up=args.round_size_up,
            num_operations=args.num_operations,
            warmup_operations=args.warmup,
        )

        benchmark = RoundingBenchmark(config)
        stats = benchmark.run()
        results.append((tick_size, stats))

        gc.collect()

    reporter = ComparativeReporter(
        "Comparative Summary (mean latency ns)",
        size_label="Tick size",
    )
    reporter.print_comparative_table(
        results,
        [
            "bid_scalar",
            "ask_scalar",
            "size_scalar",
            "bids_vector",
            "asks_vector",
            "sizes_vector",
        ],
    )


def run_multi_array(args: argparse.Namespace) -> None:
    """Run array-size comparison benchmark and print comparative summary."""
    array_sizes = [1, 4, 16, 64]
    results = []

    print("=" * 100)
    print("Rounder Array Size Comparison")
    print("=" * 100)

    for array_size in array_sizes:
        print(f"\nTesting array size: {array_size}")
        config = RoundingBenchmarkConfig(
            tick_size=args.tick_size,
            lot_size=args.lot_size,
            array_size=array_size,
            round_bids_down=args.round_bids_down,
            round_asks_up=args.round_asks_up,
            round_size_up=args.round_size_up,
            num_operations=args.num_operations,
            warmup_operations=args.warmup,
        )

        benchmark = RoundingBenchmark(config)
        stats = benchmark.run()
        results.append((array_size, stats))

        gc.collect()

    reporter = ComparativeReporter(
        "Comparative Summary (mean latency ns)",
        size_label="Array size",
    )
    reporter.print_comparative_table(
        results,
        [
            "bid_scalar",
            "ask_scalar",
            "size_scalar",
            "bids_vector",
            "asks_vector",
            "sizes_vector",
        ],
    )


def main() -> None:
    """Main entry point."""
    cli = BenchmarkCLI("Benchmark Rounder scalar and vectorized operations")

    cli.parser.add_argument(
        "--tick-size",
        type=float,
        default=0.01,
        help="Tick size for price rounding (default: 0.01)",
    )
    cli.parser.add_argument(
        "--lot-size",
        type=float,
        default=0.001,
        help="Lot size for size rounding (default: 0.001)",
    )
    cli.parser.add_argument(
        "--array-size",
        type=int,
        default=64,
        choices=[1, 4, 16, 64],
        help="Array size for vectorized operations (default: 64)",
    )
    cli.parser.add_argument(
        "--round-bids-down",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Round bids down (default: True)",
    )
    cli.parser.add_argument(
        "--round-asks-up",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Round asks up (default: True)",
    )
    cli.parser.add_argument(
        "--round-size-up",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Round sizes up (default: True)",
    )
    cli.parser.add_argument(
        "--num-operations",
        type=int,
        default=100_000,
        help="Number of operations to benchmark (default: 100,000)",
    )
    cli.parser.add_argument(
        "--multi-tick",
        action="store_true",
        help="Run across tick sizes [1.0, 0.5, 0.01, 0.0001] and print comparative table",
    )
    cli.parser.add_argument(
        "--multi-array",
        action="store_true",
        help="Run across array sizes [1, 4, 16, 64] and print comparative table",
    )

    args = cli.parse()

    if args.multi_tick:
        run_multi_tick(args)
        return

    if args.multi_array:
        run_multi_array(args)
        return

    config = RoundingBenchmarkConfig(
        tick_size=args.tick_size,
        lot_size=args.lot_size,
        array_size=args.array_size,
        round_bids_down=args.round_bids_down,
        round_asks_up=args.round_asks_up,
        round_size_up=args.round_size_up,
        num_operations=args.num_operations,
        warmup_operations=args.warmup,
    )
    run_single(config)


if __name__ == "__main__":
    main()
