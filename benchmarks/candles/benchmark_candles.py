"""Benchmarks for candles module.

Usage:
    uv run python benchmarks/candles/benchmark_candles.py

Measures the time to process a batch of synthetic trades through each of the
five candle aggregator types. A single trade batch is generated once and then
fed through a fresh candle instance on each measurement iteration.
"""

from __future__ import annotations

import argparse
import gc
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from benchmarks.core import (
        BaseBenchmarkConfig,
        BenchmarkCLI,
        BenchmarkReporter,
        BenchmarkRunner,
        ComparativeReporter,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from benchmarks.core import (
        BaseBenchmarkConfig,
        BenchmarkCLI,
        BenchmarkReporter,
        BenchmarkRunner,
        ComparativeReporter,
    )

from mm_toolbox.candles.base import Trade
from mm_toolbox.candles.multi import MultiCandles
from mm_toolbox.candles.price import PriceCandles
from mm_toolbox.candles.tick import TickCandles
from mm_toolbox.candles.time import TimeCandles
from mm_toolbox.candles.volume import VolumeCandles


@dataclass
class CandlesBenchmarkConfig(BaseBenchmarkConfig):
    """Configuration for candles benchmarking."""

    trade_count: int = 100_000
    num_candles: int = 1000
    store_trades: bool = True


def generate_trades(count: int) -> list[Trade]:
    """Generate realistic synthetic trades."""
    trades = []
    price = 50000.0
    time_ms = 0.0

    for _ in range(count):
        time_ms += random.randint(1, 100)
        price += random.gauss(0, 10.0)
        price = max(49000.0, min(51000.0, price))

        size = math.exp(random.gauss(0, 0.5))
        size = max(0.01, min(100.0, size))

        is_buy = random.random() < 0.5

        trades.append(
            Trade(
                time_ms=int(time_ms),
                is_buy=is_buy,
                price=price,
                size=size,
            )
        )

    return trades


class CandleBenchmarkRunner(BenchmarkRunner[CandlesBenchmarkConfig]):
    """Benchmark runner for a single candle type."""

    def __init__(
        self,
        config: CandlesBenchmarkConfig,
        name: str,
        candle_factory,
        trades: list[Trade],
    ) -> None:
        super().__init__(config)
        self.name = name
        self.candle_factory = candle_factory
        self.trades = trades

    def _create_subject(self):
        return None

    def _run_benchmark_suite(self, subject) -> None:
        trades = self.trades
        factory = self.candle_factory

        def op():
            candle = factory()
            for trade in trades:
                candle.process_trade(trade)

        self.measure_operation("process_trades", op)


def _print_throughput_summary(results, trade_count: int) -> None:
    print("\nThroughput Summary (per-trade metrics)")
    print("-" * 80)
    print(f"{'Candle Type':>20} {'ns/trade':>15} {'trades/sec':>15}")
    print("-" * 80)
    for name, stats in results:
        metrics = stats.get_operation("process_trades")
        if metrics:
            pcts = metrics.compute_percentiles()
            mean_ns = pcts["mean"]
            if mean_ns > 0:
                ns_per_trade = mean_ns / trade_count
                trades_per_sec = trade_count * 1e9 / mean_ns
                print(
                    f"{name:>20} {ns_per_trade:>15.2f} {trades_per_sec:>15,.0f}"
                )
    print("=" * 80)


def run_benchmarks(
    config: CandlesBenchmarkConfig,
) -> list[tuple[str, BenchmarkStatistics]]:
    random.seed(42)
    trades = generate_trades(config.trade_count)

    candle_configs = [
        (
            "TimeCandles",
            lambda: TimeCandles(
                secs_per_bucket=60.0,
                num_candles=config.num_candles,
                store_trades=config.store_trades,
            ),
        ),
        (
            "TickCandles",
            lambda: TickCandles(
                ticks_per_bucket=1000,
                num_candles=config.num_candles,
                store_trades=config.store_trades,
            ),
        ),
        (
            "VolumeCandles",
            lambda: VolumeCandles(
                volume_per_bucket=10000.0,
                num_candles=config.num_candles,
                store_trades=config.store_trades,
            ),
        ),
        (
            "PriceCandles",
            lambda: PriceCandles(
                price_bucket=50.0,
                num_candles=config.num_candles,
                store_trades=config.store_trades,
            ),
        ),
        (
            "MultiCandles",
            lambda: MultiCandles(
                max_duration_secs=30.0,
                max_ticks=500,
                max_size=5000.0,
                num_candles=config.num_candles,
                store_trades=config.store_trades,
            ),
        ),
    ]

    results = []

    for name, factory in candle_configs:
        print(f"\nBenchmarking {name}...")

        runner = CandleBenchmarkRunner(
            config=config,
            name=name,
            candle_factory=factory,
            trades=trades,
        )
        stats = runner.run()

        reporter = BenchmarkReporter(
            f"{name} Benchmark Results",
            {
                "trade_count": config.trade_count,
                "num_candles": config.num_candles,
                "store_trades": config.store_trades,
                "num_operations": config.num_operations,
            },
        )
        reporter.print_full_report(stats, warmup=config.warmup_operations)

        results.append((name, stats))
        gc.collect()

    return results


def main() -> None:
    cli = BenchmarkCLI("Benchmark candles module")

    cli.parser.add_argument(
        "--trade-count",
        type=int,
        default=100_000,
        help="Number of trades per batch (default: 100000)",
    )
    cli.parser.add_argument(
        "--num-candles",
        type=int,
        default=1000,
        help="Number of candles to store in ring buffer (default: 1000)",
    )
    cli.parser.add_argument(
        "--store-trades",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to store trades in candles (default: True)",
    )

    args = cli.parse()

    config = CandlesBenchmarkConfig(
        trade_count=args.trade_count,
        num_candles=args.num_candles,
        store_trades=args.store_trades,
        num_operations=args.operations,
        warmup_operations=args.warmup,
    )

    results = run_benchmarks(config)

    reporter = ComparativeReporter(
        "Comparative Summary (mean batch latency ns)",
        size_label="Candle Type",
    )
    reporter.print_comparative_table(
        results,
        key_operations=["process_trades"],
    )

    _print_throughput_summary(results, config.trade_count)


if __name__ == "__main__":
    main()
