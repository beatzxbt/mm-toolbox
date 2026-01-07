"""Benchmarks Standard Orderbook performance with collected Binance data.

Usage:
    uv run python benchmarks/orderbook/benchmark_standard_orderbook.py [--input FILE]

Processes collected orderbook messages through Standard Orderbook and measures:
- Per-operation latency (nanoseconds)
- Throughput (operations/second)
- Level change statistics
"""

from __future__ import annotations

import argparse
import gc
import time
from dataclasses import dataclass, field
from pathlib import Path

import msgspec
import numpy as np

from mm_toolbox.orderbook.standard import Orderbook, OrderbookLevel


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark run."""

    input_path: str
    tick_size: float = 0.01
    lot_size: float = 0.001
    warmup_messages: int = 100


@dataclass
class OperationMetric:
    """Metrics for a single operation.

    Args:
        msg_type: Type of message (snapshot, delta, bbo).
        latency_ns: Processing latency in nanoseconds.
        num_levels_consumed: Total levels in incoming message (asks + bids).
    """

    msg_type: str
    latency_ns: int
    num_levels_consumed: int


@dataclass
class BenchmarkStats:
    """Aggregated benchmark statistics."""

    snapshot_latencies: list[int] = field(default_factory=list)
    snapshot_levels: list[int] = field(default_factory=list)

    bbo_latencies: list[int] = field(default_factory=list)
    bbo_levels: list[int] = field(default_factory=list)

    delta_latencies: list[int] = field(default_factory=list)
    delta_levels: list[int] = field(default_factory=list)

    total_messages: int = 0
    total_time_ns: int = 0

    def add_metric(self, metric: OperationMetric) -> None:
        """Add a single operation metric.

        Args:
            metric: Operation metric to add.
        """
        if metric.msg_type == "snapshot":
            self.snapshot_latencies.append(metric.latency_ns)
            self.snapshot_levels.append(metric.num_levels_consumed)
        elif metric.msg_type == "bbo":
            self.bbo_latencies.append(metric.latency_ns)
            self.bbo_levels.append(metric.num_levels_consumed)
        elif metric.msg_type == "delta":
            self.delta_latencies.append(metric.latency_ns)
            self.delta_levels.append(metric.num_levels_consumed)

        self.total_messages += 1

    def _compute_percentiles(self, data: list[int] | list[float]) -> dict[str, float]:
        """Compute percentile statistics.

        Args:
            data: List of values.

        Returns:
            Dictionary with mean, p10, p25, p50, p95, p99, p99.9.
        """
        if not data:
            return {
                "count": 0,
                "mean": 0.0,
                "p10": 0.0,
                "p25": 0.0,
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "p99_9": 0.0,
            }

        arr = np.array(data, dtype=np.float64)
        return {
            "count": len(data),
            "mean": float(np.mean(arr)),
            "p10": float(np.percentile(arr, 10)),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "p99_9": float(np.percentile(arr, 99.9)),
        }

    def print_report(self, input_path: str, warmup: int) -> None:
        """Print formatted benchmark report.

        Args:
            input_path: Path to input file.
            warmup: Number of warmup messages skipped.
        """
        total_time_s = self.total_time_ns / 1e9
        throughput = self.total_messages / total_time_s if total_time_s > 0 else 0

        print("=" * 80)
        print("Standard Orderbook Benchmark Results")
        print("=" * 80)
        print(f"Input: {input_path}")
        print(
            f"Messages: {self.total_messages + warmup} "
            f"(warmup: {warmup}, measured: {self.total_messages})"
        )
        print(f"Total time: {total_time_s:.3f}s | Throughput: {throughput:.0f} ops/sec")
        print()

        print("Latency by Operation Type (nanoseconds per level)")
        print("-" * 80)
        header = f"{'':>10} {'Count':>6} {'Mean':>8} {'P10':>8} {'P25':>8} {'P50':>8} {'P95':>8} {'P99':>8} {'P99.9':>8}"
        print(header)

        for name, latencies, levels in [
            ("Snapshot", self.snapshot_latencies, self.snapshot_levels),
            ("BBO", self.bbo_latencies, self.bbo_levels),
            ("Delta", self.delta_latencies, self.delta_levels),
        ]:
            if not latencies:
                continue

            ns_per_level = [
                lat / lvl for lat, lvl in zip(latencies, levels) if lvl > 0 and lat >= 0
            ]

            if not ns_per_level:
                continue

            stats = self._compute_percentiles(ns_per_level)
            print(
                f"{name:>10} {stats['count']:>6} {stats['mean']:>8.1f} "
                f"{stats['p10']:>8.1f} {stats['p25']:>8.1f} {stats['p50']:>8.1f} "
                f"{stats['p95']:>8.1f} {stats['p99']:>8.1f} {stats['p99_9']:>8.1f}",
                flush=True,
            )

        print()

        if self.delta_latencies:
            print("Delta Processing by Level Count (non-linearity analysis)")
            print("Note: Level count refers to levels in incoming messages")
            print("-" * 80)

            max_levels_consumed = max(self.delta_levels) if self.delta_levels else 0

            level_ranges = [
                (1, 5),
                (6, 10),
                (11, 20),
                (21, 50),
                (51, 100),
                (101, 500),
                (501, float("inf")),
            ]

            header = (
                f"{'Level Range':>15} {'Count':>8} {'Mean ns':>12} {'ns/Level':>12}"
            )
            print(header, flush=True)

            for min_lvl, max_lvl in level_ranges:
                try:
                    range_data = [
                        (lat, lvl)
                        for lat, lvl in zip(self.delta_latencies, self.delta_levels)
                        if min_lvl <= lvl <= max_lvl
                    ]

                    if not range_data:
                        continue

                    latencies_in_range = [lat for lat, _ in range_data]
                    ns_per_level_in_range = [
                        lat / lvl for lat, lvl in range_data if lvl > 0
                    ]

                    count = len(latencies_in_range)
                    mean_ns = np.mean(latencies_in_range)
                    mean_ns_per_level = (
                        np.mean(ns_per_level_in_range) if ns_per_level_in_range else 0
                    )

                    if max_lvl == float("inf"):
                        range_str = f"{min_lvl}-{max_levels_consumed}"
                    else:
                        range_str = f"{min_lvl}-{max_lvl}"
                    print(
                        f"{range_str:>15} {count:>8} {mean_ns:>12.0f} {mean_ns_per_level:>12.1f}",
                        flush=True,
                    )
                except Exception as e:
                    print(
                        f"Error processing range {min_lvl}-{max_lvl}: {e}",
                        flush=True,
                    )
                    continue

        print("=" * 80, flush=True)


class StandardOrderbookBenchmark:
    """Benchmark runner for Standard Orderbook."""

    def __init__(self, config: BenchmarkConfig) -> None:
        """Initialize the benchmark.

        Args:
            config: Benchmark configuration.
        """
        self.config = config
        self.orderbook: Orderbook | None = None
        self.stats = BenchmarkStats()
        self.decoder = msgspec.json.Decoder()
        self._reset()

    def _reset(self) -> None:
        """Reset the benchmark with a new orderbook.

        Explicitly clears and deletes the old orderbook, then creates a new one.
        """
        if self.orderbook is not None:
            del self.orderbook
            self.orderbook = None

        gc.collect()

        self.orderbook = Orderbook(
            tick_size=self.config.tick_size,
            lot_size=self.config.lot_size,
        )
        self.stats = BenchmarkStats()

    def _parse_levels_to_list(self, levels: list[list[str]]) -> list[OrderbookLevel]:
        """Parse Binance level format to OrderbookLevel list.

        Args:
            levels: List of [price, size] string pairs.

        Returns:
            List of OrderbookLevel objects.
        """
        if not levels:
            return []

        return [
            OrderbookLevel.from_values(
                price=float(level[0]),
                size=float(level[1]),
                norders=1,
                tick_size=self.config.tick_size,
                lot_size=self.config.lot_size,
            )
            for level in levels
        ]

    def _process_snapshot(self, data: dict) -> OperationMetric:
        """Process a snapshot message.

        Args:
            data: Snapshot data with bids and asks.

        Returns:
            Operation metric.

        Raises:
            RuntimeError: If snapshot processing fails or has empty side.
        """
        asks = self._parse_levels_to_list(data["asks"])
        bids = self._parse_levels_to_list(data["bids"])

        if len(asks) == 0 or len(bids) == 0:
            raise RuntimeError(
                f"Snapshot has empty side: asks={len(asks)}, bids={len(bids)}"
            )

        num_levels = len(asks) + len(bids)

        start = time.perf_counter_ns()
        try:
            # CRITICAL: Standard uses (bids, asks) order, not (asks, bids)!
            self.orderbook.consume_snapshot(bids, asks)
        except Exception as e:
            raise RuntimeError(
                f"Failed to process snapshot: {e}. "
                f"Snapshot: {len(asks)} asks, {len(bids)} bids"
            ) from e
        elapsed = time.perf_counter_ns() - start

        return OperationMetric(
            msg_type="snapshot",
            latency_ns=elapsed,
            num_levels_consumed=num_levels,
        )

    def _process_delta(self, data: dict) -> OperationMetric:
        """Process a delta message.

        Binance delta format uses 'a' for asks and 'b' for bids.

        Args:
            data: Delta data with bid and ask updates.

        Returns:
            Operation metric.

        Raises:
            RuntimeError: If delta processing fails.
        """
        asks = self._parse_levels_to_list(data.get("a", []))
        bids = self._parse_levels_to_list(data.get("b", []))

        num_levels = len(asks) + len(bids)

        start = time.perf_counter_ns()
        try:
            # CRITICAL: Standard uses (bids, asks) order, not (asks, bids)!
            self.orderbook.consume_deltas(bids, asks)
        except Exception as e:
            raise RuntimeError(
                f"Failed to process delta: {e}. "
                f"Delta: {len(asks)} asks, {len(bids)} bids"
            ) from e
        elapsed = time.perf_counter_ns() - start

        return OperationMetric(
            msg_type="delta",
            latency_ns=elapsed,
            num_levels_consumed=num_levels,
        )

    def _process_bbo(self, data: dict) -> OperationMetric:
        """Process a BBO message.

        Binance BBO format: b=bid_price, B=bid_qty, a=ask_price, A=ask_qty.
        BBO always contains exactly 2 levels (1 bid + 1 ask).

        Args:
            data: BBO data with best bid/ask.

        Returns:
            Operation metric.
        """
        bid_level = OrderbookLevel.from_values(
            price=float(data["b"]),
            size=float(data["B"]),
            norders=1,
            tick_size=self.config.tick_size,
            lot_size=self.config.lot_size,
        )
        ask_level = OrderbookLevel.from_values(
            price=float(data["a"]),
            size=float(data["A"]),
            norders=1,
            tick_size=self.config.tick_size,
            lot_size=self.config.lot_size,
        )

        num_levels = 2

        start = time.perf_counter_ns()
        # CRITICAL: Standard uses consume_bbo(bid, ask), not consume_bbo(ask, bid)!
        self.orderbook.consume_bbo(bid_level, ask_level)
        elapsed = time.perf_counter_ns() - start

        return OperationMetric(
            msg_type="bbo",
            latency_ns=elapsed,
            num_levels_consumed=num_levels,
        )

    def _load_messages(self) -> list[dict]:
        """Load messages from input file.

        Returns:
            List of decoded messages.

        Raises:
            FileNotFoundError: If input file doesn't exist.
        """
        input_path = Path(self.config.input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        messages: list[dict] = []
        with open(input_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    messages.append(self.decoder.decode(line))

        return messages

    def _run_benchmark(self, messages: list[dict], print_progress: bool = True) -> None:
        """Run the benchmark with pre-loaded messages.

        Args:
            messages: Pre-loaded list of messages.
            print_progress: Whether to print progress messages.
        """
        if print_progress:
            print(f"Loaded {len(messages)} messages")
            print(f"Warmup: {self.config.warmup_messages} messages")
            print("Running benchmark...")

        snapshot_processed = False
        warmup_count = 0

        for i, msg in enumerate(messages):
            msg_type = msg["type"]
            data = msg["data"]

            if not snapshot_processed and msg_type == "snapshot":
                try:
                    self._process_snapshot(data)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to process initial snapshot at message {i}: {e}"
                    ) from e
                snapshot_processed = True
                continue

            if warmup_count < self.config.warmup_messages:
                try:
                    if msg_type == "snapshot":
                        self._process_snapshot(data)
                    elif msg_type == "delta":
                        self._process_delta(data)
                    elif msg_type == "bbo":
                        self._process_bbo(data)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to process warmup message {i} (type: {msg_type}): {e}"
                    ) from e
                warmup_count += 1
                continue

            break

        start_idx = 1 + warmup_count

        benchmark_start = time.perf_counter_ns()
        total_to_process = len(messages) - start_idx
        processed_count = 0
        last_progress_time = time.perf_counter_ns()

        for msg_idx, msg in enumerate(messages[start_idx:], start=start_idx):
            msg_type = msg["type"]
            data = msg["data"]

            try:
                if msg_type == "snapshot":
                    metric = self._process_snapshot(data)
                elif msg_type == "delta":
                    metric = self._process_delta(data)
                elif msg_type == "bbo":
                    metric = self._process_bbo(data)
                else:
                    continue

                self.stats.add_metric(metric)
                processed_count += 1

                current_time = time.perf_counter_ns()
                if (
                    processed_count % max(1, total_to_process // 10) == 0
                    or (current_time - last_progress_time) > 5e9
                ):
                    if print_progress:
                        progress_pct = (processed_count / total_to_process) * 100
                        print(
                            f"Progress: {processed_count}/{total_to_process} "
                            f"({progress_pct:.1f}%) messages processed",
                            flush=True,
                        )
                    last_progress_time = current_time
            except Exception as e:
                raise RuntimeError(
                    f"Failed to process message {msg_idx} (type: {msg_type}): {e}"
                ) from e

        benchmark_end = time.perf_counter_ns()
        self.stats.total_time_ns = benchmark_end - benchmark_start

        if print_progress:
            print(
                f"\nStats collected: "
                f"snapshot={len(self.stats.snapshot_latencies)}, "
                f"bbo={len(self.stats.bbo_latencies)}, "
                f"delta={len(self.stats.delta_latencies)}, "
                f"total={self.stats.total_messages}",
                flush=True,
            )

    def run(self, print_report: bool = True) -> None:
        """Run the benchmark.

        Args:
            print_report: Whether to print the detailed report after running.
        """
        input_path = Path(self.config.input_path)
        print(f"Loading data from {input_path}...")

        messages = self._load_messages()
        self._run_benchmark(messages)

        if print_report:
            print()
            self.stats.print_report(str(input_path), self.config.warmup_messages)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark Standard Orderbook with collected Binance data"
    )
    parser.add_argument(
        "--input",
        "-i",
        default="benchmarks/orderbook/data/btcusdt_100k.jsonl",
        help="Input file path (default: benchmarks/orderbook/data/btcusdt_100k.jsonl)",
    )
    parser.add_argument(
        "--warmup",
        "-w",
        type=int,
        default=100,
        help="Number of warmup messages to skip (default: 100)",
    )

    args = parser.parse_args()

    config = BenchmarkConfig(
        input_path=args.input,
        warmup_messages=args.warmup,
    )

    benchmark = StandardOrderbookBenchmark(config)

    try:
        benchmark.run()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run collect_data.py first to generate the input file.")
    except KeyboardInterrupt:
        print("\nBenchmark interrupted.")


if __name__ == "__main__":
    main()
