"""Benchmarks AdvancedOrderbook performance with collected Binance data.

Usage:
    uv run python benchmarks/orderbook/benchmark_orderbook.py [--input FILE]

Processes collected orderbook messages through AdvancedOrderbook and measures:
- Per-operation latency (nanoseconds)
- Throughput (operations/second)
- Level change statistics
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from pathlib import Path

import msgspec
import numpy as np
import numpy.typing as npt

try:
    from benchmarks.core import (
        BaseBenchmarkConfig,
        BenchmarkCLI,
        BenchmarkRunner,
        BenchmarkStatistics,
        ComparativeReporter,
    )
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from benchmarks.core import (
        BaseBenchmarkConfig,
        BenchmarkCLI,
        BenchmarkRunner,
        BenchmarkStatistics,
        ComparativeReporter,
    )

from mm_toolbox.orderbook.advanced import (
    AdvancedOrderbook,
    OrderbookLevel,
    OrderbookSortedness,
)


@dataclass
class AdvancedOrderbookBenchmarkConfig(BaseBenchmarkConfig):
    """Configuration for AdvancedOrderbook benchmark run."""

    input_path: str = "benchmarks/orderbook/data/btcusdt_100k.jsonl"
    tick_size: float = 0.01
    lot_size: float = 0.001
    num_levels: int = 2048


@dataclass
class MultiSizeSummary:
    """Summary statistics for a single orderbook size."""

    num_levels: int
    throughput: float
    snapshot_mean_ns: float
    delta_mean_ns: float
    bbo_mean_ns: float


def _compute_percentiles(data: list[int] | list[float]) -> dict[str, float]:
    """Compute percentile statistics."""
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


def _operation_latencies(stats: BenchmarkStatistics, name: str) -> list[int]:
    """Get raw latency series for an operation."""
    metrics = stats.get_operation(name)
    return metrics.latencies_ns if metrics else []


def _operation_levels(stats: BenchmarkStatistics, name: str) -> list[int]:
    """Get consumed-level metadata for an operation."""
    metrics = stats.get_operation(name)
    if metrics is None:
        return []
    levels = metrics.metadata.get("num_levels_consumed", [])
    return [int(level) for level in levels]


def _ns_per_level(stats: BenchmarkStatistics, name: str) -> list[float]:
    """Get ns/level values for an operation."""
    latencies = _operation_latencies(stats, name)
    levels = _operation_levels(stats, name)
    return [lat / lvl for lat, lvl in zip(latencies, levels) if lvl > 0 and lat >= 0]


def _print_detailed_report(
    stats: BenchmarkStatistics,
    input_path: str,
    warmup_messages: int,
    num_levels: int,
) -> None:
    """Print detailed benchmark report."""
    total_time_s = stats.total_time_ns / 1e9
    throughput = stats.total_operations / total_time_s if total_time_s > 0 else 0

    print("=" * 80)
    print("AdvancedOrderbook Benchmark Results")
    print("=" * 80)
    print(f"Orderbook Size: {num_levels} levels")
    print(f"Input: {input_path}")
    print(
        f"Messages: {stats.total_operations + warmup_messages} "
        f"(warmup: {warmup_messages}, measured: {stats.total_operations})"
    )
    print(f"Total time: {total_time_s:.3f}s | Throughput: {throughput:.0f} ops/sec")
    print()

    print("Latency by Operation Type (nanoseconds per level)")
    print("-" * 80)
    header = f"{'':>10} {'Count':>6} {'Mean':>8} {'P10':>8} {'P25':>8} {'P50':>8} {'P95':>8} {'P99':>8} {'P99.9':>8}"
    print(header)

    for label, name in [("Snapshot", "snapshot"), ("BBO", "bbo"), ("Delta", "delta")]:
        values = _ns_per_level(stats, name)
        if not values:
            continue

        pcts = _compute_percentiles(values)
        print(
            f"{label:>10} {int(pcts['count']):>6} {pcts['mean']:>8.1f} "
            f"{pcts['p10']:>8.1f} {pcts['p25']:>8.1f} {pcts['p50']:>8.1f} "
            f"{pcts['p95']:>8.1f} {pcts['p99']:>8.1f} {pcts['p99_9']:>8.1f}",
            flush=True,
        )

    print()

    delta_latencies = _operation_latencies(stats, "delta")
    delta_levels = _operation_levels(stats, "delta")

    if delta_latencies:
        print("Delta Processing by Level Count (non-linearity analysis)")
        print("Note: Level count refers to levels in incoming messages, not orderbook capacity")
        print("-" * 80)

        max_levels_consumed = max(delta_levels) if delta_levels else 0

        level_ranges = [
            (1, 5),
            (6, 10),
            (11, 20),
            (21, 50),
            (51, 100),
            (101, 500),
            (501, float("inf")),
        ]

        header = f"{'Level Range':>15} {'Count':>8} {'Mean ns':>12} {'ns/Level':>12}"
        print(header, flush=True)

        for min_lvl, max_lvl in level_ranges:
            try:
                range_data = [
                    (lat, lvl)
                    for lat, lvl in zip(delta_latencies, delta_levels)
                    if min_lvl <= lvl <= max_lvl
                ]

                if not range_data:
                    continue

                latencies_in_range = [lat for lat, _ in range_data]
                ns_per_level_in_range = [lat / lvl for lat, lvl in range_data if lvl > 0]

                count = len(latencies_in_range)
                mean_ns = float(np.mean(latencies_in_range))
                mean_ns_per_level = (
                    float(np.mean(ns_per_level_in_range)) if ns_per_level_in_range else 0.0
                )

                range_str = (
                    f"{min_lvl}-{max_levels_consumed}"
                    if max_lvl == float("inf")
                    else f"{min_lvl}-{max_lvl}"
                )
                print(
                    f"{range_str:>15} {count:>8} {mean_ns:>12.0f} {mean_ns_per_level:>12.1f}",
                    flush=True,
                )
            except Exception as exc:
                print(f"Error processing range {min_lvl}-{max_lvl}: {exc}", flush=True)
                continue

    print("=" * 80, flush=True)


def _summary_for_size(stats: BenchmarkStatistics, num_levels: int) -> MultiSizeSummary:
    """Build comparative summary row for one size."""
    total_time_s = stats.total_time_ns / 1e9
    throughput = stats.total_operations / total_time_s if total_time_s > 0 else 0

    snapshot_stats = _compute_percentiles(_operation_latencies(stats, "snapshot"))
    delta_stats = _compute_percentiles(_operation_latencies(stats, "delta"))
    bbo_stats = _compute_percentiles(_operation_latencies(stats, "bbo"))

    return MultiSizeSummary(
        num_levels=num_levels,
        throughput=throughput,
        snapshot_mean_ns=snapshot_stats["mean"],
        delta_mean_ns=delta_stats["mean"],
        bbo_mean_ns=bbo_stats["mean"],
    )


def print_comparative_summary(summaries: list[MultiSizeSummary]) -> None:
    """Print comparative summary table across multiple orderbook sizes."""
    print("=" * 80)
    print("Comparative Summary: Performance by Orderbook Size")
    print("=" * 80)
    header = (
        f"{'Size':>8} {'Throughput':>12} {'Snapshot Mean':>15} "
        f"{'Delta Mean':>15} {'BBO Mean':>15}"
    )
    print(header)
    print("-" * 80)

    for summary in summaries:
        print(
            f"{summary.num_levels:>8} {summary.throughput:>12.0f} "
            f"{summary.snapshot_mean_ns:>15.0f} {summary.delta_mean_ns:>15.0f} "
            f"{summary.bbo_mean_ns:>15.0f}"
        )

    print("=" * 80)
    print()


class AdvancedOrderbookBenchmark(BenchmarkRunner[AdvancedOrderbookBenchmarkConfig]):
    """Benchmark runner for AdvancedOrderbook."""

    def __init__(self, config: AdvancedOrderbookBenchmarkConfig) -> None:
        """Initialize benchmark runner.

        Args:
            config: Benchmark configuration.
        """
        super().__init__(config)
        self.decoder = msgspec.json.Decoder()
        self._messages_override: list[dict] | None = None

    def set_messages(self, messages: list[dict]) -> None:
        """Set pre-loaded messages for reuse across runs."""
        self._messages_override = messages

    def _create_subject(self) -> AdvancedOrderbook:
        """Create AdvancedOrderbook instance."""
        return AdvancedOrderbook(
            tick_size=self.config.tick_size,
            lot_size=self.config.lot_size,
            num_levels=self.config.num_levels,
            delta_sortedness=OrderbookSortedness.BIDS_DESCENDING_ASKS_ASCENDING,
            snapshot_sortedness=OrderbookSortedness.BIDS_DESCENDING_ASKS_ASCENDING,
        )

    def _parse_levels_to_numpy(
        self, levels: list[list[str]]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Parse Binance level format to numpy arrays."""
        if not levels:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

        prices = np.array([float(level[0]) for level in levels], dtype=np.float64)
        sizes = np.array([float(level[1]) for level in levels], dtype=np.float64)
        return prices, sizes

    def _process_snapshot(self, orderbook: AdvancedOrderbook, data: dict) -> tuple[int, int]:
        """Process snapshot and return latency + consumed level count."""
        ask_prices, ask_sizes = self._parse_levels_to_numpy(data["asks"])
        bid_prices, bid_sizes = self._parse_levels_to_numpy(data["bids"])

        max_levels = self.config.num_levels
        if len(ask_prices) > max_levels:
            ask_prices = ask_prices[:max_levels]
            ask_sizes = ask_sizes[:max_levels]
        if len(bid_prices) > max_levels:
            bid_prices = bid_prices[:max_levels]
            bid_sizes = bid_sizes[:max_levels]

        if len(ask_prices) == 0 or len(bid_prices) == 0:
            raise RuntimeError(
                f"Snapshot has empty side: asks={len(ask_prices)}, bids={len(bid_prices)}"
            )

        num_levels = len(ask_prices) + len(bid_prices)

        start = time.perf_counter_ns()
        try:
            orderbook.consume_snapshot_numpy(ask_prices, ask_sizes, bid_prices, bid_sizes)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to process snapshot: {exc}. "
                f"Orderbook size: {max_levels}, "
                f"Snapshot: {len(ask_prices)} asks, {len(bid_prices)} bids"
            ) from exc
        elapsed = time.perf_counter_ns() - start

        return elapsed, num_levels

    def _process_delta(self, orderbook: AdvancedOrderbook, data: dict) -> tuple[int, int]:
        """Process delta and return latency + consumed level count."""
        ask_prices, ask_sizes = self._parse_levels_to_numpy(data.get("a", []))
        bid_prices, bid_sizes = self._parse_levels_to_numpy(data.get("b", []))

        num_levels = len(ask_prices) + len(bid_prices)

        start = time.perf_counter_ns()
        try:
            orderbook.consume_deltas_numpy(ask_prices, ask_sizes, bid_prices, bid_sizes)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to process delta: {exc}. "
                f"Orderbook size: {self.config.num_levels}, "
                f"Delta: {len(ask_prices)} asks, {len(bid_prices)} bids"
            ) from exc
        elapsed = time.perf_counter_ns() - start

        return elapsed, num_levels

    def _process_bbo(self, orderbook: AdvancedOrderbook, data: dict) -> tuple[int, int]:
        """Process BBO and return latency + consumed level count."""
        bid_level = OrderbookLevel(
            price=float(data["b"]),
            size=float(data["B"]),
        )
        ask_level = OrderbookLevel(
            price=float(data["a"]),
            size=float(data["A"]),
        )

        start = time.perf_counter_ns()
        orderbook.consume_bbo(ask_level, bid_level)
        elapsed = time.perf_counter_ns() - start

        return elapsed, 2

    def _load_messages(self) -> list[dict]:
        """Load messages from input file or override."""
        if self._messages_override is not None:
            return self._messages_override

        input_path = Path(self.config.input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        messages: list[dict] = []
        with open(input_path) as file:
            for line in file:
                line = line.strip()
                if line:
                    messages.append(self.decoder.decode(line))

        return messages

    def _record_metric(self, operation: str, latency_ns: int, num_levels_consumed: int) -> None:
        """Record one operation sample."""
        metrics = self.stats.get_operation(operation)
        if metrics is None:
            metrics = self.stats.add_operation(operation)
        metrics.add_latency(latency_ns, num_levels_consumed=num_levels_consumed)

    def _run_benchmark_suite(self, orderbook: AdvancedOrderbook) -> None:
        """Run benchmark and populate shared BenchmarkStatistics."""
        messages = self._load_messages()

        print(f"Loaded {len(messages)} messages")
        print(f"Warmup: {self.config.warmup_operations} messages")
        print("Running benchmark...")

        snapshot_processed = False
        warmup_count = 0

        for idx, msg in enumerate(messages):
            msg_type = msg["type"]
            data = msg["data"]

            if not snapshot_processed and msg_type == "snapshot":
                try:
                    self._process_snapshot(orderbook, data)
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to process initial snapshot at message {idx}: {exc}"
                    ) from exc
                snapshot_processed = True
                continue

            if warmup_count < self.config.warmup_operations:
                try:
                    if msg_type == "snapshot":
                        self._process_snapshot(orderbook, data)
                    elif msg_type == "delta":
                        self._process_delta(orderbook, data)
                    elif msg_type == "bbo":
                        self._process_bbo(orderbook, data)
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to process warmup message {idx} (type: {msg_type}): {exc}"
                    ) from exc
                warmup_count += 1
                continue

            break

        start_idx = 1 + warmup_count

        benchmark_start = time.perf_counter_ns()
        total_to_process = len(messages) - start_idx
        processed_count = 0
        last_progress_time = time.perf_counter_ns()

        max_measured_messages = self.config.num_operations

        for msg_idx, msg in enumerate(messages[start_idx:], start=start_idx):
            if max_measured_messages > 0 and processed_count >= max_measured_messages:
                break

            msg_type = msg["type"]
            data = msg["data"]

            try:
                if msg_type == "snapshot":
                    latency_ns, num_levels = self._process_snapshot(orderbook, data)
                    self._record_metric("snapshot", latency_ns, num_levels)
                elif msg_type == "delta":
                    latency_ns, num_levels = self._process_delta(orderbook, data)
                    self._record_metric("delta", latency_ns, num_levels)
                elif msg_type == "bbo":
                    latency_ns, num_levels = self._process_bbo(orderbook, data)
                    self._record_metric("bbo", latency_ns, num_levels)
                else:
                    continue

                processed_count += 1

                current_time = time.perf_counter_ns()
                if (
                    processed_count % max(1, max(1, total_to_process) // 10) == 0
                    or (current_time - last_progress_time) > 5e9
                ):
                    progress_pct = (
                        (processed_count / total_to_process) * 100
                        if total_to_process > 0
                        else 100.0
                    )
                    print(
                        f"Progress: {processed_count}/{total_to_process} "
                        f"({progress_pct:.1f}%) messages processed",
                        flush=True,
                    )
                    last_progress_time = current_time
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to process message {msg_idx} (type: {msg_type}): {exc}"
                ) from exc

        benchmark_end = time.perf_counter_ns()
        self.stats.total_time_ns = benchmark_end - benchmark_start

        print(
            "\nStats collected: "
            f"snapshot={len(_operation_latencies(self.stats, 'snapshot'))}, "
            f"bbo={len(_operation_latencies(self.stats, 'bbo'))}, "
            f"delta={len(_operation_latencies(self.stats, 'delta'))}, "
            f"total={self.stats.total_operations}",
            flush=True,
        )


def _stats_copy(stats: BenchmarkStatistics) -> BenchmarkStatistics:
    """Deep-copy benchmark stats for later reporting."""
    copied = BenchmarkStatistics()
    copied.total_time_ns = stats.total_time_ns

    for name, metrics in stats.operations.items():
        target = copied.add_operation(name)
        target.latencies_ns = metrics.latencies_ns.copy()
        target.metadata = {
            key: value.copy() if isinstance(value, list) else list(value)
            for key, value in metrics.metadata.items()
        }

    return copied


def _build_config_from_args(args) -> AdvancedOrderbookBenchmarkConfig:
    """Build config from CLI args."""
    max_messages = 0 if args.operations == 100_000 else args.operations

    return AdvancedOrderbookBenchmarkConfig(
        input_path=args.input,
        warmup_operations=args.warmup,
        num_operations=max_messages,
        num_levels=args.levels,
    )


def _run_single(config: AdvancedOrderbookBenchmarkConfig) -> None:
    """Run single-size benchmark."""
    benchmark = AdvancedOrderbookBenchmark(config)

    try:
        input_path = Path(config.input_path)
        print(f"Loading data from {input_path}...")
        stats = benchmark.run()
        print()
        _print_detailed_report(stats, str(input_path), config.warmup_operations, config.num_levels)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        print("Run collect_data.py first to generate the input file.")


def _run_multi_size(base_config: AdvancedOrderbookBenchmarkConfig) -> None:
    """Run benchmark across multiple orderbook sizes."""
    sizes = [2**i for i in range(6, 13, 2)]
    summaries: list[MultiSizeSummary] = []
    all_stats: list[tuple[int, BenchmarkStatistics]] = []

    try:
        input_path = Path(base_config.input_path)
        print(f"Loading data from {input_path}...")

        loader = AdvancedOrderbookBenchmark(base_config)
        messages = loader._load_messages()
        del loader
        gc.collect()

        print(f"Loaded {len(messages)} messages\n")

        comparative_rows: list[tuple[int, BenchmarkStatistics]] = []

        for size in sizes:
            print(f"{'=' * 80}")
            print(f"Testing orderbook size: {size} levels")
            print(f"{'=' * 80}")

            config = AdvancedOrderbookBenchmarkConfig(
                input_path=base_config.input_path,
                tick_size=base_config.tick_size,
                lot_size=base_config.lot_size,
                num_levels=size,
                warmup_operations=base_config.warmup_operations,
                num_operations=base_config.num_operations,
            )
            benchmark = AdvancedOrderbookBenchmark(config)
            benchmark.set_messages(messages)
            stats = benchmark.run()

            summaries.append(_summary_for_size(stats, size))

            stats_copy = _stats_copy(stats)
            all_stats.append((size, stats_copy))

            comp = BenchmarkStatistics()
            for op in ["snapshot", "delta", "bbo"]:
                op_stats = comp.add_operation(op)
                op_stats.latencies_ns = _operation_latencies(stats, op).copy()
            comp.total_time_ns = stats.total_time_ns
            comparative_rows.append((size, comp))

            del benchmark
            gc.collect()

        print("\n")
        print_comparative_summary(summaries)

        comparative = ComparativeReporter(
            "Comparative Mean Latency (raw ns)",
            size_label="Levels",
        )
        comparative.print_comparative_table(comparative_rows, ["snapshot", "delta", "bbo"])

        print("\nDetailed Reports by Size:")
        print("=" * 80)
        for size, stats in all_stats:
            print(f"\n{'=' * 80}")
            print(f"Detailed Report: {size} levels")
            print(f"{'=' * 80}\n")
            _print_detailed_report(stats, str(input_path), base_config.warmup_operations, size)

    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        print("Run collect_data.py first to generate the input file.")


def main() -> None:
    """Main entry point."""
    cli = (
        BenchmarkCLI("Benchmark AdvancedOrderbook with collected Binance data")
        .add_input_file(
            default="benchmarks/orderbook/data/btcusdt_100k.jsonl",
            help_text=(
                "Input file path "
                "(default: benchmarks/orderbook/data/btcusdt_100k.jsonl)"
            ),
        )
    )
    cli.parser.add_argument(
        "--levels",
        "-l",
        type=int,
        default=2048,
        help=(
            "Number of orderbook levels "
            "(default: 2048, ignored if --multi-size is used)"
        ),
    )
    cli.parser.description += (
        ". For --multi-size, tested levels are 2^6..2^12 (64..4096)."
    )

    args = cli.parse()
    config = _build_config_from_args(args)

    try:
        if args.multi_size:
            _run_multi_size(config)
        else:
            _run_single(config)
    except KeyboardInterrupt:
        print("\nBenchmark interrupted.")


if __name__ == "__main__":
    main()
