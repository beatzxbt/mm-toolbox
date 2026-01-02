"""Benchmarks NumericRingBuffer performance with synthetic data.

Usage:
    uv run python benchmarks/ringbuffer/benchmark_numeric.py [--buffer-size SIZE]
    uv run python benchmarks/ringbuffer/benchmark_numeric.py --multi-size

Measures NumericRingBuffer operations:
- insert, insert_batch, consume, consume_all
- contains, peekright, peekleft, __getitem__
- Async operations: aconsume, aconsume_iterable
- Comparison: disable_async=True vs False
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import time
from dataclasses import dataclass, field

import numpy as np

from mm_toolbox.ringbuffer.numeric import NumericRingBuffer


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark run."""

    buffer_size: int
    num_operations: int = 100_000
    warmup_operations: int = 1_000
    batch_sizes: list[int] = field(default_factory=lambda: [1, 10, 100, 1000])
    disable_async: bool = True
    dtype: type = float


@dataclass
class OperationStats:
    """Statistics for a single operation type."""

    name: str
    latencies_ns: list[int] = field(default_factory=list)

    def add_latency(self, latency_ns: int) -> None:
        """Add a latency measurement."""
        self.latencies_ns.append(latency_ns)

    def compute_percentiles(self) -> dict[str, float]:
        """Compute percentile statistics."""
        if not self.latencies_ns:
            return {
                "count": 0,
                "mean": 0.0,
                "p10": 0.0,
                "p25": 0.0,
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "p99_9": 0.0,
                "ops_per_sec": 0.0,
            }

        arr = np.array(self.latencies_ns, dtype=np.float64)
        mean_ns = float(np.mean(arr))
        ops_per_sec = 1e9 / mean_ns if mean_ns > 0 else 0

        return {
            "count": len(self.latencies_ns),
            "mean": mean_ns,
            "p10": float(np.percentile(arr, 10)),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "p99_9": float(np.percentile(arr, 99.9)),
            "ops_per_sec": ops_per_sec,
        }


@dataclass
class BenchmarkResults:
    """Aggregated benchmark results."""

    config: BenchmarkConfig
    operations: dict[str, OperationStats] = field(default_factory=dict)

    def add_operation(self, name: str) -> OperationStats:
        """Add a new operation for tracking."""
        stats = OperationStats(name=name)
        self.operations[name] = stats
        return stats

    def get_operation(self, name: str) -> OperationStats:
        """Get existing operation stats."""
        return self.operations[name]

    def print_report(self) -> None:
        """Print formatted benchmark report."""
        async_mode = "disabled" if self.config.disable_async else "enabled"
        dtype_name = np.dtype(self.config.dtype).name
        print("=" * 100)
        print(f"NumericRingBuffer Benchmark Results (dtype={dtype_name}, capacity={self.config.buffer_size}, async={async_mode})")
        print("=" * 100)
        print(f"Operations: {self.config.num_operations:,} (warmup: {self.config.warmup_operations:,})")
        print()

        print("Operation Performance")
        print("-" * 100)
        header = f"{'Operation':<30} {'Count':>8} {'Mean ns':>10} {'P50':>10} {'P95':>10} {'P99':>10} {'ops/sec':>12}"
        print(header)

        for name in sorted(self.operations.keys()):
            stats = self.operations[name]
            pcts = stats.compute_percentiles()
            if pcts["count"] == 0:
                continue

            print(
                f"{name:<30} {pcts['count']:>8} {pcts['mean']:>10.1f} "
                f"{pcts['p50']:>10.1f} {pcts['p95']:>10.1f} {pcts['p99']:>10.1f} "
                f"{pcts['ops_per_sec']:>12.0f}"
            )

        print("=" * 100)


class NumericRingBufferBenchmark:
    """Benchmark runner for NumericRingBuffer."""

    def __init__(self, config: BenchmarkConfig) -> None:
        """Initialize the benchmark.

        Args:
            config: Benchmark configuration.
        """
        self.config = config
        self.results = BenchmarkResults(config=config)
        self.data: np.ndarray = self._generate_data()

    def _generate_data(self) -> np.ndarray:
        """Generate synthetic numeric data for benchmarking.

        Returns:
            Array of simulated price ticks.
        """
        num_samples = self.config.num_operations + self.config.warmup_operations + 10000
        np.random.seed(42)
        prices = 50000.0 + np.random.randn(num_samples).cumsum() * 10.0
        return prices.astype(self.config.dtype)

    def _benchmark_insert(self, rb: NumericRingBuffer) -> None:
        """Benchmark single insert operation."""
        stats = self.results.add_operation("insert")

        for _ in range(self.config.warmup_operations):
            idx = np.random.randint(0, len(self.data))
            rb.insert(float(self.data[idx]))

        for i in range(self.config.num_operations):
            idx = i % len(self.data)
            start = time.perf_counter_ns()
            rb.insert(float(self.data[idx]))
            elapsed = time.perf_counter_ns() - start
            stats.add_latency(elapsed)

    def _benchmark_insert_batch(self, rb: NumericRingBuffer) -> None:
        """Benchmark batch insert operations."""
        for batch_size in self.config.batch_sizes:
            stats = self.results.add_operation(f"insert_batch(n={batch_size})")

            num_batches = self.config.num_operations // batch_size
            warmup_batches = self.config.warmup_operations // batch_size

            for _ in range(warmup_batches):
                start_idx = np.random.randint(0, len(self.data) - batch_size)
                batch = self.data[start_idx:start_idx + batch_size]
                rb.insert_batch(batch)

            for i in range(num_batches):
                start_idx = (i * batch_size) % (len(self.data) - batch_size)
                batch = self.data[start_idx:start_idx + batch_size]

                start = time.perf_counter_ns()
                rb.insert_batch(batch)
                elapsed = time.perf_counter_ns() - start
                stats.add_latency(elapsed)

    def _benchmark_consume(self, rb: NumericRingBuffer) -> None:
        """Benchmark single consume operation."""
        for i in range(self.config.warmup_operations + self.config.num_operations):
            rb.insert(self.data[i % len(self.data)])

        stats = self.results.add_operation("consume")

        for _ in range(self.config.warmup_operations):
            if not rb.is_empty():
                rb.consume()

        for _ in range(self.config.num_operations):
            if rb.is_empty():
                rb.insert(self.data[np.random.randint(0, len(self.data))])

            start = time.perf_counter_ns()
            rb.consume()
            elapsed = time.perf_counter_ns() - start
            stats.add_latency(elapsed)

    def _benchmark_consume_all(self, rb: NumericRingBuffer) -> None:
        """Benchmark consume_all operation."""
        stats = self.results.add_operation("consume_all")

        fill_size = min(self.config.buffer_size, 1000)
        num_iterations = self.config.num_operations // 100

        for _ in range(num_iterations):
            rb.insert_batch(self.data[:fill_size])

            start = time.perf_counter_ns()
            rb.consume_all()
            elapsed = time.perf_counter_ns() - start
            stats.add_latency(elapsed)

    def _benchmark_contains(self, rb: NumericRingBuffer) -> None:
        """Benchmark contains operation."""
        fill_size = min(self.config.buffer_size, 100)
        rb.insert_batch(self.data[:fill_size])

        stats = self.results.add_operation("contains")

        for _ in range(self.config.warmup_operations):
            idx = np.random.randint(0, len(self.data))
            rb.contains(self.data[idx])

        for i in range(self.config.num_operations):
            idx = i % len(self.data)
            start = time.perf_counter_ns()
            rb.contains(self.data[idx])
            elapsed = time.perf_counter_ns() - start
            stats.add_latency(elapsed)

    def _benchmark_peek(self, rb: NumericRingBuffer) -> None:
        """Benchmark peek operations."""
        fill_size = min(self.config.buffer_size, 100)
        rb.insert_batch(self.data[:fill_size])

        for operation in ["peekright", "peekleft"]:
            stats = self.results.add_operation(operation)
            method = getattr(rb, operation)

            for _ in range(self.config.warmup_operations):
                method()

            for _ in range(self.config.num_operations):
                start = time.perf_counter_ns()
                method()
                elapsed = time.perf_counter_ns() - start
                stats.add_latency(elapsed)

    def _benchmark_getitem(self, rb: NumericRingBuffer) -> None:
        """Benchmark random access via __getitem__."""
        fill_size = min(self.config.buffer_size, 100)
        rb.insert_batch(self.data[:fill_size])

        stats = self.results.add_operation("__getitem__")

        for _ in range(self.config.warmup_operations):
            idx = np.random.randint(0, fill_size)
            _ = rb[idx]

        for _ in range(self.config.num_operations):
            idx = np.random.randint(0, fill_size)
            start = time.perf_counter_ns()
            _ = rb[idx]
            elapsed = time.perf_counter_ns() - start
            stats.add_latency(elapsed)

    async def _benchmark_aconsume(self, rb: NumericRingBuffer) -> None:
        """Benchmark async consume operation."""
        for i in range(self.config.warmup_operations + self.config.num_operations):
            rb.insert(self.data[i % len(self.data)])

        stats = self.results.add_operation("aconsume")

        for _ in range(self.config.warmup_operations):
            if not rb.is_empty():
                await rb.aconsume()

        for _ in range(self.config.num_operations):
            if rb.is_empty():
                rb.insert(self.data[np.random.randint(0, len(self.data))])

            start = time.perf_counter_ns()
            await rb.aconsume()
            elapsed = time.perf_counter_ns() - start
            stats.add_latency(elapsed)

    async def _benchmark_aconsume_iterable(self, rb: NumericRingBuffer) -> None:
        """Benchmark async consume iterable."""
        fill_size = min(self.config.buffer_size, 1000)

        stats = self.results.add_operation("aconsume_iterable")

        num_iterations = self.config.num_operations // fill_size

        for iteration in range(num_iterations):
            rb.insert_batch(self.data[:fill_size])

            count = 0
            start = time.perf_counter_ns()
            async for _ in rb.aconsume_iterable():
                count += 1
                if count >= fill_size:
                    break
            elapsed = time.perf_counter_ns() - start

            if iteration >= (num_iterations // 10):
                stats.add_latency(elapsed // count if count > 0 else elapsed)

    def run_sync(self) -> None:
        """Run synchronous benchmarks."""
        print(f"\nRunning synchronous benchmarks...")

        rb = NumericRingBuffer(
            self.config.buffer_size,
            self.config.dtype,
            self.config.disable_async,
        )

        print("  - Testing insert...")
        self._benchmark_insert(rb)

        rb = NumericRingBuffer(
            self.config.buffer_size,
            self.config.dtype,
            self.config.disable_async,
        )
        print("  - Testing insert_batch...")
        self._benchmark_insert_batch(rb)

        rb = NumericRingBuffer(
            self.config.buffer_size,
            self.config.dtype,
            self.config.disable_async,
        )
        print("  - Testing consume...")
        self._benchmark_consume(rb)

        rb = NumericRingBuffer(
            self.config.buffer_size,
            self.config.dtype,
            self.config.disable_async,
        )
        print("  - Testing consume_all...")
        self._benchmark_consume_all(rb)

        rb = NumericRingBuffer(
            self.config.buffer_size,
            self.config.dtype,
            self.config.disable_async,
        )
        print("  - Testing contains...")
        self._benchmark_contains(rb)

        rb = NumericRingBuffer(
            self.config.buffer_size,
            self.config.dtype,
            self.config.disable_async,
        )
        print("  - Testing peek operations...")
        self._benchmark_peek(rb)

        rb = NumericRingBuffer(
            self.config.buffer_size,
            self.config.dtype,
            self.config.disable_async,
        )
        print("  - Testing __getitem__...")
        self._benchmark_getitem(rb)

    async def run_async(self) -> None:
        """Run asynchronous benchmarks."""
        print(f"\nRunning asynchronous benchmarks...")

        rb = NumericRingBuffer(
            self.config.buffer_size,
            self.config.dtype,
            self.config.disable_async,
        )
        print("  - Testing aconsume...")
        await self._benchmark_aconsume(rb)

        rb = NumericRingBuffer(
            self.config.buffer_size,
            self.config.dtype,
            self.config.disable_async,
        )
        print("  - Testing aconsume_iterable...")
        await self._benchmark_aconsume_iterable(rb)

    def run(self) -> None:
        """Run all benchmarks."""
        dtype_name = np.dtype(self.config.dtype).name
        print(f"Starting NumericRingBuffer benchmark...")
        print(f"Buffer size: {self.config.buffer_size}")
        print(f"dtype: {dtype_name}")
        print(f"Async mode: {'disabled' if self.config.disable_async else 'enabled'}")

        self.run_sync()
        asyncio.run(self.run_async())

        print("\n" + "=" * 100)
        self.results.print_report()


def run_multi_size(base_config: BenchmarkConfig, disable_async: bool) -> None:
    """Run benchmarks across multiple buffer sizes.

    Args:
        base_config: Base configuration to use.
        disable_async: Whether to disable async.
    """
    sizes = [2**i for i in range(4, 13, 2)]

    async_mode = "disabled" if disable_async else "enabled"
    print(f"\n{'=' * 100}")
    print(f"Multi-Size Benchmark (async={async_mode})")
    print(f"{'=' * 100}\n")

    all_results: list[tuple[int, BenchmarkResults]] = []

    for size in sizes:
        config = BenchmarkConfig(
            buffer_size=size,
            num_operations=base_config.num_operations,
            warmup_operations=base_config.warmup_operations,
            batch_sizes=base_config.batch_sizes,
            disable_async=disable_async,
            dtype=float,
        )

        print(f"\n{'=' * 100}")
        print(f"Testing buffer size: {size}")
        print(f"{'=' * 100}")

        benchmark = NumericRingBufferBenchmark(config)
        benchmark.run()

        all_results.append((size, benchmark.results))

        gc.collect()

    print(f"\n{'=' * 100}")
    print(f"Comparative Summary (async={async_mode})")
    print(f"{'=' * 100}")

    print(f"{'Size':>8} {'insert':>12} {'consume':>12} {'contains':>12} {'peekright':>12}")
    print("-" * 100)

    for size, results in all_results:
        insert_stats = results.operations.get("insert", OperationStats("insert")).compute_percentiles()
        consume_stats = results.operations.get("consume", OperationStats("consume")).compute_percentiles()
        contains_stats = results.operations.get("contains", OperationStats("contains")).compute_percentiles()
        peek_stats = results.operations.get("peekright", OperationStats("peekright")).compute_percentiles()

        print(
            f"{size:>8} {insert_stats['mean']:>12.1f} {consume_stats['mean']:>12.1f} "
            f"{contains_stats['mean']:>12.1f} {peek_stats['mean']:>12.1f}"
        )

    print("=" * 100)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark NumericRingBuffer performance"
    )
    parser.add_argument(
        "--buffer-size",
        "-b",
        type=int,
        default=1024,
        help="Buffer capacity (default: 1024)",
    )
    parser.add_argument(
        "--operations",
        "-n",
        type=int,
        default=100_000,
        help="Number of operations to benchmark (default: 100,000)",
    )
    parser.add_argument(
        "--warmup",
        "-w",
        type=int,
        default=1_000,
        help="Number of warmup operations (default: 1,000)",
    )
    parser.add_argument(
        "--multi-size",
        "-m",
        action="store_true",
        help="Test multiple buffer sizes from 2^4 (16) to 2^12 (4096)",
    )
    parser.add_argument(
        "--compare-async",
        "-a",
        action="store_true",
        help="Compare disable_async=True vs False",
    )

    args = parser.parse_args()

    config = BenchmarkConfig(
        buffer_size=args.buffer_size,
        num_operations=args.operations,
        warmup_operations=args.warmup,
        dtype=float,
    )

    if args.multi_size:
        if args.compare_async:
            print("Testing with disable_async=True")
            run_multi_size(config, disable_async=True)
            print("\n" * 3)
            print("Testing with disable_async=False")
            run_multi_size(config, disable_async=False)
        else:
            run_multi_size(config, disable_async=True)
    elif args.compare_async:
        print("Testing with disable_async=True")
        config.disable_async = True
        benchmark = NumericRingBufferBenchmark(config)
        benchmark.run()

        print("\n" * 3)
        print("Testing with disable_async=False")
        config.disable_async = False
        benchmark = NumericRingBufferBenchmark(config)
        benchmark.run()
    else:
        benchmark = NumericRingBufferBenchmark(config)
        benchmark.run()


if __name__ == "__main__":
    main()
