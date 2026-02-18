"""Benchmarks NumericRingBuffer performance with synthetic data.

Usage:
    uv run python benchmarks/ringbuffer/benchmark_numeric.py [--size SIZE]
    uv run python benchmarks/ringbuffer/benchmark_numeric.py --multi-size

Measures NumericRingBuffer operations:
- insert, insert_batch, consume, consume_all
- contains, peekright, peekleft, __getitem__
- Async operations: aconsume, aconsume_iterable (when async is enabled)
- Comparison: disable_async=True vs False
"""

from __future__ import annotations

import asyncio
import gc
import time
from dataclasses import dataclass, field

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
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from benchmarks.core import (
        BaseBenchmarkConfig,
        BenchmarkCLI,
        BenchmarkReporter,
        BenchmarkRunner,
        ComparativeReporter,
    )
from mm_toolbox.ringbuffer.numeric import NumericRingBuffer


@dataclass
class NumericBenchmarkConfig(BaseBenchmarkConfig):
    """Configuration for NumericRingBuffer benchmark."""

    buffer_size: int = 1024
    batch_sizes: list[int] = field(default_factory=lambda: [1, 10, 100, 1000])
    disable_async: bool = True
    dtype: type = np.float32


class NumericRingBufferBenchmark(BenchmarkRunner[NumericBenchmarkConfig]):
    """Benchmark runner for NumericRingBuffer."""

    def __init__(self, config: NumericBenchmarkConfig) -> None:
        """Initialize benchmark runner.

        Args:
            config: Benchmark configuration.
        """
        super().__init__(config)
        self.data = self._generate_data()

    def _generate_data(self) -> np.ndarray:
        """Generate synthetic numeric data for benchmarking.

        Returns:
            Array of simulated price ticks.
        """
        num_samples = (
            self.config.num_operations + self.config.warmup_operations + 10_000
        )
        np.random.seed(42)
        prices = 50_000.0 + np.random.randn(num_samples).cumsum() * 10.0
        return prices.astype(self.config.dtype)

    def _create_subject(self) -> NumericRingBuffer:
        """Create NumericRingBuffer instance."""
        return NumericRingBuffer(
            self.config.buffer_size,
            self.config.dtype,
            self.config.disable_async,
        )

    def _new_buffer(self) -> NumericRingBuffer:
        """Create a fresh ringbuffer instance."""
        return self._create_subject()

    def _to_py_float(self, idx: int) -> float:
        """Convert numpy scalar to Python float for Cython fused dispatch."""
        return float(self.data[idx % len(self.data)])

    def _run_benchmark_suite(self, _subject: NumericRingBuffer) -> None:
        """Run full benchmark suite.

        Args:
            _subject: Unused base subject required by BenchmarkRunner interface.
        """
        print("Running synchronous benchmarks...")
        self._run_sync_suite()

        if self.config.disable_async:
            print("\nSkipping asynchronous benchmarks (async mode is disabled)")
        else:
            print("\nRunning asynchronous benchmarks...")
            asyncio.run(self._run_async_suite())

    def _run_sync_suite(self) -> None:
        """Run all synchronous benchmark operations."""
        self._benchmark_insert(self._new_buffer())
        self._benchmark_insert_batch(self._new_buffer())
        self._benchmark_consume(self._new_buffer())
        self._benchmark_consume_all(self._new_buffer())
        self._benchmark_contains(self._new_buffer())
        self._benchmark_peek(self._new_buffer())
        self._benchmark_getitem(self._new_buffer())

    async def _run_async_suite(self) -> None:
        """Run all asynchronous benchmark operations."""
        await self._benchmark_aconsume(self._new_buffer())
        await self._benchmark_aconsume_iterable(self._new_buffer())

    def _benchmark_insert(self, rb: NumericRingBuffer) -> None:
        """Benchmark single insert operation."""
        idx = 0

        def operation() -> None:
            nonlocal idx
            rb.insert(self._to_py_float(idx))
            idx += 1

        self.measure_operation("insert", operation)

    def _benchmark_insert_batch(self, rb: NumericRingBuffer) -> None:
        """Benchmark batch insert operations."""
        for batch_size in self.config.batch_sizes:
            warmup_batches = max(1, self.config.warmup_operations // batch_size)
            num_batches = max(1, self.config.num_operations // batch_size)
            batch_idx = 0

            def operation() -> None:
                nonlocal batch_idx
                start_idx = ((batch_idx + warmup_batches) * batch_size) % (
                    len(self.data) - batch_size
                )
                batch = self.data[start_idx : start_idx + batch_size]
                rb.insert_batch(batch)
                batch_idx += 1

            self.measure_operation(
                f"insert_batch(n={batch_size})",
                operation,
                warmup_count=warmup_batches,
                measure_count=num_batches,
            )

    def _benchmark_consume(self, rb: NumericRingBuffer) -> None:
        """Benchmark single consume operation."""
        for i in range(self.config.warmup_operations + self.config.num_operations):
            rb.insert(self._to_py_float(i))

        def operation() -> None:
            if rb.is_empty():
                rb.insert(self._to_py_float(np.random.randint(0, len(self.data))))
            rb.consume()

        self.measure_operation("consume", operation)

    def _benchmark_consume_all(self, rb: NumericRingBuffer) -> None:
        """Benchmark consume_all operation."""
        fill_size = min(self.config.buffer_size, 1000)
        num_iterations = max(1, self.config.num_operations // 100)

        def operation() -> None:
            rb.insert_batch(self.data[:fill_size])
            rb.consume_all()

        self.measure_operation(
            "consume_all",
            operation,
            warmup_count=1,
            measure_count=num_iterations,
        )

    def _benchmark_contains(self, rb: NumericRingBuffer) -> None:
        """Benchmark contains operation."""
        fill_size = min(self.config.buffer_size, 100)
        rb.insert_batch(self.data[:fill_size])
        idx = 0

        def operation() -> None:
            nonlocal idx
            rb.contains(self._to_py_float(idx))
            idx += 1

        self.measure_operation("contains", operation)

    def _benchmark_peek(self, rb: NumericRingBuffer) -> None:
        """Benchmark peek operations."""
        fill_size = min(self.config.buffer_size, 100)
        rb.insert_batch(self.data[:fill_size])

        for name in ("peekright", "peekleft"):
            method = getattr(rb, name)
            self.measure_operation(name, method)

    def _benchmark_getitem(self, rb: NumericRingBuffer) -> None:
        """Benchmark random access via __getitem__."""
        fill_size = min(self.config.buffer_size, 100)
        rb.insert_batch(self.data[:fill_size])

        def operation() -> None:
            idx = np.random.randint(0, fill_size)
            _ = rb[idx]

        self.measure_operation("__getitem__", operation)

    async def _benchmark_aconsume(self, rb: NumericRingBuffer) -> None:
        """Benchmark async consume operation."""
        for i in range(self.config.warmup_operations + self.config.num_operations):
            rb.insert(self._to_py_float(i))

        metrics = self.stats.add_operation("aconsume")

        for _ in range(self.config.warmup_operations):
            if not rb.is_empty():
                await rb.aconsume()

        for _ in range(self.config.num_operations):
            if rb.is_empty():
                rb.insert(self._to_py_float(np.random.randint(0, len(self.data))))

            start = time.perf_counter_ns()
            await rb.aconsume()
            elapsed = time.perf_counter_ns() - start
            metrics.add_latency(elapsed)

    async def _benchmark_aconsume_iterable(self, rb: NumericRingBuffer) -> None:
        """Benchmark async consume iterable."""
        fill_size = min(self.config.buffer_size, 1000)
        num_iterations = max(1, self.config.num_operations // fill_size)
        metrics = self.stats.add_operation("aconsume_iterable")

        for _ in range(num_iterations):
            rb.insert_batch(self.data[:fill_size])

            count = 0
            start = time.perf_counter_ns()
            async for _ in rb.aconsume_iterable():
                count += 1
                if count >= fill_size:
                    break
            elapsed = time.perf_counter_ns() - start
            metrics.add_latency(elapsed // count if count > 0 else elapsed)


def _build_config(
    base: NumericBenchmarkConfig,
    buffer_size: int,
    disable_async: bool,
) -> NumericBenchmarkConfig:
    """Build a benchmark config from a base template."""
    return NumericBenchmarkConfig(
        buffer_size=buffer_size,
        num_operations=base.num_operations,
        warmup_operations=base.warmup_operations,
        batch_sizes=base.batch_sizes,
        disable_async=disable_async,
        dtype=base.dtype,
    )


def run_single(config: NumericBenchmarkConfig) -> None:
    """Run a single benchmark configuration and print report."""
    benchmark = NumericRingBufferBenchmark(config)
    stats = benchmark.run()

    reporter = BenchmarkReporter(
        "NumericRingBuffer Benchmark Results",
        {
            "Buffer size": config.buffer_size,
            "dtype": np.dtype(config.dtype).name,
            "Async": "disabled" if config.disable_async else "enabled",
        },
    )
    reporter.print_full_report(stats, warmup=config.warmup_operations)


def run_multi_size(base_config: NumericBenchmarkConfig, disable_async: bool) -> None:
    """Run benchmarks across multiple buffer sizes."""
    sizes = [2**i for i in range(4, 13, 2)]
    results = []

    print(f"\n{'=' * 100}")
    print(
        "NumericRingBuffer Multi-Size "
        f"(async={'disabled' if disable_async else 'enabled'})"
    )
    print(f"{'=' * 100}")

    for size in sizes:
        print(f"\nTesting buffer size: {size}")
        config = _build_config(base_config, size, disable_async)

        benchmark = NumericRingBufferBenchmark(config)
        stats = benchmark.run()
        results.append((size, stats))

        gc.collect()

    reporter = ComparativeReporter(
        "Comparative Summary (mean latency ns)",
        size_label="Buffer",
    )
    reporter.print_comparative_table(
        results,
        ["insert", "consume", "contains", "peekright"],
    )


def main() -> None:
    """Main entry point."""
    cli = (
        BenchmarkCLI("Benchmark NumericRingBuffer performance")
        .add_size_arg(default=1024, help_text="Buffer capacity (default: 1024)")
        .add_comparison_flag("async", "Compare disable_async=True vs False")
    )
    args = cli.parse()

    base_config = NumericBenchmarkConfig(
        buffer_size=args.size,
        num_operations=args.operations,
        warmup_operations=args.warmup,
        disable_async=True,
        dtype=np.float32,
    )

    if args.multi_size:
        if args.compare_async:
            run_multi_size(base_config, disable_async=True)
            print("\n" * 2)
            run_multi_size(base_config, disable_async=False)
        else:
            run_multi_size(base_config, disable_async=True)
        return

    if args.compare_async:
        run_single(_build_config(base_config, args.size, disable_async=True))
        print("\n" * 2)
        run_single(_build_config(base_config, args.size, disable_async=False))
    else:
        run_single(_build_config(base_config, args.size, disable_async=True))


if __name__ == "__main__":
    main()
