"""Benchmarks BytesRingBuffer performance with various message sizes.

Usage:
    uv run python benchmarks/ringbuffer/benchmark_bytes.py [--size SIZE]
    uv run python benchmarks/ringbuffer/benchmark_bytes.py --multi-size

Measures BytesRingBuffer operations:
- insert, insert_batch, consume, consume_all
- overwrite_latest, contains
- Async operations: aconsume, aconsume_iterable
- Comparison: disable_async=True vs False, only_insert_unique=True vs False
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import numpy as np

from mm_toolbox.ringbuffer.bytes import BytesRingBuffer

from benchmarks.core import (
    BaseBenchmarkConfig,
    BenchmarkCLI,
    BenchmarkReporter,
    BenchmarkRunner,
)


@dataclass
class BytesBenchmarkConfig(BaseBenchmarkConfig):
    """Configuration for BytesRingBuffer benchmark.

    Args:
        buffer_size: Buffer capacity.
        num_operations: Number of operations to benchmark.
        warmup_operations: Number of warmup operations.
        item_sizes: List of byte sizes to test.
        batch_sizes: List of batch sizes for insert_batch tests.
        only_insert_unique: Whether to enable deduplication.
        disable_async: Whether to disable async operations.
    """

    buffer_size: int = 1024
    item_sizes: list[int] = field(default_factory=lambda: [32, 128, 512, 2048])
    batch_sizes: list[int] = field(default_factory=lambda: [1, 10, 100, 1000])
    only_insert_unique: bool = False
    disable_async: bool = True


class BytesRingBufferBenchmark(BenchmarkRunner[BytesBenchmarkConfig]):
    """Benchmark runner for BytesRingBuffer."""

    def __init__(self, config: BytesBenchmarkConfig) -> None:
        """Initialize benchmark runner.

        Args:
            config: Benchmark configuration.
        """
        super().__init__(config)
        self.data_by_size = self._generate_data()

    def _generate_data(self) -> dict[int, list[bytes]]:
        """Generate synthetic bytes data for benchmarking.

        Returns:
            Dictionary mapping size to list of bytes objects.
        """
        np.random.seed(42)
        data_by_size = {}
        num_samples = (
            self.config.num_operations + self.config.warmup_operations + 10000
        )

        for size in self.config.item_sizes:
            data = [np.random.bytes(size) for _ in range(num_samples)]
            data_by_size[size] = data

        return data_by_size

    def _create_subject(self) -> BytesRingBuffer:
        """Create BytesRingBuffer instance."""
        return BytesRingBuffer(
            max_capacity=self.config.buffer_size,
            disable_async=self.config.disable_async,
            only_insert_unique=self.config.only_insert_unique,
        )

    def _run_benchmark_suite(self, rb: BytesRingBuffer) -> None:
        """Run all benchmark operations.

        Args:
            rb: BytesRingBuffer instance to benchmark.
        """
        print("Running synchronous benchmarks...")
        for item_size in self.config.item_sizes:
            self._run_sync_suite(item_size)

        if not self.config.disable_async:
            print("\nRunning asynchronous benchmarks...")
            for item_size in self.config.item_sizes:
                asyncio.run(self._run_async_suite(item_size))

    def _run_sync_suite(self, item_size: int) -> None:
        """Run synchronous benchmarks for a given item size.

        Args:
            item_size: Size of items in bytes.
        """
        print(f"  - Testing size={item_size}...")

        rb = self._create_subject()
        self._benchmark_insert(rb, item_size)
        self._benchmark_insert_batch(rb, item_size)
        self._benchmark_overwrite_latest(rb, item_size)
        self._benchmark_consume(rb, item_size)
        self._benchmark_consume_all(rb, item_size)
        self._benchmark_contains(rb, item_size)

    async def _run_async_suite(self, item_size: int) -> None:
        """Run asynchronous benchmarks for a given item size.

        Args:
            item_size: Size of items in bytes.
        """
        print(f"  - Testing async size={item_size}...")

        rb = self._create_subject()
        await self._benchmark_aconsume(rb, item_size)
        await self._benchmark_aconsume_iterable(rb, item_size)

    def _benchmark_insert(self, rb: BytesRingBuffer, item_size: int) -> None:
        """Benchmark single insert operation."""
        data = self.data_by_size[item_size]
        idx = 0

        def operation():
            nonlocal idx
            rb.insert(data[idx % len(data)])
            idx += 1

        self.measure_operation(f"insert(size={item_size})", operation)

    def _benchmark_insert_batch(self, rb: BytesRingBuffer, item_size: int) -> None:
        """Benchmark batch insert operations."""
        data = self.data_by_size[item_size]

        for batch_size in self.config.batch_sizes:
            num_batches = self.config.num_operations // batch_size
            warmup_batches = self.config.warmup_operations // batch_size
            batch_idx = 0

            def operation():
                nonlocal batch_idx
                start_idx = (
                    (batch_idx + warmup_batches) * batch_size
                ) % (len(data) - batch_size)
                batch = data[start_idx : start_idx + batch_size]
                rb.insert_batch(batch)
                batch_idx += 1

            self.measure_operation(
                f"insert_batch(size={item_size}, n={batch_size})",
                operation,
                warmup_count=warmup_batches,
                measure_count=num_batches,
            )

    def _benchmark_overwrite_latest(
        self, rb: BytesRingBuffer, item_size: int
    ) -> None:
        """Benchmark overwrite_latest operation."""
        data = self.data_by_size[item_size]
        rb.insert(data[0])
        idx = 0

        def operation():
            nonlocal idx
            rb.overwrite_latest(data[idx % len(data)])
            idx += 1

        self.measure_operation(f"overwrite_latest(size={item_size})", operation)

    def _benchmark_consume(self, rb: BytesRingBuffer, item_size: int) -> None:
        """Benchmark single consume operation."""
        data = self.data_by_size[item_size]

        for i in range(
            self.config.warmup_operations + self.config.num_operations
        ):
            rb.insert(data[i % len(data)])

        idx = 0

        def operation():
            nonlocal idx
            if rb.is_empty():
                rb.insert(data[idx % len(data)])
            rb.consume()
            idx += 1

        self.measure_operation(f"consume(size={item_size})", operation)

    def _benchmark_consume_all(self, rb: BytesRingBuffer, item_size: int) -> None:
        """Benchmark consume_all operation."""
        data = self.data_by_size[item_size]
        fill_size = min(self.config.buffer_size, 1000)
        num_iterations = self.config.num_operations // 100

        def operation():
            rb.insert_batch(data[:fill_size])
            rb.consume_all()

        self.measure_operation(
            f"consume_all(size={item_size})",
            operation,
            warmup_count=0,
            measure_count=num_iterations,
        )

    def _benchmark_contains(self, rb: BytesRingBuffer, item_size: int) -> None:
        """Benchmark contains operation."""
        data = self.data_by_size[item_size]
        fill_size = min(self.config.buffer_size, 100)
        rb.insert_batch(data[:fill_size])
        idx = 0

        def operation():
            nonlocal idx
            rb.contains(data[idx % len(data)])
            idx += 1

        self.measure_operation(f"contains(size={item_size})", operation)

    async def _benchmark_aconsume(
        self, rb: BytesRingBuffer, item_size: int
    ) -> None:
        """Benchmark async consume operation."""
        data = self.data_by_size[item_size]

        for i in range(
            self.config.warmup_operations + self.config.num_operations
        ):
            rb.insert(data[i % len(data)])

        metrics = self.stats.add_operation(f"aconsume(size={item_size})")

        for _ in range(self.config.warmup_operations):
            if not rb.is_empty():
                await rb.aconsume()

        for i in range(self.config.num_operations):
            if rb.is_empty():
                rb.insert(data[i % len(data)])

            start = time.perf_counter_ns()
            await rb.aconsume()
            elapsed = time.perf_counter_ns() - start
            metrics.add_latency(elapsed)

    async def _benchmark_aconsume_iterable(
        self, rb: BytesRingBuffer, item_size: int
    ) -> None:
        """Benchmark async consume iterable."""
        data = self.data_by_size[item_size]
        fill_size = min(self.config.buffer_size, 1000)
        num_iterations = self.config.num_operations // fill_size

        metrics = self.stats.add_operation(f"aconsume_iterable(size={item_size})")

        for iteration in range(num_iterations):
            rb.insert_batch(data[:fill_size])

            count = 0
            start = time.perf_counter_ns()
            async for _ in rb.aconsume_iterable():
                count += 1
                if count >= fill_size:
                    break
            elapsed = time.perf_counter_ns() - start

            if iteration >= (num_iterations // 10):
                metrics.add_latency(elapsed // count if count > 0 else elapsed)


def main() -> None:
    """Main entry point."""
    cli = (
        BenchmarkCLI("Benchmark BytesRingBuffer performance")
        .add_size_arg(default=1024, help_text="Buffer capacity (default: 1024)")
        .add_comparison_flag("async", "Compare disable_async=True vs False")
        .add_comparison_flag("unique", "Compare only_insert_unique=True vs False")
    )
    args = cli.parse()

    config = BytesBenchmarkConfig(
        buffer_size=args.size,
        num_operations=args.operations,
        warmup_operations=args.warmup,
    )

    if args.multi_size:
        print("Multi-size benchmarking not yet implemented with new framework")
        return

    if args.compare_async or args.compare_unique:
        for disable_async in [True, False] if args.compare_async else [True]:
            for only_unique in [False, True] if args.compare_unique else [False]:
                config.disable_async = disable_async
                config.only_insert_unique = only_unique

                print(f"\n{'=' * 100}")
                print(
                    f"Running with async={'disabled' if disable_async else 'enabled'}, "
                    f"unique={'enabled' if only_unique else 'disabled'}"
                )
                print(f"{'=' * 100}\n")

                benchmark = BytesRingBufferBenchmark(config)
                stats = benchmark.run()

                reporter = BenchmarkReporter(
                    "BytesRingBuffer Benchmark Results",
                    {
                        "Buffer size": config.buffer_size,
                        "Async": "disabled" if config.disable_async else "enabled",
                        "Unique": "enabled"
                        if config.only_insert_unique
                        else "disabled",
                    },
                )
                reporter.print_full_report(stats, warmup=config.warmup_operations)
                print("\n" * 2)
    else:
        benchmark = BytesRingBufferBenchmark(config)
        stats = benchmark.run()

        reporter = BenchmarkReporter(
            "BytesRingBuffer Benchmark Results",
            {
                "Buffer size": config.buffer_size,
                "Async": "disabled" if config.disable_async else "enabled",
                "Unique": "enabled" if config.only_insert_unique else "disabled",
            },
        )
        reporter.print_full_report(stats, warmup=config.warmup_operations)


if __name__ == "__main__":
    main()
