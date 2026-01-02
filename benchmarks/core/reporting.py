"""Report formatting utilities for benchmarks.

Provides reusable report formatting and printing for consistent output across benchmarks.
Replaces duplicated report generation methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .stats import BenchmarkStatistics


class BenchmarkReporter:
    """Formats and prints benchmark reports.

    Provides consistent report formatting for single-run benchmarks with
    configuration display and operation performance tables.

    Args:
        title: Benchmark title (e.g., "BytesRingBuffer Benchmark").
        config_info: Dictionary of config parameters to display.
    """

    def __init__(self, title: str, config_info: dict[str, str | int | float]) -> None:
        """Initialize reporter.

        Args:
            title: Benchmark title.
            config_info: Config parameters (e.g., {"buffer_size": 1024}).
        """
        self.title = title
        self.config_info = config_info

    def print_header(self, stats: BenchmarkStatistics, warmup: int = 0) -> None:
        """Print benchmark header with config and summary.

        Args:
            stats: Benchmark statistics.
            warmup: Number of warmup operations.
        """
        print("=" * 100)
        print(self.title)
        print("=" * 100)

        for key, value in self.config_info.items():
            print(f"{key}: {value}")

        if warmup > 0:
            print(f"Operations: {stats.total_operations:,} (warmup: {warmup:,})")
        else:
            print(f"Operations: {stats.total_operations:,}")

        if stats.total_time_ns > 0:
            total_time_s = stats.total_time_ns / 1e9
            print(
                f"Total time: {total_time_s:.3f}s | Throughput: {stats.throughput:.0f} ops/sec"
            )
        print()

    def print_operation_table(
        self,
        stats: BenchmarkStatistics,
        sort_operations: bool = True,
    ) -> None:
        """Print ASCII table of operation performance.

        Args:
            stats: Benchmark statistics.
            sort_operations: Whether to sort operation names alphabetically.
        """
        print("Operation Performance")
        print("-" * 100)
        header = (
            f"{'Operation':<40} {'Count':>8} {'Mean ns':>10} "
            f"{'P50':>10} {'P95':>10} {'P99':>10} {'ops/sec':>12}"
        )
        print(header)

        operation_names = (
            sorted(stats.operations.keys())
            if sort_operations
            else stats.operations.keys()
        )

        for name in operation_names:
            metrics = stats.operations[name]
            pcts = metrics.compute_percentiles()

            if pcts["count"] == 0:
                continue

            print(
                f"{name:<40} {pcts['count']:>8} {pcts['mean']:>10.1f} "
                f"{pcts['p50']:>10.1f} {pcts['p95']:>10.1f} {pcts['p99']:>10.1f} "
                f"{pcts['ops_per_sec']:>12.0f}"
            )

        print("=" * 100)

    def print_full_report(
        self,
        stats: BenchmarkStatistics,
        warmup: int = 0,
    ) -> None:
        """Print complete benchmark report.

        Args:
            stats: Benchmark statistics.
            warmup: Number of warmup operations.
        """
        self.print_header(stats, warmup)
        self.print_operation_table(stats)


class ComparativeReporter:
    """Formats comparative summaries for multi-size benchmarks.

    Provides side-by-side comparison of benchmark results across different
    sizes or configurations.

    Args:
        title: Title for comparative table.
        size_label: Label for size column (e.g., "Size", "Buffer Size").
    """

    def __init__(self, title: str, size_label: str = "Size") -> None:
        """Initialize comparative reporter.

        Args:
            title: Title for comparative table.
            size_label: Label for size column.
        """
        self.title = title
        self.size_label = size_label

    def print_comparative_table(
        self,
        results: list[tuple[int, BenchmarkStatistics]],
        key_operations: list[str],
    ) -> None:
        """Print comparative table across sizes.

        Args:
            results: List of (size, statistics) tuples.
            key_operations: List of operation names to compare.
        """
        print("=" * 100)
        print(self.title)
        print("=" * 100)

        # Build header
        header = f"{self.size_label:>10}"
        for op_name in key_operations:
            header += f" {op_name:>15}"
        print(header)
        print("-" * 100)

        # Print rows
        for size, stats in results:
            row = f"{size:>10}"
            for op_name in key_operations:
                metrics = stats.get_operation(op_name)
                if metrics:
                    pcts = metrics.compute_percentiles()
                    row += f" {pcts['mean']:>15.1f}"
                else:
                    row += f" {'-':>15}"
            print(row)

        print("=" * 100)
