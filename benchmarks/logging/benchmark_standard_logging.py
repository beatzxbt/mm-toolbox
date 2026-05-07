"""Benchmarks Standard Logger overhead without handler I/O.

Usage:
    uv run python benchmarks/logging/benchmark_standard_logging.py [--size SIZE]
    uv run python benchmarks/logging/benchmark_standard_logging.py --multi-size

Measures:
- Emit throughput (`logger.info(...)`)
- Shutdown path overhead with an empty buffer
- Shutdown path overhead with buffered messages to flush
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import threading
import time
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
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from benchmarks.core import (
        BaseBenchmarkConfig,
        BenchmarkCLI,
        BenchmarkReporter,
        BenchmarkRunner,
        ComparativeReporter,
    )

from mm_toolbox.logging.standard import BaseLogHandler, LogLevel, Logger, LoggerConfig


class CountingLogHandler(BaseLogHandler):
    """In-memory handler that counts flushes and consumed messages."""

    def __init__(self) -> None:
        super().__init__()
        self._flush_calls = 0
        self._messages_seen = 0
        self._lock = threading.Lock()

    async def push(self, buffer: list[str]) -> None:
        """Record flush/message counts without performing I/O."""
        with self._lock:
            self._flush_calls += 1
            self._messages_seen += len(buffer)

    @property
    def flush_calls(self) -> int:
        """Get total number of flushes received."""
        with self._lock:
            return self._flush_calls

    @property
    def messages_seen(self) -> int:
        """Get total number of consumed messages received."""
        with self._lock:
            return self._messages_seen


@dataclass
class DeliverySummary:
    """Produced vs consumed message summary."""

    produced: int
    consumed: int
    flush_calls: int

    @property
    def lost(self) -> int:
        """Get produced-consumed delta."""
        return max(0, self.produced - self.consumed)

    @property
    def loss_pct(self) -> float:
        """Get produced-consumed loss percentage."""
        if self.produced <= 0:
            return 0.0
        return (self.lost / self.produced) * 100.0


@dataclass
class StandardLoggingBenchmarkConfig(BaseBenchmarkConfig):
    """Configuration for Standard Logger benchmarking."""

    message_size: int = 256
    emit_batch_size: int = 1000
    flush_interval_s: float = 3600.0
    buffer_size: int = 120_000
    shutdown_batch_size: int = 128
    shutdown_iterations: int = 25


class StandardLoggingBenchmark(BenchmarkRunner[StandardLoggingBenchmarkConfig]):
    """Benchmark runner for Standard Logger overhead."""

    def __init__(self, config: StandardLoggingBenchmarkConfig) -> None:
        super().__init__(config)
        self.emit_info_delivery: DeliverySummary | None = None

    def _create_subject(self) -> None:
        """No persistent benchmark subject is required."""
        return None

    def _create_logger(self) -> tuple[Logger, CountingLogHandler]:
        """Create a logger configured for overhead-focused measurements."""
        handler = CountingLogHandler()
        config = LoggerConfig(
            base_level=LogLevel.INFO,
            do_stdout=False,
            flush_interval_s=self.config.flush_interval_s,
            buffer_size=self.config.buffer_size,
        )
        logger = Logger(
            name="standard-logging-benchmark",
            config=config,
            handlers=[handler],
        )
        return logger, handler

    @staticmethod
    def _shutdown_logger(logger: Logger) -> None:
        """Shut down logger and background thread via asyncio.run."""
        if logger.is_running():
            asyncio.run(logger.shutdown())

    def _run_benchmark_suite(self, _subject: None) -> None:
        """Run all benchmark operations."""
        self._benchmark_emit_info_ns_per_msg()
        self._benchmark_shutdown_empty()
        self._benchmark_shutdown_flush()

    def _benchmark_emit_info_ns_per_msg(self) -> None:
        """Measure batched emit compute time as ns/message percentiles."""
        if self.stats is None:
            raise RuntimeError("Statistics not initialized")

        logger, handler = self._create_logger()
        message = "x" * self.config.message_size
        metrics = self.stats.add_operation("emit_info_ns_per_msg")
        batch_size = max(
            1, min(self.config.emit_batch_size, self.config.num_operations)
        )

        try:
            # Warm up logger path before measurement.
            for _ in range(self.config.warmup_operations):
                logger.info(message)

            remaining = self.config.num_operations
            while remaining > 0:
                this_batch = min(batch_size, remaining)
                start = time.perf_counter_ns()
                for _ in range(this_batch):
                    logger.info(message)
                elapsed = time.perf_counter_ns() - start
                ns_per_msg = max(1, elapsed // this_batch)
                for _ in range(this_batch):
                    metrics.add_latency(
                        ns_per_msg,
                        batch_size=this_batch,
                        batch_elapsed_ns=elapsed,
                    )
                remaining -= this_batch
        finally:
            self._shutdown_logger(logger)

        produced = self.config.warmup_operations + self.config.num_operations
        self.emit_info_delivery = DeliverySummary(
            produced=produced,
            consumed=handler.messages_seen,
            flush_calls=handler.flush_calls,
        )

    def _run_shutdown_cycle(
        self,
        *,
        prefill_messages: int,
        message: str,
    ) -> tuple[int, int, int]:
        """Create a logger, optionally prefill messages, and time shutdown."""
        logger, handler = self._create_logger()
        try:
            for _ in range(prefill_messages):
                logger.info(message)

            start = time.perf_counter_ns()
            self._shutdown_logger(logger)
            elapsed = time.perf_counter_ns() - start
            return elapsed, handler.messages_seen, handler.flush_calls
        finally:
            if logger.is_running():
                self._shutdown_logger(logger)

    def _benchmark_shutdown_empty(self) -> None:
        """Measure shutdown overhead when no buffered messages exist."""
        if self.stats is None:
            raise RuntimeError("Statistics not initialized")

        warmup_cycles = min(
            self.config.warmup_operations, self.config.shutdown_iterations
        )
        measure_cycles = max(1, self.config.shutdown_iterations)
        message = "x" * self.config.message_size
        metrics = self.stats.add_operation("shutdown_empty")

        for _ in range(warmup_cycles):
            self._run_shutdown_cycle(prefill_messages=0, message=message)

        for _ in range(measure_cycles):
            elapsed, consumed, flush_calls = self._run_shutdown_cycle(
                prefill_messages=0, message=message
            )
            metrics.add_latency(
                elapsed,
                produced_count=0,
                consumed_count=consumed,
                loss_messages=0,
                flush_calls=flush_calls,
            )

    def _benchmark_shutdown_flush(self) -> None:
        """Measure shutdown overhead with buffered messages to flush."""
        if self.stats is None:
            raise RuntimeError("Statistics not initialized")

        warmup_cycles = min(
            self.config.warmup_operations, self.config.shutdown_iterations
        )
        measure_cycles = max(1, self.config.shutdown_iterations)
        message = "x" * self.config.message_size
        metrics = self.stats.add_operation("shutdown_flush")

        for _ in range(warmup_cycles):
            self._run_shutdown_cycle(
                prefill_messages=self.config.shutdown_batch_size,
                message=message,
            )

        for _ in range(measure_cycles):
            elapsed, consumed, flush_calls = self._run_shutdown_cycle(
                prefill_messages=self.config.shutdown_batch_size,
                message=message,
            )
            metrics.add_latency(
                elapsed,
                produced_count=self.config.shutdown_batch_size,
                consumed_count=consumed,
                loss_messages=max(0, self.config.shutdown_batch_size - consumed),
                flush_calls=flush_calls,
            )


def _parse_message_sizes(raw: str) -> list[int]:
    """Parse comma-separated message sizes."""
    values = [token.strip() for token in raw.split(",")]
    sizes = [int(token) for token in values if token]
    if not sizes:
        raise ValueError("At least one message size is required")
    if any(size <= 0 for size in sizes):
        raise ValueError("Message sizes must be positive integers")
    return sizes


def _resolve_buffer_size(
    explicit_buffer_size: int,
    *,
    operations: int,
    warmup: int,
    shutdown_batch_size: int,
) -> int:
    """Resolve effective logger buffer size."""
    if explicit_buffer_size > 0:
        return explicit_buffer_size

    return max(
        1024,
        operations + warmup + shutdown_batch_size + 256,
    )


def _build_config(
    *,
    message_size: int,
    args: argparse.Namespace,
) -> StandardLoggingBenchmarkConfig:
    """Build benchmark config from CLI args."""
    effective_buffer_size = _resolve_buffer_size(
        args.buffer_size,
        operations=max(1, int(args.operations)),
        warmup=max(0, int(args.warmup)),
        shutdown_batch_size=max(1, int(args.shutdown_batch_size)),
    )

    return StandardLoggingBenchmarkConfig(
        message_size=max(1, int(message_size)),
        emit_batch_size=max(1, int(args.emit_batch_size)),
        num_operations=max(1, int(args.operations)),
        warmup_operations=max(0, int(args.warmup)),
        flush_interval_s=max(0.001, float(args.flush_interval_s)),
        buffer_size=effective_buffer_size,
        shutdown_batch_size=max(1, int(args.shutdown_batch_size)),
        shutdown_iterations=max(1, int(args.shutdown_iterations)),
    )


def run_single(config: StandardLoggingBenchmarkConfig) -> None:
    """Run one benchmark configuration and print a report."""
    benchmark = StandardLoggingBenchmark(config)
    stats = benchmark.run()

    reporter = BenchmarkReporter(
        "Standard Logger Benchmark Results",
        {
            "Message size (bytes)": config.message_size,
            "Flush interval (s)": config.flush_interval_s,
            "Buffer size": config.buffer_size,
            "Shutdown batch size": config.shutdown_batch_size,
            "Shutdown iterations": config.shutdown_iterations,
        },
    )
    reporter.print_full_report(stats, warmup=config.warmup_operations)

    emit_metrics = stats.get_operation("emit_info_ns_per_msg")
    if emit_metrics is not None:
        pcts = emit_metrics.compute_percentiles()
        print(
            "Emit compute ns/msg percentiles: "
            f"mean={pcts['mean']:.1f}, "
            f"p50={pcts['p50']:.1f}, "
            f"p95={pcts['p95']:.1f}, "
            f"p99={pcts['p99']:.1f}"
        )

    if benchmark.emit_info_delivery is not None:
        summary = benchmark.emit_info_delivery
        print(
            "Emit delivery summary: "
            f"produced={summary.produced:,}, "
            f"consumed={summary.consumed:,}, "
            f"loss={summary.lost:,} ({summary.loss_pct:.2f}%), "
            f"flushes={summary.flush_calls:,}"
        )


def run_multi_size(args: argparse.Namespace) -> None:
    """Run message-size comparison benchmark and print comparative summary."""
    sizes = _parse_message_sizes(args.message_sizes)
    results = []
    delivery_rows: list[tuple[int, DeliverySummary]] = []

    print("=" * 100)
    print("Standard Logger Message Size Comparison")
    print("=" * 100)

    for size in sizes:
        print(f"\nTesting message size: {size} bytes")
        config = _build_config(message_size=size, args=args)

        benchmark = StandardLoggingBenchmark(config)
        stats = benchmark.run()
        results.append((size, stats))
        if benchmark.emit_info_delivery is not None:
            delivery_rows.append((size, benchmark.emit_info_delivery))

        gc.collect()

    reporter = ComparativeReporter(
        "Comparative Summary (mean latency ns)",
        size_label="Msg bytes",
    )
    reporter.print_comparative_table(
        results,
        ["emit_info_ns_per_msg", "shutdown_empty", "shutdown_flush"],
    )

    if delivery_rows:
        print("\nEmit delivery summary by message size")
        print("-" * 100)
        print(
            f"{'Msg bytes':>10} {'Produced':>12} {'Consumed':>12} "
            f"{'Loss':>12} {'Loss%':>8} {'Flushes':>10}"
        )
        for size, summary in delivery_rows:
            print(
                f"{size:>10} {summary.produced:>12} {summary.consumed:>12} "
                f"{summary.lost:>12} {summary.loss_pct:>8.2f} "
                f"{summary.flush_calls:>10}"
            )


def main() -> None:
    """Main entry point."""
    cli = BenchmarkCLI("Benchmark Standard Logger overhead").add_size_arg(
        default=256,
        help_text="Log message size in bytes for single-size runs (default: 256)",
    )
    cli.parser.add_argument(
        "--message-sizes",
        default="32,128,512,1024",
        help=(
            "Comma-separated message sizes used with --multi-size "
            "(default: 32,128,512,1024)"
        ),
    )
    cli.parser.add_argument(
        "--emit-batch-size",
        type=int,
        default=1000,
        help=(
            "Messages per timed emit batch used for compute percentiles (default: 1000)"
        ),
    )
    cli.parser.add_argument(
        "--flush-interval-s",
        type=float,
        default=3600.0,
        help="Logger flush interval in seconds (default: 3600.0)",
    )
    cli.parser.add_argument(
        "--buffer-size",
        type=int,
        default=0,
        help=(
            "Logger buffer size; 0 means auto-sized from operations/warmup (default: 0)"
        ),
    )
    cli.parser.add_argument(
        "--shutdown-batch-size",
        type=int,
        default=128,
        help="Messages queued before shutdown flush test (default: 128)",
    )
    cli.parser.add_argument(
        "--shutdown-iterations",
        type=int,
        default=25,
        help="Measurement iterations for shutdown operations (default: 25)",
    )

    args = cli.parse()

    if args.multi_size:
        run_multi_size(args)
        return

    config = _build_config(message_size=args.size, args=args)
    run_single(config)


if __name__ == "__main__":
    main()
