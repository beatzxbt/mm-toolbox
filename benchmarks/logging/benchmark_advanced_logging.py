"""Benchmarks advanced multi-process logging throughput and per-message latency.

Usage:
    uv run python benchmarks/logging/benchmark_advanced_logging.py --help

This benchmark measures the advanced logger in true multi-process mode:
- 1 master process (this benchmark process) with an in-memory counting handler
- N worker processes emitting logs as fast as possible

It provides two comparison axes:
1) Message-size sweep with fixed worker count (default: 8)
2) Worker-count sweep with fixed message size
"""

from __future__ import annotations

import contextlib
import hashlib
import multiprocessing
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty as QueueEmpty

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
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from benchmarks.core import (
        BaseBenchmarkConfig,
        BenchmarkCLI,
        BenchmarkReporter,
        BenchmarkRunner,
        BenchmarkStatistics,
        ComparativeReporter,
    )

from mm_toolbox.logging.advanced import (
    BaseLogHandler,
    LoggerConfig,
    MasterLogger,
    WorkerLogger,
)


def _get_or_add_operation(stats: BenchmarkStatistics, name: str):
    """Get existing operation metrics or create them."""
    metrics = stats.get_operation(name)
    if metrics is None:
        metrics = stats.add_operation(name)
    return metrics


def _parse_positive_int_csv(raw: str, arg_name: str) -> list[int]:
    """Parse comma-separated positive integers while preserving order."""
    values: list[int] = []
    seen: set[int] = set()

    for token in (part.strip() for part in raw.split(",")):
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError(f"{arg_name} values must be positive; got {value}")
        if value not in seen:
            seen.add(value)
            values.append(value)

    if not values:
        raise ValueError(f"{arg_name} must contain at least one positive integer")
    return values


def _to_socket_path(ipc_path: str) -> Path:
    """Convert ipc:// path to filesystem path."""
    return Path(ipc_path.replace("ipc://", "", 1))


def _cleanup_ipc_path(ipc_path: str) -> None:
    """Remove stale IPC socket path, if present."""
    socket_path = _to_socket_path(ipc_path)
    if socket_path.exists():
        with contextlib.suppress(OSError):
            socket_path.unlink()


def _build_ipc_path(ipc_dir: Path, scope: str) -> str:
    """Build a short, deterministic IPC path under .ipc."""
    ipc_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(scope.encode("utf-8")).hexdigest()[:16]
    socket_path = ipc_dir / f"advlog_{digest}"
    return f"ipc://{socket_path.resolve()}"


def _join_processes(
    processes: list[multiprocessing.Process],
    timeout_sec: float,
) -> int:
    """Join worker processes with timeout; terminate stragglers.

    Returns:
        Number of workers that had to be force-terminated.
    """
    forced_terminations = 0
    deadline = time.monotonic() + max(0.1, timeout_sec)

    for proc in processes:
        remaining = max(0.0, deadline - time.monotonic())
        proc.join(timeout=remaining)
        if proc.is_alive():
            forced_terminations += 1
            proc.terminate()
            proc.join(timeout=1.0)

    return forced_terminations


class CountingLogHandler(BaseLogHandler):
    """Lightweight in-memory handler that only counts consumed messages."""

    def __init__(self) -> None:
        super().__init__()
        self._count = 0
        self._batches = 0
        self._lock = threading.Lock()

    def push(self, logs) -> None:
        with self._lock:
            self._count += len(logs)
            self._batches += 1

    @property
    def count(self) -> int:
        with self._lock:
            return self._count

    @property
    def batches(self) -> int:
        with self._lock:
            return self._batches


def _worker_emit_burst(
    path: str,
    worker_name: str,
    message_size: int,
    duration_sec: float,
    flush_interval_s: float,
    start_barrier: multiprocessing.Barrier,
    start_event: multiprocessing.Event,
    startup_timeout_s: float,
    result_queue: multiprocessing.Queue,
) -> None:
    """Worker process: emit INFO logs as fast as possible for fixed duration."""
    logger = WorkerLogger(
        config=LoggerConfig(
            path=path,
            do_stdout=False,
            flush_interval_s=flush_interval_s,
            emit_internal=False,
        ),
        name=worker_name,
    )
    payload = b"x" * message_size
    produced = 0
    elapsed_ns = 0

    try:
        start_barrier.wait(timeout=startup_timeout_s)
        if not start_event.wait(timeout=startup_timeout_s):
            raise TimeoutError("Worker start event wait timed out")

        start_ns = time.perf_counter_ns()
        end_ns = start_ns + int(duration_sec * 1e9)

        while time.perf_counter_ns() < end_ns:
            logger.info(msg_bytes=payload)
            produced += 1

        elapsed_ns = time.perf_counter_ns() - start_ns
        result_queue.put(("ok", produced, elapsed_ns))
    except BaseException as exc:
        result_queue.put(
            ("error", produced, elapsed_ns, f"{type(exc).__name__}: {exc}")
        )
    finally:
        with contextlib.suppress(Exception):
            logger.shutdown()


@dataclass
class ScenarioSample:
    """One measured run for a single worker/message-size configuration."""

    workers: int
    message_size: int
    duration_ns: int
    produced_count: int
    consumed_count: int
    worker_errors: int
    forced_terminations: int

    @property
    def produced_msgs_per_sec(self) -> float:
        if self.duration_ns <= 0:
            return 0.0
        return self.produced_count / (self.duration_ns / 1e9)

    @property
    def consumed_msgs_per_sec(self) -> float:
        if self.duration_ns <= 0:
            return 0.0
        return self.consumed_count / (self.duration_ns / 1e9)

    @property
    def produced_ns_per_msg(self) -> int:
        if self.produced_count <= 0:
            return max(1, self.duration_ns)
        return max(1, int(self.duration_ns / self.produced_count))

    @property
    def consumed_ns_per_msg(self) -> int:
        if self.consumed_count <= 0:
            return max(1, self.duration_ns)
        return max(1, int(self.duration_ns / self.consumed_count))

    @property
    def lost_messages(self) -> int:
        return max(0, self.produced_count - self.consumed_count)


@dataclass
class AxisSummary:
    """Aggregated summary over repeated samples for one axis point."""

    axis_value: int
    workers: int
    message_size: int
    duration_ns: int
    produced_count: int
    consumed_count: int
    produced_msgs_per_sec: float
    consumed_msgs_per_sec: float
    produced_ns_per_msg: float
    consumed_ns_per_msg: float
    lost_messages: int
    loss_pct: float
    worker_errors: int
    forced_terminations: int


def _summarize_samples(axis_value: int, samples: list[ScenarioSample]) -> AxisSummary:
    """Aggregate repeated samples into one row."""
    if not samples:
        raise ValueError("Cannot summarize empty sample list")

    workers = samples[0].workers
    message_size = samples[0].message_size
    duration_ns = max(1, sum(max(1, sample.duration_ns) for sample in samples))
    produced = sum(sample.produced_count for sample in samples)
    consumed = sum(sample.consumed_count for sample in samples)
    worker_errors = sum(sample.worker_errors for sample in samples)
    forced_terminations = sum(sample.forced_terminations for sample in samples)
    lost = max(0, produced - consumed)
    loss_pct = (lost / produced * 100.0) if produced > 0 else 0.0

    produced_msgs_per_sec = produced / (duration_ns / 1e9) if duration_ns > 0 else 0.0
    consumed_msgs_per_sec = consumed / (duration_ns / 1e9) if duration_ns > 0 else 0.0
    produced_ns_per_msg = duration_ns / produced if produced > 0 else float(duration_ns)
    consumed_ns_per_msg = duration_ns / consumed if consumed > 0 else float(duration_ns)

    return AxisSummary(
        axis_value=axis_value,
        workers=workers,
        message_size=message_size,
        duration_ns=duration_ns,
        produced_count=produced,
        consumed_count=consumed,
        produced_msgs_per_sec=produced_msgs_per_sec,
        consumed_msgs_per_sec=consumed_msgs_per_sec,
        produced_ns_per_msg=produced_ns_per_msg,
        consumed_ns_per_msg=consumed_ns_per_msg,
        lost_messages=lost,
        loss_pct=loss_pct,
        worker_errors=worker_errors,
        forced_terminations=forced_terminations,
    )


@dataclass
class AdvancedLoggingBenchmarkConfig(BaseBenchmarkConfig):
    """Configuration for advanced logging benchmark."""

    axis: str = "both"
    fixed_workers: int = 8
    fixed_message_size: int = 256
    message_sizes: list[int] = field(default_factory=lambda: [32, 128, 512, 2048, 8192])
    worker_counts: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 12, 16])
    duration_sec: float = 2.0
    flush_interval_s: float = 0.05
    startup_timeout_s: float = 10.0
    drain_timeout_s: float = 5.0
    ipc_dir: str = ".ipc"


class AdvancedLoggingBenchmark(BenchmarkRunner[AdvancedLoggingBenchmarkConfig]):
    """Benchmark runner for advanced logger throughput (master + workers)."""

    def __init__(self, config: AdvancedLoggingBenchmarkConfig) -> None:
        super().__init__(config)
        self.message_size_axis_rows: list[AxisSummary] = []
        self.worker_count_axis_rows: list[AxisSummary] = []
        self.message_size_axis_stats: list[tuple[int, BenchmarkStatistics]] = []
        self.worker_count_axis_stats: list[tuple[int, BenchmarkStatistics]] = []

    def _create_subject(self) -> None:
        """No persistent subject is required."""
        return None

    def _run_scenario_once(
        self,
        workers: int,
        message_size: int,
        scope_tag: str,
    ) -> ScenarioSample:
        """Run one timed multi-process scenario."""
        ipc_dir = Path(self.config.ipc_dir)
        scope = f"{os.getpid()}:{scope_tag}:w{workers}:s{message_size}:{time.time_ns()}"
        ipc_path = _build_ipc_path(ipc_dir, scope)
        _cleanup_ipc_path(ipc_path)

        start_barrier = multiprocessing.Barrier(workers + 1)
        start_event = multiprocessing.Event()
        result_queue: multiprocessing.Queue = multiprocessing.Queue()
        processes: list[multiprocessing.Process] = []

        handler = CountingLogHandler()
        master = MasterLogger(
            config=LoggerConfig(
                path=ipc_path,
                do_stdout=False,
                flush_interval_s=self.config.flush_interval_s,
                emit_internal=False,
            ),
            log_handlers=[handler],
        )

        worker_errors = 0
        forced_terminations = 0
        produced_total = 0
        worker_durations: list[int] = []

        try:
            for worker_idx in range(workers):
                proc = multiprocessing.Process(
                    target=_worker_emit_burst,
                    args=(
                        ipc_path,
                        f"W{worker_idx}",
                        message_size,
                        self.config.duration_sec,
                        self.config.flush_interval_s,
                        start_barrier,
                        start_event,
                        self.config.startup_timeout_s,
                        result_queue,
                    ),
                )
                proc.start()
                processes.append(proc)

            start_barrier.wait(timeout=self.config.startup_timeout_s)
            start_event.set()

            join_timeout = (
                self.config.startup_timeout_s
                + self.config.duration_sec
                + self.config.drain_timeout_s
                + 2.0
            )
            forced_terminations = _join_processes(processes, join_timeout)

            collect_deadline = time.monotonic() + max(1.0, self.config.drain_timeout_s)
            received = 0
            while received < workers and time.monotonic() < collect_deadline:
                try:
                    record = result_queue.get(timeout=0.05)
                except QueueEmpty:
                    continue

                received += 1
                status = record[0]
                if status == "ok":
                    _, produced, duration_ns = record
                    produced_total += int(produced)
                    worker_durations.append(max(1, int(duration_ns)))
                else:
                    _, produced, duration_ns, _err = record
                    produced_total += int(produced)
                    worker_durations.append(max(1, int(duration_ns)))
                    worker_errors += 1

            if received < workers:
                worker_errors += workers - received

            drain_deadline = time.monotonic() + max(0.1, self.config.drain_timeout_s)
            while time.monotonic() < drain_deadline:
                if handler.count >= produced_total:
                    break
                time.sleep(0.002)

            master.shutdown()

            duration_ns = max(worker_durations) if worker_durations else 1
            consumed_count = handler.count

            return ScenarioSample(
                workers=workers,
                message_size=message_size,
                duration_ns=duration_ns,
                produced_count=produced_total,
                consumed_count=consumed_count,
                worker_errors=worker_errors,
                forced_terminations=forced_terminations,
            )
        finally:
            with contextlib.suppress(Exception):
                if master.is_running():
                    master.shutdown()

            for proc in processes:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=1.0)

            with contextlib.suppress(Exception):
                result_queue.close()
                result_queue.join_thread()

            _cleanup_ipc_path(ipc_path)

    def _record_sample(
        self,
        stats: BenchmarkStatistics,
        produced_operation: str,
        consumed_operation: str,
        sample: ScenarioSample,
    ) -> None:
        """Record one scenario sample into benchmark stats."""
        produced_metrics = _get_or_add_operation(stats, produced_operation)
        produced_metrics.add_latency(
            sample.produced_ns_per_msg,
            workers=sample.workers,
            message_size=sample.message_size,
            produced_count=sample.produced_count,
            consumed_count=sample.consumed_count,
            duration_ns=sample.duration_ns,
            throughput_msgs_per_sec=sample.produced_msgs_per_sec,
            lost_messages=sample.lost_messages,
            worker_errors=sample.worker_errors,
            forced_terminations=sample.forced_terminations,
        )

        consumed_metrics = _get_or_add_operation(stats, consumed_operation)
        consumed_metrics.add_latency(
            sample.consumed_ns_per_msg,
            workers=sample.workers,
            message_size=sample.message_size,
            produced_count=sample.produced_count,
            consumed_count=sample.consumed_count,
            duration_ns=sample.duration_ns,
            throughput_msgs_per_sec=sample.consumed_msgs_per_sec,
            lost_messages=sample.lost_messages,
            worker_errors=sample.worker_errors,
            forced_terminations=sample.forced_terminations,
        )

    def _run_axis_sweep(
        self,
        axis: str,
        sweep_values: list[int],
        fixed_workers: int,
        fixed_message_size: int,
    ) -> tuple[list[AxisSummary], list[tuple[int, BenchmarkStatistics]]]:
        """Run one sweep axis and return aggregate rows + per-point stats."""
        rows: list[AxisSummary] = []
        per_value_stats: list[tuple[int, BenchmarkStatistics]] = []
        repeats = max(1, self.config.num_operations)
        warmups = max(0, self.config.warmup_operations)

        for value in sweep_values:
            if axis == "message-size":
                workers = fixed_workers
                message_size = value
                scenario_label = (
                    f"[message-size] message_size={value}B, workers={workers}"
                )
            else:
                workers = value
                message_size = fixed_message_size
                scenario_label = (
                    f"[worker-count] workers={value}, message_size={message_size}B"
                )

            print(f"\n{scenario_label}")

            for warmup_idx in range(warmups):
                _ = self._run_scenario_once(
                    workers,
                    message_size,
                    scope_tag=f"{axis}:warmup:{value}:{warmup_idx}",
                )

            scenario_samples: list[ScenarioSample] = []
            scenario_stats = BenchmarkStatistics()

            for repeat_idx in range(repeats):
                sample = self._run_scenario_once(
                    workers,
                    message_size,
                    scope_tag=f"{axis}:repeat:{value}:{repeat_idx}",
                )
                scenario_samples.append(sample)

                self._record_sample(
                    self.stats,
                    produced_operation=f"{axis}.produced_ns_per_msg(value={value})",
                    consumed_operation=f"{axis}.consumed_ns_per_msg(value={value})",
                    sample=sample,
                )
                self._record_sample(
                    scenario_stats,
                    produced_operation="produced_ns_per_msg",
                    consumed_operation="consumed_ns_per_msg",
                    sample=sample,
                )

            rows.append(_summarize_samples(value, scenario_samples))
            per_value_stats.append((value, scenario_stats))

        return rows, per_value_stats

    def _run_benchmark_suite(self, _subject: None) -> None:
        """Run selected benchmark axes."""
        if self.config.axis in ("both", "message-size"):
            rows, stats = self._run_axis_sweep(
                axis="message-size",
                sweep_values=self.config.message_sizes,
                fixed_workers=self.config.fixed_workers,
                fixed_message_size=self.config.fixed_message_size,
            )
            self.message_size_axis_rows = rows
            self.message_size_axis_stats = stats

        if self.config.axis in ("both", "worker-count"):
            rows, stats = self._run_axis_sweep(
                axis="worker-count",
                sweep_values=self.config.worker_counts,
                fixed_workers=self.config.fixed_workers,
                fixed_message_size=self.config.fixed_message_size,
            )
            self.worker_count_axis_rows = rows
            self.worker_count_axis_stats = stats


def _print_axis_summary(
    title: str,
    axis_label: str,
    rows: list[AxisSummary],
) -> None:
    """Print throughput/count summary table for one axis."""
    if not rows:
        return

    print("=" * 132)
    print(title)
    print("=" * 132)
    print(
        f"{axis_label:>10} {'Workers':>8} {'MsgB':>8} {'Produced':>12} "
        f"{'Consumed':>12} {'Prod msg/s':>12} {'Cons msg/s':>12} "
        f"{'Prod ns/msg':>12} {'Cons ns/msg':>12} {'Loss%':>8} {'Err':>6}"
    )
    print("-" * 132)

    for row in rows:
        print(
            f"{row.axis_value:>10} {row.workers:>8} {row.message_size:>8} "
            f"{row.produced_count:>12} {row.consumed_count:>12} "
            f"{row.produced_msgs_per_sec:>12.0f} {row.consumed_msgs_per_sec:>12.0f} "
            f"{row.produced_ns_per_msg:>12.1f} {row.consumed_ns_per_msg:>12.1f} "
            f"{row.loss_pct:>8.2f} {row.worker_errors + row.forced_terminations:>6}"
        )

    print("=" * 132)


def main() -> None:
    """Main entry point."""
    cli = BenchmarkCLI(
        "Benchmark advanced logger throughput (1 master + N workers, pure IPC overhead)"
    )
    cli.parser.set_defaults(operations=1, warmup=0)
    cli.parser.add_argument(
        "--axis",
        choices=["both", "message-size", "worker-count"],
        default="both",
        help="Which comparison axis to run (default: both)",
    )
    cli.parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Fixed worker count for message-size sweep (default: 8)",
    )
    cli.parser.add_argument(
        "--message-size",
        type=int,
        default=256,
        help="Fixed message size in bytes for worker-count sweep (default: 256)",
    )
    cli.parser.add_argument(
        "--message-sizes",
        default="32,128,512,2048,8192",
        help=(
            "Comma-separated message sizes for message-size sweep "
            "(default: 32,128,512,2048,8192)"
        ),
    )
    cli.parser.add_argument(
        "--worker-counts",
        default="1,2,4,8,12,16",
        help=(
            "Comma-separated worker counts for worker-count sweep "
            "(default: 1,2,4,8,12,16)"
        ),
    )
    cli.parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="Timed duration per scenario run in seconds (default: 2.0)",
    )
    cli.parser.add_argument(
        "--flush-interval",
        type=float,
        default=0.05,
        help="Logger flush interval in seconds (default: 0.05)",
    )
    cli.parser.add_argument(
        "--startup-timeout",
        type=float,
        default=10.0,
        help="Worker startup barrier/event timeout in seconds (default: 10.0)",
    )
    cli.parser.add_argument(
        "--drain-timeout",
        type=float,
        default=5.0,
        help="Master drain wait timeout in seconds (default: 5.0)",
    )
    cli.parser.add_argument(
        "--ipc-dir",
        default=".ipc",
        help="Directory used for IPC socket files (default: .ipc)",
    )

    args = cli.parse()

    if args.duration <= 0:
        raise ValueError("--duration must be > 0")
    if args.flush_interval <= 0:
        raise ValueError("--flush-interval must be > 0")
    if args.startup_timeout <= 0:
        raise ValueError("--startup-timeout must be > 0")
    if args.drain_timeout <= 0:
        raise ValueError("--drain-timeout must be > 0")
    if args.workers <= 0:
        raise ValueError("--workers must be > 0")
    if args.message_size <= 0:
        raise ValueError("--message-size must be > 0")

    message_sizes = _parse_positive_int_csv(args.message_sizes, "--message-sizes")
    worker_counts = _parse_positive_int_csv(args.worker_counts, "--worker-counts")

    config = AdvancedLoggingBenchmarkConfig(
        num_operations=max(1, args.operations),
        warmup_operations=max(0, args.warmup),
        axis=args.axis,
        fixed_workers=args.workers,
        fixed_message_size=args.message_size,
        message_sizes=message_sizes,
        worker_counts=worker_counts,
        duration_sec=args.duration,
        flush_interval_s=args.flush_interval,
        startup_timeout_s=args.startup_timeout,
        drain_timeout_s=args.drain_timeout,
        ipc_dir=args.ipc_dir,
    )

    benchmark = AdvancedLoggingBenchmark(config)
    stats = benchmark.run()

    reporter = BenchmarkReporter(
        "Advanced Logging Benchmark Results",
        {
            "Architecture": "1 master + N workers",
            "Axis": config.axis,
            "Fixed workers": config.fixed_workers,
            "Fixed message size (B)": config.fixed_message_size,
            "Duration per run (s)": config.duration_sec,
            "Repeats": config.num_operations,
            "Warmup repeats": config.warmup_operations,
            "Flush interval (s)": config.flush_interval_s,
            "IPC dir": config.ipc_dir,
        },
    )
    reporter.print_full_report(stats, warmup=config.warmup_operations)

    if benchmark.message_size_axis_rows:
        _print_axis_summary(
            title="Message-Size Sweep (fixed workers)",
            axis_label="MsgSize",
            rows=benchmark.message_size_axis_rows,
        )
        comparative = ComparativeReporter(
            "Message-Size Sweep (mean ns/message)",
            size_label="MsgSize",
        )
        comparative.print_comparative_table(
            benchmark.message_size_axis_stats,
            ["produced_ns_per_msg", "consumed_ns_per_msg"],
        )

    if benchmark.worker_count_axis_rows:
        _print_axis_summary(
            title="Worker-Count Sweep (fixed message size)",
            axis_label="Workers",
            rows=benchmark.worker_count_axis_rows,
        )
        comparative = ComparativeReporter(
            "Worker-Count Sweep (mean ns/message)",
            size_label="Workers",
        )
        comparative.print_comparative_table(
            benchmark.worker_count_axis_stats,
            ["produced_ns_per_msg", "consumed_ns_per_msg"],
        )


if __name__ == "__main__":
    main()
