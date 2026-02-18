"""Performance benchmark for SHM ring buffer.

Usage:
    uv run python benchmarks/ringbuffer/benchmark_shm.py [--size SIZE]
    uv run python benchmarks/ringbuffer/benchmark_shm.py --multi-size

Measures insert/consume latency and producer/consumer throughput for the
shared memory ring buffer implementation.
"""

from __future__ import annotations

import multiprocessing
import os
import time
from dataclasses import dataclass, field
from multiprocessing import Queue
from pathlib import Path

import numpy as np

try:
    from benchmarks.core import (
        BaseBenchmarkConfig,
        BenchmarkCLI,
        BenchmarkReporter,
        BenchmarkRunner,
    )
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from benchmarks.core import (
        BaseBenchmarkConfig,
        BenchmarkCLI,
        BenchmarkReporter,
        BenchmarkRunner,
    )
from mm_toolbox.ringbuffer.shm import (
    SharedBytesRingBufferConsumer,
    SharedBytesRingBufferProducer,
)


@dataclass
class SHMBenchmarkConfig(BaseBenchmarkConfig):
    """Configuration for SHM ringbuffer benchmark."""

    capacity_bytes: int = 2**20
    payload_sizes: list[int] = field(default_factory=lambda: [32, 128, 512, 2048, 8192])
    latency_iterations: int = 100_000
    throughput_duration_sec: float = 3.0
    run_latency: bool = True
    run_throughput: bool = True


def latency_benchmark_insert(
    capacity_bytes: int,
    payload_size: int,
    num_iterations: int,
    path: str,
) -> np.ndarray:
    """Benchmark insert latency (producer only, no consumer pressure)."""
    producer = SharedBytesRingBufferProducer(
        path, capacity_bytes, create=True, unlink_on_close=True
    )
    payload = b"x" * payload_size
    latencies = np.zeros(num_iterations, dtype=np.int64)

    for _ in range(min(1000, num_iterations // 10)):
        producer.insert(payload)

    producer = SharedBytesRingBufferProducer(
        path, capacity_bytes, create=True, unlink_on_close=True
    )

    for i in range(num_iterations):
        start = time.perf_counter_ns()
        producer.insert(payload)
        latencies[i] = time.perf_counter_ns() - start

    return latencies


def latency_benchmark_consume(
    capacity_bytes: int,
    payload_size: int,
    num_iterations: int,
    path: str,
) -> np.ndarray:
    """Benchmark consume latency (pre-filled buffer)."""
    producer = SharedBytesRingBufferProducer(
        path, capacity_bytes, create=True, unlink_on_close=True
    )
    payload = b"x" * payload_size

    for _ in range(num_iterations):
        producer.insert(payload)

    consumer = SharedBytesRingBufferConsumer(path)
    latencies = np.zeros(num_iterations, dtype=np.int64)

    for i in range(num_iterations):
        start = time.perf_counter_ns()
        consumer.consume()
        latencies[i] = time.perf_counter_ns() - start

    return latencies


def _producer_process(
    path: str,
    capacity_bytes: int,
    payload_size: int,
    duration_sec: float,
    result_queue: Queue,
    barrier: multiprocessing.Barrier,
) -> None:
    """Producer process for throughput benchmark."""
    producer = SharedBytesRingBufferProducer(
        path, capacity_bytes, create=True, unlink_on_close=False
    )
    payload = b"x" * payload_size

    barrier.wait()
    time.sleep(0.05)

    start_ns = time.perf_counter_ns()
    end_time_ns = start_ns + int(duration_sec * 1e9)
    count = 0

    while time.perf_counter_ns() < end_time_ns:
        producer.insert(payload)
        count += 1

    actual_end_ns = time.perf_counter_ns()
    result_queue.put((actual_end_ns - start_ns, count))


def _consumer_process(
    path: str,
    capacity_bytes: int,
    duration_sec: float,
    result_queue: Queue,
    barrier: multiprocessing.Barrier,
) -> None:
    """Consumer process for throughput benchmark."""
    barrier.wait()

    consumer = SharedBytesRingBufferConsumer(path)

    start_ns = time.perf_counter_ns()
    end_time_ns = start_ns + int(duration_sec * 1e9)
    count = 0

    while time.perf_counter_ns() < end_time_ns:
        msg = consumer.peekleft()
        if msg is not None:
            consumer.consume()
            count += 1
        else:
            time.sleep(0.00001)

    actual_end_ns = time.perf_counter_ns()
    result_queue.put((actual_end_ns - start_ns, count))


def throughput_benchmark(
    capacity_bytes: int,
    payload_size: int,
    duration_sec: float,
    path: str,
) -> tuple[int, int, int, int]:
    """Benchmark producer-consumer throughput across processes."""
    result_queue: Queue = Queue()
    barrier = multiprocessing.Barrier(2)

    prod_proc = multiprocessing.Process(
        target=_producer_process,
        args=(path, capacity_bytes, payload_size, duration_sec, result_queue, barrier),
    )
    cons_proc = multiprocessing.Process(
        target=_consumer_process,
        args=(path, capacity_bytes, duration_sec, result_queue, barrier),
    )

    cons_proc.start()
    prod_proc.start()

    prod_proc.join()
    cons_proc.join()

    prod_ns, prod_count = result_queue.get()
    cons_ns, cons_count = result_queue.get()

    return prod_ns, cons_ns, prod_count, cons_count


class SHMRingBufferBenchmark(BenchmarkRunner[SHMBenchmarkConfig]):
    """Benchmark runner for SHM ring buffer."""

    def _create_subject(self) -> None:
        """No persistent subject is required for SHM benchmarks."""
        return None

    def _record_latency_series(
        self,
        operation_name: str,
        payload_size: int,
        latencies: np.ndarray,
    ) -> None:
        """Record all latency samples into shared benchmark stats."""
        metrics = self.stats.add_operation(operation_name)
        for latency in latencies:
            metrics.add_latency(int(latency), payload_size=payload_size)

    def _record_throughput_sample(
        self,
        operation_name: str,
        payload_size: int,
        duration_ns: int,
        count: int,
    ) -> None:
        """Record ns-per-message latency sample derived from throughput run."""
        metrics = self.stats.add_operation(operation_name)
        ns_per_message = int(duration_ns / count) if count > 0 else int(duration_ns)
        metrics.add_latency(
            ns_per_message,
            payload_size=payload_size,
            duration_ns=duration_ns,
            total_messages=count,
        )

    def _cleanup_path(self, path: str) -> None:
        """Remove stale shared-memory backing file if present."""
        p = Path(path)
        if p.exists():
            p.unlink()

    def _run_benchmark_suite(self, _subject: None) -> None:
        """Run configured SHM benchmark suite."""
        base_path = f"/tmp/shm_bench_{os.getpid()}"

        for payload_size in self.config.payload_sizes:
            if self.config.run_latency:
                lat_path = f"{base_path}_lat_{payload_size}"
                self._cleanup_path(lat_path)

                try:
                    insert_latencies = latency_benchmark_insert(
                        self.config.capacity_bytes,
                        payload_size,
                        self.config.latency_iterations,
                        lat_path,
                    )
                    self._record_latency_series(
                        f"insert(payload={payload_size})",
                        payload_size,
                        insert_latencies,
                    )
                finally:
                    self._cleanup_path(lat_path)

                try:
                    consume_latencies = latency_benchmark_consume(
                        self.config.capacity_bytes,
                        payload_size,
                        self.config.latency_iterations,
                        lat_path,
                    )
                    self._record_latency_series(
                        f"consume(payload={payload_size})",
                        payload_size,
                        consume_latencies,
                    )
                finally:
                    self._cleanup_path(lat_path)

            if self.config.run_throughput:
                tp_path = f"{base_path}_tp_{payload_size}"
                self._cleanup_path(tp_path)

                try:
                    prod_ns, cons_ns, prod_count, cons_count = throughput_benchmark(
                        self.config.capacity_bytes,
                        payload_size,
                        self.config.throughput_duration_sec,
                        tp_path,
                    )
                    self._record_throughput_sample(
                        f"throughput_producer(payload={payload_size})",
                        payload_size,
                        prod_ns,
                        prod_count,
                    )
                    self._record_throughput_sample(
                        f"throughput_consumer(payload={payload_size})",
                        payload_size,
                        cons_ns,
                        cons_count,
                    )
                finally:
                    self._cleanup_path(tp_path)


def _parse_payload_sizes(raw: str) -> list[int]:
    """Parse comma-separated payload sizes."""
    values = [token.strip() for token in raw.split(",")]
    sizes = [int(token) for token in values if token]
    if not sizes:
        raise ValueError("At least one payload size is required")
    return sizes


def main() -> None:
    """Main entry point."""
    cli = BenchmarkCLI("Benchmark SHM ring buffer performance").add_size_arg(
        default=512,
        help_text="Payload size in bytes for single-size runs (default: 512)",
    )
    cli.parser.add_argument(
        "--payload-sizes",
        default="32,128,512,2048,8192",
        help=(
            "Comma-separated payload sizes used with --multi-size "
            "(default: 32,128,512,2048,8192)"
        ),
    )
    cli.parser.add_argument(
        "--capacity-bytes",
        type=int,
        default=2**20,
        help="Ring buffer capacity in bytes (default: 1048576)",
    )
    cli.parser.add_argument(
        "--latency-iterations",
        type=int,
        default=100_000,
        help="Number of iterations per latency test (default: 100000)",
    )
    cli.parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="Throughput benchmark duration in seconds (default: 3.0)",
    )
    cli.parser.add_argument(
        "--latency-only",
        action="store_true",
        help="Run only latency benchmarks",
    )
    cli.parser.add_argument(
        "--throughput-only",
        action="store_true",
        help="Run only throughput benchmarks",
    )

    args = cli.parse()

    if args.latency_only and args.throughput_only:
        raise ValueError("--latency-only and --throughput-only are mutually exclusive")

    if args.multi_size:
        payload_sizes = _parse_payload_sizes(args.payload_sizes)
    else:
        payload_sizes = [args.size]

    config = SHMBenchmarkConfig(
        num_operations=args.operations,
        warmup_operations=args.warmup,
        capacity_bytes=args.capacity_bytes,
        payload_sizes=payload_sizes,
        latency_iterations=args.latency_iterations,
        throughput_duration_sec=args.duration,
        run_latency=not args.throughput_only,
        run_throughput=not args.latency_only,
    )

    benchmark = SHMRingBufferBenchmark(config)
    stats = benchmark.run()

    reporter = BenchmarkReporter(
        "SHM Ring Buffer Benchmark Results",
        {
            "Capacity bytes": config.capacity_bytes,
            "Payload sizes": ", ".join(str(size) for size in payload_sizes),
            "Latency iterations": config.latency_iterations,
            "Throughput duration": config.throughput_duration_sec,
        },
    )
    reporter.print_full_report(stats)


if __name__ == "__main__":
    main()
