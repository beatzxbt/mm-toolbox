"""Performance benchmark for MPSC SHM ring buffer.

Usage:
    uv run python benchmarks/ringbuffer/benchmark_shm_mpsc.py [--size SIZE]
    uv run python benchmarks/ringbuffer/benchmark_shm_mpsc.py --multi-size
    uv run python benchmarks/ringbuffer/benchmark_shm_mpsc.py --producers 4 --size 128

Measures insert/consume latency (single producer) and multi-producer throughput
for the MPSC shared-memory ring buffer implementation.
"""

from __future__ import annotations

import atexit
import multiprocessing
import os
import queue
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
    ShmMpscConsumer,
    ShmMpscProducer,
)


_shm_cleanup_paths = []


def _register_shm_cleanup(path: str) -> None:
    _shm_cleanup_paths.append(path)


def _shm_atexit_cleanup() -> None:
    for path in _shm_cleanup_paths:
        try:
            p = Path(path)
            if p.exists():
                p.unlink()
        except Exception:
            pass


atexit.register(_shm_atexit_cleanup)


@dataclass
class MPSCSHMBenchmarkConfig(BaseBenchmarkConfig):
    """Configuration for MPSC SHM ringbuffer benchmark."""

    capacity_bytes: int = 2**20
    num_rings: int = 0
    payload_sizes: list[int] = field(default_factory=lambda: [128, 512, 1024])
    latency_iterations: int = 4_096
    throughput_duration_sec: float = 1.0
    run_latency: bool = True
    run_throughput: bool = True
    run_scalability: bool = True
    run_fairness: bool = True
    throughput_producer_counts: list[int] = field(default_factory=lambda: [1, 2, 4])
    scalability_producer_counts: list[int] = field(default_factory=lambda: [1, 4])


def latency_benchmark_insert(
    capacity_bytes: int,
    payload_size: int,
    num_iterations: int,
    path: str,
) -> np.ndarray:
    """Benchmark insert latency (single producer, no consumer pressure).

    Uses num_rings=1 to avoid message dropping and ensure deterministic
    single-producer behavior.
    """
    num_rings = 1
    producer = ShmMpscProducer(
        path, capacity_bytes, num_rings=num_rings, create=True, unlink_on_close=True
    )
    payload = b"x" * payload_size
    latencies = np.zeros(num_iterations, dtype=np.int64)

    for _ in range(min(1000, num_iterations // 10)):
        producer.insert(payload)

    producer.close()

    producer = ShmMpscProducer(
        path, capacity_bytes, num_rings=num_rings, create=True, unlink_on_close=True
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
    """Benchmark consume latency (pre-filled buffer).

    Uses num_rings=1 to avoid message dropping and ensure all pre-filled
    messages are available for consumption.
    """
    num_rings = 1
    producer = ShmMpscProducer(
        path, capacity_bytes, num_rings=num_rings, create=True, unlink_on_close=False
    )
    payload = b"x" * payload_size

    for _ in range(num_iterations):
        producer.insert(payload)

    consumer = ShmMpscConsumer(path)
    latencies = np.zeros(num_iterations, dtype=np.int64)

    for i in range(num_iterations):
        start = time.perf_counter_ns()
        consumer.consume()
        latencies[i] = time.perf_counter_ns() - start

    consumer.close()
    producer.close()
    return latencies


def _producer_process(
    path: str,
    capacity_bytes: int,
    num_rings: int,
    payload_size: int,
    duration_sec: float,
    result_queue: Queue,
    barrier: multiprocessing.Barrier,
    producer_id: int = 0,
) -> None:
    """Producer process for throughput benchmark."""
    producer = ShmMpscProducer(
        path, capacity_bytes, num_rings=num_rings, create=False, unlink_on_close=False
    )
    payload = bytes([producer_id & 0xFF]) + b"x" * (payload_size - 1)
    try:
        barrier.wait()

        start_ns = time.perf_counter_ns()
        end_time_ns = start_ns + int(duration_sec * 1e9)
        successful_inserts = 0
        dropped_messages = 0
        while time.perf_counter_ns() < end_time_ns:
            if producer.insert(payload):
                successful_inserts += 1
                dropped_messages += producer.last_insert_dropped

        # Tail inserts to unblock a consumer that is waiting at the deadline.
        for _ in range(10):
            if producer.insert(payload):
                successful_inserts += 1
                dropped_messages += producer.last_insert_dropped

        actual_end_ns = time.perf_counter_ns()
        result_queue.put(
            (
                "producer",
                producer_id,
                actual_end_ns - start_ns,
                successful_inserts,
                producer.failed_inserts,
                dropped_messages,
            )
        )
    finally:
        producer.close()


def _consumer_process(
    path: str,
    payload_size: int,
    duration_sec: float,
    result_queue: Queue,
    barrier: multiprocessing.Barrier,
    track_fairness: bool = False,
) -> None:
    """Consumer process for throughput benchmark."""
    barrier.wait()

    consumer = ShmMpscConsumer(path)

    # Pre-allocate a pool of buffers to eliminate allocation overhead
    pool_size = 1024
    buffer_pool = [bytearray(payload_size) for _ in range(pool_size)]

    start_ns = time.perf_counter_ns()
    end_time_ns = start_ns + int(duration_sec * 1e9)
    count = 0
    per_producer_counts: dict[int, int] = {}

    try:
        while time.perf_counter_ns() < end_time_ns:
            copied = consumer.consume_all_into(buffer_pool)
            count += copied
            if track_fairness:
                for i in range(copied):
                    msg = buffer_pool[i]
                    if msg:
                        pid = msg[0]
                        per_producer_counts[pid] = per_producer_counts.get(pid, 0) + 1
        actual_end_ns = time.perf_counter_ns()
        result_queue.put(
            ("consumer", actual_end_ns - start_ns, count, per_producer_counts)
        )
    finally:
        consumer.close()


def _join_worker_processes(
    processes: list[multiprocessing.Process], timeout_sec: float, benchmark_name: str
) -> None:
    """Wait for all workers and fail before result collection if one stopped badly."""
    deadline = time.monotonic() + timeout_sec
    for process in processes:
        process.join(timeout=max(0.0, deadline - time.monotonic()))

    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
        if process.is_alive():
            process.kill()
            process.join(timeout=1.0)

    if any(process.is_alive() for process in processes):
        raise RuntimeError(f"{benchmark_name} worker did not stop after termination")

    failed_processes = [
        (process.name, process.exitcode)
        for process in processes
        if process.exitcode != 0
    ]
    if failed_processes:
        raise RuntimeError(f"{benchmark_name} worker failed: {failed_processes}")


def throughput_benchmark(
    capacity_bytes: int,
    num_rings: int,
    payload_size: int,
    duration_sec: float,
    num_producers: int,
    path: str,
) -> tuple[int, int, int, int, list[tuple[int, int, int, int, int]]]:
    """Benchmark multi-producer throughput across processes.

    Returns:
        Tuple of (consumer_ns, consumer_count, overwritten, remaining,
        producer results).
        Each producer result contains its ID, elapsed time, successful inserts,
        failed inserts, and overwritten records.
    """
    # Create the ring in the main process so child processes can attach
    creator = ShmMpscProducer(
        path, capacity_bytes, num_rings=num_rings, create=True, unlink_on_close=False
    )
    creator.close()

    result_queue: Queue = Queue()
    barrier = multiprocessing.Barrier(num_producers + 1)

    producer_procs = []
    for i in range(num_producers):
        proc = multiprocessing.Process(
            target=_producer_process,
            args=(
                path,
                capacity_bytes,
                num_rings,
                payload_size,
                duration_sec,
                result_queue,
                barrier,
                i,
            ),
        )
        producer_procs.append(proc)

    cons_proc = multiprocessing.Process(
        target=_consumer_process,
        args=(path, payload_size, duration_sec, result_queue, barrier),
    )

    processes = [cons_proc, *producer_procs]
    try:
        for process in processes:
            process.start()

        # Give extra time for barrier synchronization and process startup.
        _join_worker_processes(processes, duration_sec + 10.0, "MPSC throughput")

        consumer_result = None
        producer_results = []
        for _ in processes:
            result = result_queue.get(timeout=5)
            if result[0] == "consumer":
                consumer_result = result
            else:
                producer_results.append(result[1:])

        if consumer_result is None or len(producer_results) != num_producers:
            raise RuntimeError("MPSC throughput workers returned incomplete results")

        _, cons_ns, cons_count, _ = consumer_result
        dropped_count = sum(dropped for _, _, _, _, dropped in producer_results)
        successful_count = sum(count for _, _, count, _, _ in producer_results)
        consumer = ShmMpscConsumer(path)
        try:
            remaining_count = len(consumer)
        finally:
            consumer.close()
        accounted_count = cons_count + dropped_count + remaining_count
        if successful_count != accounted_count:
            raise RuntimeError(
                "MPSC message accounting mismatch: "
                f"published={successful_count}, consumed={cons_count}, "
                f"dropped={dropped_count}, remaining={remaining_count}"
            )
        return cons_ns, cons_count, dropped_count, remaining_count, producer_results
    except queue.Empty as exc:
        raise RuntimeError(
            "MPSC throughput workers did not return every result"
        ) from exc
    finally:
        for process in processes:
            if process.pid is not None and process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
            if process.pid is not None and not process.is_alive():
                process.close()
        result_queue.close()
        result_queue.join_thread()


def fairness_benchmark(
    capacity_bytes: int,
    num_rings: int,
    payload_size: int,
    duration_sec: float,
    path: str,
) -> tuple[
    int,
    int,
    dict[int, int],
    int,
    int,
    list[tuple[int, int, int, int, int]],
]:
    """Fairness benchmark with 4 producers and per-producer tracking.

    Returns:
        Tuple of (consumer_ns, consumer_count, per_producer_consume_counts,
                  overwritten, remaining, producer results).
        Each producer result contains its ID,
                  elapsed time, successful inserts, failed inserts, and
                  overwritten records.
    """
    # Create the ring in the main process so child processes can attach
    creator = ShmMpscProducer(
        path, capacity_bytes, num_rings=num_rings, create=True, unlink_on_close=False
    )
    creator.close()

    num_producers = 4
    result_queue: Queue = Queue()
    barrier = multiprocessing.Barrier(num_producers + 1)

    producer_procs = []
    for i in range(num_producers):
        proc = multiprocessing.Process(
            target=_producer_process,
            args=(
                path,
                capacity_bytes,
                num_rings,
                payload_size,
                duration_sec,
                result_queue,
                barrier,
                i,
            ),
        )
        producer_procs.append(proc)

    cons_proc = multiprocessing.Process(
        target=_consumer_process,
        args=(path, payload_size, duration_sec, result_queue, barrier, True),
    )

    processes = [cons_proc, *producer_procs]
    try:
        for process in processes:
            process.start()

        _join_worker_processes(processes, duration_sec + 10.0, "MPSC fairness")

        consumer_result = None
        producer_results = []
        for _ in processes:
            result = result_queue.get(timeout=5)
            if result[0] == "consumer":
                consumer_result = result
            else:
                producer_results.append(result[1:])

        if consumer_result is None or len(producer_results) != num_producers:
            raise RuntimeError("MPSC fairness workers returned incomplete results")

        _, cons_ns, cons_count, per_producer_counts = consumer_result
        dropped_count = sum(dropped for _, _, _, _, dropped in producer_results)
        successful_count = sum(count for _, _, count, _, _ in producer_results)
        consumer = ShmMpscConsumer(path)
        try:
            remaining_count = len(consumer)
        finally:
            consumer.close()
        accounted_count = cons_count + dropped_count + remaining_count
        if successful_count != accounted_count:
            raise RuntimeError(
                "MPSC fairness message accounting mismatch: "
                f"published={successful_count}, consumed={cons_count}, "
                f"dropped={dropped_count}, remaining={remaining_count}"
            )
        return (
            cons_ns,
            cons_count,
            per_producer_counts,
            dropped_count,
            remaining_count,
            producer_results,
        )
    except queue.Empty as exc:
        raise RuntimeError("MPSC fairness workers did not return every result") from exc
    finally:
        for process in processes:
            if process.pid is not None and process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
            if process.pid is not None and not process.is_alive():
                process.close()
        result_queue.close()
        result_queue.join_thread()


class ShmMpscRingBufferBenchmark(BenchmarkRunner[MPSCSHMBenchmarkConfig]):
    """Benchmark runner for MPSC SHM ring buffer."""

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
        """Run configured MPSC SHM benchmark suite."""
        base_path = f"/tmp/shm_mpsc_bench_{os.getpid()}"
        config = self.config

        for payload_size in config.payload_sizes:
            if config.run_latency:
                # Latency benchmarks run without a concurrent consumer, so keep
                # the sample count below ring capacity for larger payloads.
                latency_iterations = min(
                    config.latency_iterations,
                    max(1, config.capacity_bytes // (payload_size + 256)),
                )
                lat_path = f"{base_path}_lat_{payload_size}"
                _register_shm_cleanup(lat_path)
                self._cleanup_path(lat_path)

                try:
                    insert_latencies = latency_benchmark_insert(
                        config.capacity_bytes,
                        payload_size,
                        latency_iterations,
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
                        config.capacity_bytes,
                        payload_size,
                        latency_iterations,
                        lat_path,
                    )
                    self._record_latency_series(
                        f"consume(payload={payload_size})",
                        payload_size,
                        consume_latencies,
                    )
                finally:
                    self._cleanup_path(lat_path)

            if config.run_throughput:
                for num_producers in config.throughput_producer_counts:
                    tp_path = f"{base_path}_tp_{payload_size}_p{num_producers}"
                    _register_shm_cleanup(tp_path)
                    self._cleanup_path(tp_path)

                    try:
                        (
                            cons_ns,
                            cons_count,
                            dropped_count,
                            remaining_count,
                            prod_results,
                        ) = throughput_benchmark(
                            config.capacity_bytes,
                            config.num_rings,
                            payload_size,
                            config.throughput_duration_sec,
                            num_producers,
                            tp_path,
                        )
                        total_prod_count = sum(
                            count for _, _, count, _, _ in prod_results
                        )
                        failed_insert_count = sum(
                            failed for _, _, _, failed, _ in prod_results
                        )
                        producer_duration_ns = max(
                            duration for _, duration, _, _, _ in prod_results
                        )
                        self._record_throughput_sample(
                            f"throughput_producer_aggregate(payload={payload_size},producers={num_producers})",
                            payload_size,
                            producer_duration_ns,
                            total_prod_count,
                        )
                        self._record_throughput_sample(
                            f"throughput_consumer(payload={payload_size},producers={num_producers})",
                            payload_size,
                            cons_ns,
                            cons_count,
                        )
                        for pid, prod_ns, prod_count, _, _ in prod_results:
                            self._record_throughput_sample(
                                f"throughput_producer{pid}(payload={payload_size},producers={num_producers})",
                                payload_size,
                                prod_ns,
                                prod_count,
                            )
                        print(
                            f"MPSC payload={payload_size} producers={num_producers}: "
                            f"successful inserts={total_prod_count:,}, "
                            f"failed inserts={failed_insert_count:,}, "
                            f"overwritten={dropped_count:,}, "
                            f"consumer received={cons_count:,}, "
                            f"remaining={remaining_count:,}",
                            flush=True,
                        )
                    finally:
                        self._cleanup_path(tp_path)

        if config.run_scalability:
            payload_size = 128
            for num_producers in config.scalability_producer_counts:
                scal_path = f"{base_path}_scal_p{num_producers}"
                _register_shm_cleanup(scal_path)
                self._cleanup_path(scal_path)
                try:
                    (
                        cons_ns,
                        cons_count,
                        dropped_count,
                        remaining_count,
                        prod_results,
                    ) = throughput_benchmark(
                        config.capacity_bytes,
                        config.num_rings,
                        payload_size,
                        config.throughput_duration_sec,
                        num_producers,
                        scal_path,
                    )
                    total_prod_count = sum(count for _, _, count, _, _ in prod_results)
                    failed_insert_count = sum(
                        failed for _, _, _, failed, _ in prod_results
                    )
                    producer_duration_ns = max(
                        duration for _, duration, _, _, _ in prod_results
                    )
                    self._record_throughput_sample(
                        f"scalability_producer_aggregate(producers={num_producers})",
                        payload_size,
                        producer_duration_ns,
                        total_prod_count,
                    )
                    self._record_throughput_sample(
                        f"scalability_consumer(producers={num_producers})",
                        payload_size,
                        cons_ns,
                        cons_count,
                    )
                    print(
                        f"MPSC scalability producers={num_producers}: "
                        f"successful inserts={total_prod_count:,}, "
                        f"failed inserts={failed_insert_count:,}, "
                        f"overwritten={dropped_count:,}, "
                        f"consumer received={cons_count:,}, "
                        f"remaining={remaining_count:,}",
                        flush=True,
                    )
                finally:
                    self._cleanup_path(scal_path)

        if config.run_fairness:
            fair_path = f"{base_path}_fair"
            _register_shm_cleanup(fair_path)
            self._cleanup_path(fair_path)
            payload_size = 128
            try:
                (
                    cons_ns,
                    cons_count,
                    per_producer_counts,
                    dropped_count,
                    remaining_count,
                    prod_results,
                ) = fairness_benchmark(
                    config.capacity_bytes,
                    config.num_rings,
                    payload_size,
                    config.throughput_duration_sec,
                    fair_path,
                )
                total_prod_count = sum(count for _, _, count, _, _ in prod_results)
                failed_insert_count = sum(failed for _, _, _, failed, _ in prod_results)
                producer_duration_ns = max(
                    duration for _, duration, _, _, _ in prod_results
                )
                self._record_throughput_sample(
                    "fairness_producer_aggregate",
                    payload_size,
                    producer_duration_ns,
                    total_prod_count,
                )
                self._record_throughput_sample(
                    "fairness_consumer",
                    payload_size,
                    cons_ns,
                    cons_count,
                )
                for pid, prod_ns, prod_count, _, _ in prod_results:
                    self._record_throughput_sample(
                        f"fairness_producer{pid}",
                        payload_size,
                        prod_ns,
                        prod_count,
                    )
                    consumed = per_producer_counts.get(pid, 0)
                    self._record_throughput_sample(
                        f"fairness_consumed_from_producer{pid}",
                        payload_size,
                        cons_ns,
                        consumed,
                    )
                print(
                    f"MPSC fairness: successful inserts={total_prod_count:,}, "
                    f"failed inserts={failed_insert_count:,}, "
                    f"overwritten={dropped_count:,}, "
                    f"consumer received={cons_count:,}, "
                    f"remaining={remaining_count:,}",
                    flush=True,
                )
            finally:
                self._cleanup_path(fair_path)


def _parse_payload_sizes(raw: str) -> list[int]:
    """Parse comma-separated payload sizes."""
    values = [token.strip() for token in raw.split(",")]
    sizes = [int(token) for token in values if token]
    if not sizes:
        raise ValueError("At least one payload size is required")
    return sizes


def _parse_int_list(raw: str) -> list[int]:
    """Parse comma-separated integers."""
    values = [token.strip() for token in raw.split(",")]
    ints = [int(token) for token in values if token]
    if not ints:
        raise ValueError("At least one integer is required")
    return ints


def main() -> None:
    """Main entry point."""
    cli = BenchmarkCLI("Benchmark MPSC SHM ring buffer performance").add_size_arg(
        default=128,
        help_text="Payload size in bytes for single-size runs (default: 128)",
    )
    cli.parser.add_argument(
        "--payload-sizes",
        default="128,512,1024",
        help=(
            "Comma-separated payload sizes used with --multi-size "
            "(default: 128,512,1024)"
        ),
    )
    cli.parser.add_argument(
        "--capacity-bytes",
        type=int,
        default=2**20,
        help="Ring buffer capacity in bytes (default: 1048576)",
    )
    cli.parser.add_argument(
        "--num-rings",
        type=int,
        default=0,
        help="Number of sub-rings (0 = auto = cpu_count, default: 0)",
    )
    cli.parser.add_argument(
        "--latency-iterations",
        type=int,
        default=4_096,
        help="Number of iterations per latency test (default: 4096)",
    )
    cli.parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
        help="Throughput benchmark duration in seconds (default: 1.0)",
    )
    cli.parser.add_argument(
        "--producers",
        type=int,
        default=None,
        help="Number of producers for throughput runs (overrides --throughput-producer-counts)",
    )
    cli.parser.add_argument(
        "--throughput-producer-counts",
        default="1,2,4",
        help="Comma-separated producer counts for throughput benchmark (default: 1,2,4)",
    )
    cli.parser.add_argument(
        "--scalability-producer-counts",
        default="1,4",
        help="Comma-separated producer counts for scalability benchmark (default: 1,4)",
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
    cli.parser.add_argument(
        "--skip-scalability",
        action="store_true",
        help="Skip scalability benchmark",
    )
    cli.parser.add_argument(
        "--skip-fairness",
        action="store_true",
        help="Skip fairness benchmark",
    )

    args = cli.parse()

    if args.latency_only and args.throughput_only:
        raise ValueError("--latency-only and --throughput-only are mutually exclusive")

    if args.multi_size:
        payload_sizes = _parse_payload_sizes(args.payload_sizes)
    else:
        payload_sizes = [args.size]

    run_latency = not args.throughput_only
    run_throughput = not args.latency_only

    if args.producers is not None:
        throughput_counts = [args.producers]
    else:
        throughput_counts = _parse_int_list(args.throughput_producer_counts)

    config = MPSCSHMBenchmarkConfig(
        num_operations=args.operations,
        warmup_operations=args.warmup,
        capacity_bytes=args.capacity_bytes,
        num_rings=args.num_rings,
        payload_sizes=payload_sizes,
        latency_iterations=args.latency_iterations,
        throughput_duration_sec=args.duration,
        run_latency=run_latency,
        run_throughput=run_throughput,
        run_scalability=run_throughput
        and not args.skip_scalability
        and args.producers is None,
        run_fairness=run_throughput
        and not args.skip_fairness
        and args.producers is None,
        throughput_producer_counts=throughput_counts,
        scalability_producer_counts=_parse_int_list(args.scalability_producer_counts),
    )

    benchmark = ShmMpscRingBufferBenchmark(config)
    stats = benchmark.run()

    reporter = BenchmarkReporter(
        "MPSC SHM Ring Buffer Benchmark Results",
        {
            "Capacity bytes": config.capacity_bytes,
            "Num rings": config.num_rings if config.num_rings > 0 else "auto",
            "Payload sizes": ", ".join(str(size) for size in payload_sizes),
            "Latency iterations": config.latency_iterations,
            "Throughput duration": config.throughput_duration_sec,
            "Throughput producer counts": ", ".join(
                str(n) for n in config.throughput_producer_counts
            ),
            "Scalability producer counts": ", ".join(
                str(n) for n in config.scalability_producer_counts
            ),
        },
    )
    reporter.print_full_report(stats)


if __name__ == "__main__":
    main()
