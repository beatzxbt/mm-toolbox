"""Performance benchmark for IPC ring buffer across processes.

Usage:
    uv run python benchmarks/ringbuffer/benchmark_ipc.py [--size SIZE]
    uv run python benchmarks/ringbuffer/benchmark_ipc.py --multi-size

Measures producer/consumer throughput for:
- sync single message mode
- sync packed batch mode
- async single message mode
- async packed batch mode
"""

from __future__ import annotations

import asyncio
import multiprocessing
import os
import time
from dataclasses import dataclass, field
from multiprocessing import Queue
from pathlib import Path
from queue import Empty as QueueEmpty

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
from mm_toolbox.ringbuffer.ipc import (
    IPCRingBufferConfig,
    IPCRingBufferConsumer,
    IPCRingBufferProducer,
)


def _producer_sync_single(
    path: str,
    payload_size: int,
    duration_sec: float,
    result_queue: Queue,
    barrier: multiprocessing.Barrier,
) -> None:
    """Producer process for sync single benchmark."""
    config = IPCRingBufferConfig(
        path=path,
        backlog=2**16,
        num_producers=1,
        num_consumers=1,
    )
    producer = IPCRingBufferProducer(config)
    payload = b"x" * payload_size

    barrier.wait()
    time.sleep(0.1)

    start_ns = time.perf_counter_ns()
    end_time_ns = start_ns + int(duration_sec * 1e9)
    count = 0

    while time.perf_counter_ns() < end_time_ns:
        producer.insert(payload, copy=False)
        count += 1

    actual_end_ns = time.perf_counter_ns()
    producer.stop()
    result_queue.put(("producer", actual_end_ns - start_ns, count))


def _consumer_sync_single(
    path: str,
    duration_sec: float,
    result_queue: Queue,
    barrier: multiprocessing.Barrier,
) -> None:
    """Consumer process for sync single benchmark."""
    config = IPCRingBufferConfig(
        path=path,
        backlog=2**16,
        num_producers=1,
        num_consumers=1,
    )
    consumer = IPCRingBufferConsumer(config)

    barrier.wait()
    time.sleep(0.1)

    start_ns = time.perf_counter_ns()
    end_time_ns = start_ns + int(duration_sec * 1e9)
    count = 0

    while time.perf_counter_ns() < end_time_ns:
        batch = consumer.consume_all()
        if batch:
            count += len(batch)
        else:
            time.sleep(0.001)

    actual_end_ns = time.perf_counter_ns()
    consumer.stop()
    result_queue.put(("consumer", actual_end_ns - start_ns, count))


def _producer_sync_packed(
    path: str,
    payload_size: int,
    duration_sec: float,
    batch_size: int,
    result_queue: Queue,
    barrier: multiprocessing.Barrier,
) -> None:
    """Producer process for sync packed benchmark."""
    config = IPCRingBufferConfig(
        path=path,
        backlog=2**16,
        num_producers=1,
        num_consumers=1,
    )
    producer = IPCRingBufferProducer(config)
    payload = b"x" * payload_size
    batches = [payload] * batch_size

    barrier.wait()
    time.sleep(0.1)

    start_ns = time.perf_counter_ns()
    end_time_ns = start_ns + int(duration_sec * 1e9)
    count = 0

    while time.perf_counter_ns() < end_time_ns:
        producer.insert_packed(batches, copy=False)
        count += batch_size

    actual_end_ns = time.perf_counter_ns()
    producer.stop()
    result_queue.put(("producer", actual_end_ns - start_ns, count))


def _consumer_sync_packed(
    path: str,
    duration_sec: float,
    result_queue: Queue,
    barrier: multiprocessing.Barrier,
) -> None:
    """Consumer process for sync packed benchmark."""
    import zmq

    config = IPCRingBufferConfig(
        path=path,
        backlog=2**16,
        num_producers=1,
        num_consumers=1,
    )
    consumer = IPCRingBufferConsumer(config)

    barrier.wait()
    time.sleep(0.1)

    start_ns = time.perf_counter_ns()
    end_time_ns = start_ns + int(duration_sec * 1e9)
    count = 0

    while time.perf_counter_ns() < end_time_ns:
        try:
            buf = consumer._socket.recv(flags=zmq.DONTWAIT)
            buf_len = len(buf)
            buf_mv = memoryview(buf)
            offset = 0
            while offset + 4 <= buf_len:
                length = int.from_bytes(buf_mv[offset : offset + 4], "little")
                offset += 4
                if offset + length > buf_len:
                    break
                count += 1
                offset += length
        except zmq.Again:
            time.sleep(0.001)

    actual_end_ns = time.perf_counter_ns()
    consumer.stop()
    result_queue.put(("consumer", actual_end_ns - start_ns, count))


async def _producer_async_single(
    path: str,
    payload_size: int,
    duration_sec: float,
    result_queue: Queue,
) -> None:
    """Producer process for async single benchmark."""
    config = IPCRingBufferConfig(
        path=path,
        backlog=2**16,
        num_producers=1,
        num_consumers=1,
    )
    producer = IPCRingBufferProducer(config)
    payload = b"x" * payload_size

    await asyncio.sleep(0.1)

    start_ns = time.perf_counter_ns()
    end_time_ns = start_ns + int(duration_sec * 1e9)
    count = 0

    while time.perf_counter_ns() < end_time_ns:
        producer.insert(payload, copy=False)
        count += 1

    actual_end_ns = time.perf_counter_ns()
    producer.stop()
    result_queue.put(("producer", actual_end_ns - start_ns, count))


async def _consumer_async_single(
    path: str,
    duration_sec: float,
    result_queue: Queue,
) -> None:
    """Consumer process for async single benchmark."""
    config = IPCRingBufferConfig(
        path=path,
        backlog=2**16,
        num_producers=1,
        num_consumers=1,
    )
    consumer = IPCRingBufferConsumer(config)

    await asyncio.sleep(0.1)

    start_ns = time.perf_counter_ns()
    end_time_ns = start_ns + int(duration_sec * 1e9)
    count = 0

    while time.perf_counter_ns() < end_time_ns:
        batch = consumer.consume_all()
        if batch:
            count += len(batch)
        else:
            await asyncio.sleep(0.001)

    actual_end_ns = time.perf_counter_ns()
    consumer.stop()
    result_queue.put(("consumer", actual_end_ns - start_ns, count))


async def _producer_async_packed(
    path: str,
    payload_size: int,
    duration_sec: float,
    batch_size: int,
    result_queue: Queue,
) -> None:
    """Producer process for async packed benchmark."""
    config = IPCRingBufferConfig(
        path=path,
        backlog=2**16,
        num_producers=1,
        num_consumers=1,
    )
    producer = IPCRingBufferProducer(config)
    payload = b"x" * payload_size
    batches = [payload] * batch_size

    await asyncio.sleep(0.1)

    start_ns = time.perf_counter_ns()
    end_time_ns = start_ns + int(duration_sec * 1e9)
    count = 0

    while time.perf_counter_ns() < end_time_ns:
        producer.insert_packed(batches, copy=False)
        count += batch_size

    actual_end_ns = time.perf_counter_ns()
    producer.stop()
    result_queue.put(("producer", actual_end_ns - start_ns, count))


async def _consumer_async_packed(
    path: str,
    duration_sec: float,
    result_queue: Queue,
) -> None:
    """Consumer process for async packed benchmark."""
    config = IPCRingBufferConfig(
        path=path,
        backlog=2**16,
        num_producers=1,
        num_consumers=1,
    )
    consumer = IPCRingBufferConsumer(config)

    await asyncio.sleep(0.1)

    start_ns = time.perf_counter_ns()
    end_time_ns = start_ns + int(duration_sec * 1e9)
    count = 0

    while time.perf_counter_ns() < end_time_ns:
        batch = consumer.consume_all()
        if batch:
            count += len(batch)
        else:
            await asyncio.sleep(0.001)

    actual_end_ns = time.perf_counter_ns()
    consumer.stop()
    result_queue.put(("consumer", actual_end_ns - start_ns, count))


def _run_producer_async_single(
    path: str,
    payload_size: int,
    duration_sec: float,
    result_queue: Queue,
) -> None:
    """Wrapper to run async producer in process."""
    asyncio.run(_producer_async_single(path, payload_size, duration_sec, result_queue))


def _run_consumer_async_single(
    path: str,
    duration_sec: float,
    result_queue: Queue,
) -> None:
    """Wrapper to run async consumer in process."""
    asyncio.run(_consumer_async_single(path, duration_sec, result_queue))


def _run_producer_async_packed(
    path: str,
    payload_size: int,
    duration_sec: float,
    batch_size: int,
    result_queue: Queue,
) -> None:
    """Wrapper to run async producer in process."""
    asyncio.run(
        _producer_async_packed(path, payload_size, duration_sec, batch_size, result_queue)
    )


def _run_consumer_async_packed(
    path: str,
    duration_sec: float,
    result_queue: Queue,
) -> None:
    """Wrapper to run async consumer in process."""
    asyncio.run(_consumer_async_packed(path, duration_sec, result_queue))


def _join_processes(
    processes: list[multiprocessing.Process],
    timeout_sec: float,
    mode_name: str,
) -> None:
    """Join processes with timeout and fail fast on hangs."""
    for proc in processes:
        proc.join(timeout=timeout_sec)

    alive = [proc for proc in processes if proc.is_alive()]
    if alive:
        for proc in alive:
            proc.terminate()
            proc.join(timeout=1.0)
        raise RuntimeError(f"{mode_name}: timed out waiting for subprocesses to finish")

    bad_exit = [proc for proc in processes if proc.exitcode not in (0, None)]
    if bad_exit:
        details = ", ".join(f"pid={proc.pid} exit={proc.exitcode}" for proc in bad_exit)
        raise RuntimeError(f"{mode_name}: subprocess failure ({details})")


def _read_result(
    result_queue: Queue,
    timeout_sec: float,
    mode_name: str,
) -> tuple[str, int, int]:
    """Read one benchmark result tuple from queue with timeout."""
    try:
        role, duration_ns, count = result_queue.get(timeout=timeout_sec)
    except QueueEmpty as exc:
        raise RuntimeError(f"{mode_name}: timed out waiting for benchmark result") from exc

    if role not in ("producer", "consumer"):
        raise RuntimeError(f"{mode_name}: unexpected result role {role!r}")

    return role, int(duration_ns), int(count)


def benchmark_sync_single(
    payload_size: int, duration_sec: float, path: str
) -> tuple[int, int, int, int]:
    """Benchmark synchronous single message throughput across processes."""
    result_queue: Queue = Queue()
    barrier = multiprocessing.Barrier(2)

    prod_proc = multiprocessing.Process(
        target=_producer_sync_single,
        args=(path, payload_size, duration_sec, result_queue, barrier),
    )
    cons_proc = multiprocessing.Process(
        target=_consumer_sync_single,
        args=(path, duration_sec, result_queue, barrier),
    )

    cons_proc.start()
    prod_proc.start()

    timeout_sec = max(30.0, duration_sec * 20.0)
    _join_processes([cons_proc, prod_proc], timeout_sec, "sync_single")

    first_role, first_ns, first_count = _read_result(
        result_queue, timeout_sec, "sync_single"
    )
    second_role, second_ns, second_count = _read_result(
        result_queue, timeout_sec, "sync_single"
    )

    role_map = {
        first_role: (first_ns, first_count),
        second_role: (second_ns, second_count),
    }
    if "producer" not in role_map or "consumer" not in role_map:
        raise RuntimeError("sync_single: missing producer/consumer results")

    prod_ns, prod_count = role_map["producer"]
    cons_ns, cons_count = role_map["consumer"]

    return prod_ns, cons_ns, prod_count, cons_count


def benchmark_sync_packed(
    payload_size: int, duration_sec: float, batch_size: int, path: str
) -> tuple[int, int, int, int]:
    """Benchmark synchronous packed batch throughput across processes."""
    result_queue: Queue = Queue()
    barrier = multiprocessing.Barrier(2)

    prod_proc = multiprocessing.Process(
        target=_producer_sync_packed,
        args=(path, payload_size, duration_sec, batch_size, result_queue, barrier),
    )
    cons_proc = multiprocessing.Process(
        target=_consumer_sync_packed,
        args=(path, duration_sec, result_queue, barrier),
    )

    cons_proc.start()
    prod_proc.start()

    timeout_sec = max(30.0, duration_sec * 20.0)
    _join_processes([cons_proc, prod_proc], timeout_sec, "sync_packed")

    first_role, first_ns, first_count = _read_result(
        result_queue, timeout_sec, "sync_packed"
    )
    second_role, second_ns, second_count = _read_result(
        result_queue, timeout_sec, "sync_packed"
    )

    role_map = {
        first_role: (first_ns, first_count),
        second_role: (second_ns, second_count),
    }
    if "producer" not in role_map or "consumer" not in role_map:
        raise RuntimeError("sync_packed: missing producer/consumer results")

    prod_ns, prod_count = role_map["producer"]
    cons_ns, cons_count = role_map["consumer"]

    return prod_ns, cons_ns, prod_count, cons_count


async def benchmark_async_single(
    payload_size: int, duration_sec: float, path: str
) -> tuple[int, int, int, int]:
    """Benchmark asynchronous single message throughput across processes."""
    prod_queue: Queue = Queue()
    cons_queue: Queue = Queue()

    prod_proc = multiprocessing.Process(
        target=_run_producer_async_single,
        args=(path, payload_size, duration_sec, prod_queue),
    )
    cons_proc = multiprocessing.Process(
        target=_run_consumer_async_single,
        args=(path, duration_sec, cons_queue),
    )

    cons_proc.start()
    prod_proc.start()

    timeout_sec = max(30.0, duration_sec * 20.0)
    _join_processes([cons_proc, prod_proc], timeout_sec, "async_single")

    prod_role, prod_ns, prod_count = _read_result(prod_queue, timeout_sec, "async_single")
    cons_role, cons_ns, cons_count = _read_result(cons_queue, timeout_sec, "async_single")

    if prod_role != "producer" or cons_role != "consumer":
        raise RuntimeError("async_single: producer/consumer result role mismatch")

    return prod_ns, cons_ns, prod_count, cons_count


async def benchmark_async_packed(
    payload_size: int, duration_sec: float, batch_size: int, path: str
) -> tuple[int, int, int, int]:
    """Benchmark asynchronous packed batch throughput across processes."""
    prod_queue: Queue = Queue()
    cons_queue: Queue = Queue()

    prod_proc = multiprocessing.Process(
        target=_run_producer_async_packed,
        args=(path, payload_size, duration_sec, batch_size, prod_queue),
    )
    cons_proc = multiprocessing.Process(
        target=_run_consumer_async_packed,
        args=(path, duration_sec, cons_queue),
    )

    cons_proc.start()
    prod_proc.start()

    timeout_sec = max(30.0, duration_sec * 20.0)
    _join_processes([cons_proc, prod_proc], timeout_sec, "async_packed")

    prod_role, prod_ns, prod_count = _read_result(prod_queue, timeout_sec, "async_packed")
    cons_role, cons_ns, cons_count = _read_result(cons_queue, timeout_sec, "async_packed")

    if prod_role != "producer" or cons_role != "consumer":
        raise RuntimeError("async_packed: producer/consumer result role mismatch")

    return prod_ns, cons_ns, prod_count, cons_count


@dataclass
class IPCBenchmarkConfig(BaseBenchmarkConfig):
    """Configuration for IPC ringbuffer benchmark."""

    payload_sizes: list[int] = field(
        default_factory=lambda: [64, 256, 1024, 4096, 16384, 65536, 262144]
    )
    duration_sec: float = 5.0
    batch_size: int = 100
    repeats: int = 1
    run_sync: bool = True
    run_async: bool = True


class IPCRingBufferBenchmark(BenchmarkRunner[IPCBenchmarkConfig]):
    """Benchmark runner for IPC ringbuffer."""

    def _create_subject(self) -> None:
        """No persistent subject is required for IPC benchmarks."""
        return None

    def _cleanup_socket(self, path: str) -> None:
        """Remove stale IPC socket path if present."""
        socket_path = path.replace("ipc://", "")
        p = Path(socket_path)
        if p.exists():
            p.unlink()

    def _record_throughput_sample(
        self,
        operation_name: str,
        payload_size: int,
        duration_ns: int,
        count: int,
    ) -> None:
        """Record ns-per-message sample derived from throughput run."""
        metrics = self.stats.add_operation(operation_name)
        ns_per_message = int(duration_ns / count) if count > 0 else int(duration_ns)
        metrics.add_latency(
            ns_per_message,
            payload_size=payload_size,
            duration_ns=duration_ns,
            total_messages=count,
        )

    def _record_mode_results(
        self,
        mode_name: str,
        payload_size: int,
        prod_ns: int,
        cons_ns: int,
        prod_count: int,
        cons_count: int,
    ) -> None:
        """Record producer/consumer samples for a mode run."""
        self._record_throughput_sample(
            f"{mode_name}.producer(payload={payload_size})",
            payload_size,
            prod_ns,
            prod_count,
        )
        self._record_throughput_sample(
            f"{mode_name}.consumer(payload={payload_size})",
            payload_size,
            cons_ns,
            cons_count,
        )

    def _run_benchmark_suite(self, _subject: None) -> None:
        """Run configured IPC benchmark suite."""
        ipc_dir = Path(".ipc")
        ipc_dir.mkdir(parents=True, exist_ok=True)
        base_path = f"ipc://{(ipc_dir / f'ringbuffer_bench_{os.getpid()}').resolve()}"

        for payload_size in self.config.payload_sizes:
            path = f"{base_path}_{payload_size}"

            for _ in range(self.config.repeats):
                if self.config.run_sync:
                    self._cleanup_socket(path)
                    prod_ns, cons_ns, prod_count, cons_count = benchmark_sync_single(
                        payload_size,
                        self.config.duration_sec,
                        path,
                    )
                    self._record_mode_results(
                        "sync_single",
                        payload_size,
                        prod_ns,
                        cons_ns,
                        prod_count,
                        cons_count,
                    )

                    self._cleanup_socket(path)
                    prod_ns, cons_ns, prod_count, cons_count = benchmark_sync_packed(
                        payload_size,
                        self.config.duration_sec,
                        self.config.batch_size,
                        path,
                    )
                    self._record_mode_results(
                        "sync_packed",
                        payload_size,
                        prod_ns,
                        cons_ns,
                        prod_count,
                        cons_count,
                    )

                if self.config.run_async:
                    self._cleanup_socket(path)
                    prod_ns, cons_ns, prod_count, cons_count = asyncio.run(
                        benchmark_async_single(
                            payload_size,
                            self.config.duration_sec,
                            path,
                        )
                    )
                    self._record_mode_results(
                        "async_single",
                        payload_size,
                        prod_ns,
                        cons_ns,
                        prod_count,
                        cons_count,
                    )

                    self._cleanup_socket(path)
                    prod_ns, cons_ns, prod_count, cons_count = asyncio.run(
                        benchmark_async_packed(
                            payload_size,
                            self.config.duration_sec,
                            self.config.batch_size,
                            path,
                        )
                    )
                    self._record_mode_results(
                        "async_packed",
                        payload_size,
                        prod_ns,
                        cons_ns,
                        prod_count,
                        cons_count,
                    )

            self._cleanup_socket(path)


def _parse_payload_sizes(raw: str) -> list[int]:
    """Parse comma-separated payload sizes."""
    values = [token.strip() for token in raw.split(",")]
    sizes = [int(token) for token in values if token]
    if not sizes:
        raise ValueError("At least one payload size is required")
    return sizes


def main() -> None:
    """Main entry point."""
    cli = BenchmarkCLI("Benchmark IPC ring buffer performance").add_size_arg(
        default=1024,
        help_text="Payload size in bytes for single-size runs (default: 1024)",
    )
    cli.parser.add_argument(
        "--payload-sizes",
        default="64,256,1024,4096,16384,65536,262144",
        help=(
            "Comma-separated payload sizes used with --multi-size "
            "(default: 64,256,1024,4096,16384,65536,262144)"
        ),
    )
    cli.parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Duration of each throughput run in seconds (default: 5.0)",
    )
    cli.parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for packed modes (default: 100)",
    )
    cli.parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of repetitions per payload/mode (default: 1)",
    )
    cli.parser.add_argument(
        "--sync-only",
        action="store_true",
        help="Run only synchronous modes",
    )
    cli.parser.add_argument(
        "--async-only",
        action="store_true",
        help="Run only asynchronous modes",
    )

    args = cli.parse()

    if args.sync_only and args.async_only:
        raise ValueError("--sync-only and --async-only are mutually exclusive")

    if args.multi_size:
        payload_sizes = _parse_payload_sizes(args.payload_sizes)
    else:
        payload_sizes = [args.size]

    config = IPCBenchmarkConfig(
        num_operations=args.operations,
        warmup_operations=args.warmup,
        payload_sizes=payload_sizes,
        duration_sec=args.duration,
        batch_size=args.batch_size,
        repeats=max(1, args.repeats),
        run_sync=not args.async_only,
        run_async=not args.sync_only,
    )

    benchmark = IPCRingBufferBenchmark(config)
    stats = benchmark.run()

    modes = []
    if config.run_sync:
        modes.append("sync")
    if config.run_async:
        modes.append("async")

    reporter = BenchmarkReporter(
        "IPC Ring Buffer Benchmark Results",
        {
            "Payload sizes": ", ".join(str(size) for size in payload_sizes),
            "Duration (sec)": config.duration_sec,
            "Batch size": config.batch_size,
            "Repeats": config.repeats,
            "Modes": ", ".join(modes),
        },
    )
    reporter.print_full_report(stats)


if __name__ == "__main__":
    main()
