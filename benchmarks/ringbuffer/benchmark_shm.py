"""Performance benchmark for SHM ring buffer.

Measures insert/consume latency and throughput for the shared memory
ring buffer implementation. Used to validate C optimization improvements.
"""

from __future__ import annotations

import multiprocessing
import os
import time
from multiprocessing import Queue
from pathlib import Path

import numpy as np

from mm_toolbox.ringbuffer.shm import (
    SharedBytesRingBufferConsumer,
    SharedBytesRingBufferProducer,
)


def latency_benchmark_insert(
    capacity_bytes: int,
    payload_size: int,
    num_iterations: int,
    path: str,
) -> np.ndarray:
    """Benchmark insert latency (producer only, no consumer pressure).

    Args:
        capacity_bytes: Ring buffer capacity in bytes.
        payload_size: Size of each message payload.
        num_iterations: Number of insert operations to time.
        path: Shared memory path.

    Returns:
        Array of latency measurements in nanoseconds.
    """
    producer = SharedBytesRingBufferProducer(
        path, capacity_bytes, create=True, unlink_on_close=True
    )
    payload = b"x" * payload_size
    latencies = np.zeros(num_iterations, dtype=np.int64)

    # Warmup
    for _ in range(min(1000, num_iterations // 10)):
        producer.insert(payload)

    # Reset buffer
    producer = SharedBytesRingBufferProducer(
        path, capacity_bytes, create=True, unlink_on_close=True
    )

    # Timed iterations
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

    Args:
        capacity_bytes: Ring buffer capacity in bytes.
        payload_size: Size of each message payload.
        num_iterations: Number of consume operations to time.
        path: Shared memory path.

    Returns:
        Array of latency measurements in nanoseconds.
    """
    producer = SharedBytesRingBufferProducer(
        path, capacity_bytes, create=True, unlink_on_close=True
    )
    payload = b"x" * payload_size

    # Pre-fill buffer
    for _ in range(num_iterations):
        producer.insert(payload)

    # Create consumer (attaches to existing shared memory)
    consumer = SharedBytesRingBufferConsumer(path)

    latencies = np.zeros(num_iterations, dtype=np.int64)

    # Timed iterations (buffer is pre-filled, so consume() won't block)
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
    time.sleep(0.05)  # Small delay to let consumer start

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
    # Wait for producer to create the shared memory
    barrier.wait()

    consumer = SharedBytesRingBufferConsumer(path)

    start_ns = time.perf_counter_ns()
    end_time_ns = start_ns + int(duration_sec * 1e9)
    count = 0

    while time.perf_counter_ns() < end_time_ns:
        msg = consumer.peekleft()
        if msg is not None:
            consumer.consume()  # Actually consume it
            count += 1
        else:
            time.sleep(0.00001)  # 10us backoff when empty

    actual_end_ns = time.perf_counter_ns()
    result_queue.put((actual_end_ns - start_ns, count))


def throughput_benchmark(
    capacity_bytes: int,
    payload_size: int,
    duration_sec: float,
    path: str,
) -> tuple[int, int, int, int]:
    """Benchmark producer-consumer throughput across processes.

    Args:
        capacity_bytes: Ring buffer capacity in bytes.
        payload_size: Size of each message payload.
        duration_sec: Duration of benchmark in seconds.
        path: Shared memory path.

    Returns:
        Tuple of (producer_ns, consumer_ns, producer_count, consumer_count).
    """
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

    # Start consumer first so it's ready
    cons_proc.start()
    prod_proc.start()

    prod_proc.join()
    cons_proc.join()

    prod_ns, prod_count = result_queue.get()
    cons_ns, cons_count = result_queue.get()

    return prod_ns, cons_ns, prod_count, cons_count


def print_latency_stats(name: str, latencies: np.ndarray, payload_size: int) -> None:
    """Print latency statistics."""
    count = len(latencies)
    mean = np.mean(latencies)
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    p999 = np.percentile(latencies, 99.9)
    ops_per_sec = 1e9 / mean if mean > 0 else 0

    print(f"\n{name} (payload={payload_size} bytes, n={count:,})")
    print(f"  Mean: {mean:,.0f} ns")
    print(f"  P50:  {p50:,.0f} ns")
    print(f"  P95:  {p95:,.0f} ns")
    print(f"  P99:  {p99:,.0f} ns")
    print(f"  P99.9: {p999:,.0f} ns")
    print(f"  Throughput: {ops_per_sec:,.0f} ops/sec")


def print_throughput_stats(
    payload_size: int,
    producer_ns: int,
    consumer_ns: int,
    prod_count: int,
    cons_count: int,
) -> None:
    """Print throughput benchmark results."""
    producer_time = producer_ns / 1e9
    consumer_time = consumer_ns / 1e9

    prod_ns_per_msg = producer_ns / prod_count if prod_count > 0 else 0
    cons_ns_per_msg = consumer_ns / cons_count if cons_count > 0 else 0

    prod_msg_per_sec = prod_count / producer_time if producer_time > 0 else 0
    cons_msg_per_sec = cons_count / consumer_time if consumer_time > 0 else 0
    prod_mb_per_sec = (
        (prod_count * payload_size) / (producer_time * 1024 * 1024)
        if producer_time > 0
        else 0
    )
    cons_mb_per_sec = (
        (cons_count * payload_size) / (consumer_time * 1024 * 1024)
        if consumer_time > 0
        else 0
    )

    print(f"\nThroughput - Payload: {payload_size} bytes")
    print(
        f"  Duration: {producer_time:.2f}s (producer), {consumer_time:.2f}s (consumer)"
    )
    print(f"  Messages: {prod_count:,} sent, {cons_count:,} received")
    print(
        f"  Producer: {prod_msg_per_sec:,.0f} msg/s, {prod_mb_per_sec:.2f} MB/s, {prod_ns_per_msg:,.0f} ns/msg"
    )
    print(
        f"  Consumer: {cons_msg_per_sec:,.0f} msg/s, {cons_mb_per_sec:.2f} MB/s, {cons_ns_per_msg:,.0f} ns/msg"
    )


def run_benchmarks() -> None:
    """Run all SHM ring buffer benchmarks."""
    PAYLOAD_SIZES = [32, 128, 512, 2048, 8192]
    CAPACITY_BYTES = 2**20  # 1MB
    LATENCY_ITERATIONS = 100_000
    THROUGHPUT_DURATION = 3.0

    base_path = f"/tmp/shm_bench_{os.getpid()}"

    print("=" * 80)
    print("SHM Ring Buffer Performance Benchmark")
    print("=" * 80)
    print(f"Capacity: {CAPACITY_BYTES:,} bytes ({CAPACITY_BYTES // 1024} KB)")
    print(f"Latency iterations: {LATENCY_ITERATIONS:,}")
    print(f"Throughput duration: {THROUGHPUT_DURATION}s per test")
    print(f"Payload sizes: {PAYLOAD_SIZES} bytes")

    # Latency benchmarks
    print("\n" + "-" * 40)
    print("LATENCY BENCHMARKS (single-threaded)")
    print("-" * 40)

    for payload_size in PAYLOAD_SIZES:
        path = f"{base_path}_lat_{payload_size}"

        # Clean up any existing file
        if Path(path).exists():
            Path(path).unlink()

        try:
            # Insert latency
            latencies = latency_benchmark_insert(
                CAPACITY_BYTES, payload_size, LATENCY_ITERATIONS, path
            )
            print_latency_stats("Insert", latencies, payload_size)
        finally:
            if Path(path).exists():
                Path(path).unlink()

        try:
            # Consume latency
            latencies = latency_benchmark_consume(
                CAPACITY_BYTES, payload_size, LATENCY_ITERATIONS, path
            )
            print_latency_stats("Consume", latencies, payload_size)
        finally:
            if Path(path).exists():
                Path(path).unlink()

    # Throughput benchmarks
    print("\n" + "-" * 40)
    print("THROUGHPUT BENCHMARKS (multi-process)")
    print("-" * 40)

    for payload_size in PAYLOAD_SIZES:
        path = f"{base_path}_tp_{payload_size}"

        # Clean up any existing file
        if Path(path).exists():
            Path(path).unlink()

        try:
            prod_ns, cons_ns, prod_count, cons_count = throughput_benchmark(
                CAPACITY_BYTES, payload_size, THROUGHPUT_DURATION, path
            )
            print_throughput_stats(
                payload_size, prod_ns, cons_ns, prod_count, cons_count
            )
        finally:
            if Path(path).exists():
                Path(path).unlink()

    print("\n" + "=" * 80)
    print("Benchmark complete")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmarks()
