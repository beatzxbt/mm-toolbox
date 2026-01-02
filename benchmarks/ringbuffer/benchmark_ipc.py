"""Performance benchmark for IPC ring buffer across processes."""

import asyncio
import multiprocessing
import os
import time
from multiprocessing import Queue
from pathlib import Path

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
    barrier: "multiprocessing.Barrier",
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
    result_queue.put((actual_end_ns - start_ns, count))


def _consumer_sync_single(
    path: str,
    duration_sec: float,
    result_queue: Queue,
    barrier: "multiprocessing.Barrier",
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
    result_queue.put((actual_end_ns - start_ns, count))


def _producer_sync_packed(
    path: str,
    payload_size: int,
    duration_sec: float,
    batch_size: int,
    result_queue: Queue,
    barrier: "multiprocessing.Barrier",
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
    result_queue.put((actual_end_ns - start_ns, count))


def _consumer_sync_packed(
    path: str,
    duration_sec: float,
    batch_size: int,
    result_queue: Queue,
    barrier: "multiprocessing.Barrier",
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
    result_queue.put((actual_end_ns - start_ns, count))


async def _producer_async_single(
    path: str,
    payload_size: int,
    duration_sec: float,
    result_queue: Queue,
    barrier: "multiprocessing.Barrier",
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

    barrier.wait()
    await asyncio.sleep(0.1)

    start_ns = time.perf_counter_ns()
    end_time_ns = start_ns + int(duration_sec * 1e9)
    count = 0

    while time.perf_counter_ns() < end_time_ns:
        producer.insert(payload, copy=False)
        count += 1

    actual_end_ns = time.perf_counter_ns()
    producer.stop()
    result_queue.put((actual_end_ns - start_ns, count))


async def _consumer_async_single(
    path: str,
    duration_sec: float,
    result_queue: Queue,
    barrier: "multiprocessing.Barrier",
) -> None:
    """Consumer process for async single benchmark."""
    config = IPCRingBufferConfig(
        path=path,
        backlog=2**16,
        num_producers=1,
        num_consumers=1,
    )
    consumer = IPCRingBufferConsumer(config)

    barrier.wait()
    await asyncio.sleep(0.1)

    start_ns = time.perf_counter_ns()
    end_time_ns = start_ns + int(duration_sec * 1e9)
    count = 0

    while True:
        remaining_ns = end_time_ns - time.perf_counter_ns()
        if remaining_ns <= 0:
            break

        timeout = min(remaining_ns / 1e9, 0.1)
        try:
            await asyncio.wait_for(consumer.aconsume(), timeout=timeout)
            count += 1
        except asyncio.TimeoutError:
            if time.perf_counter_ns() >= end_time_ns:
                break

    actual_end_ns = time.perf_counter_ns()
    consumer.stop()
    result_queue.put((actual_end_ns - start_ns, count))


async def _producer_async_packed(
    path: str,
    payload_size: int,
    duration_sec: float,
    batch_size: int,
    result_queue: Queue,
    barrier: "multiprocessing.Barrier",
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

    barrier.wait()
    await asyncio.sleep(0.1)

    start_ns = time.perf_counter_ns()
    end_time_ns = start_ns + int(duration_sec * 1e9)
    count = 0

    while time.perf_counter_ns() < end_time_ns:
        producer.insert_packed(batches, copy=False)
        count += batch_size

    actual_end_ns = time.perf_counter_ns()
    producer.stop()
    result_queue.put((actual_end_ns - start_ns, count))


async def _consumer_async_packed(
    path: str,
    duration_sec: float,
    batch_size: int,
    result_queue: Queue,
    barrier: "multiprocessing.Barrier",
) -> None:
    """Consumer process for async packed benchmark."""
    config = IPCRingBufferConfig(
        path=path,
        backlog=2**16,
        num_producers=1,
        num_consumers=1,
    )
    consumer = IPCRingBufferConsumer(config)

    barrier.wait()
    await asyncio.sleep(0.1)

    start_ns = time.perf_counter_ns()
    end_time_ns = start_ns + int(duration_sec * 1e9)
    count = 0

    while True:
        remaining_ns = end_time_ns - time.perf_counter_ns()
        if remaining_ns <= 0:
            break

        timeout = min(remaining_ns / 1e9, 0.1)
        try:
            items = await asyncio.wait_for(consumer.aconsume_packed(), timeout=timeout)
            count += len(items)
        except asyncio.TimeoutError:
            if time.perf_counter_ns() >= end_time_ns:
                break

    actual_end_ns = time.perf_counter_ns()
    consumer.stop()
    result_queue.put((actual_end_ns - start_ns, count))


def _run_producer_async_single(
    path: str,
    payload_size: int,
    duration_sec: float,
    result_queue: Queue,
    barrier: "multiprocessing.Barrier",
) -> None:
    """Wrapper to run async producer in process."""
    asyncio.run(
        _producer_async_single(path, payload_size, duration_sec, result_queue, barrier)
    )


def _run_consumer_async_single(
    path: str,
    duration_sec: float,
    result_queue: Queue,
    barrier: "multiprocessing.Barrier",
) -> None:
    """Wrapper to run async consumer in process."""
    asyncio.run(_consumer_async_single(path, duration_sec, result_queue, barrier))


def _run_producer_async_packed(
    path: str,
    payload_size: int,
    duration_sec: float,
    batch_size: int,
    result_queue: Queue,
    barrier: "multiprocessing.Barrier",
) -> None:
    """Wrapper to run async producer in process."""
    asyncio.run(
        _producer_async_packed(
            path, payload_size, duration_sec, batch_size, result_queue, barrier
        )
    )


def _run_consumer_async_packed(
    path: str,
    duration_sec: float,
    batch_size: int,
    result_queue: Queue,
    barrier: "multiprocessing.Barrier",
) -> None:
    """Wrapper to run async consumer in process."""
    asyncio.run(
        _consumer_async_packed(path, duration_sec, batch_size, result_queue, barrier)
    )


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
        target=_consumer_sync_single, args=(path, duration_sec, result_queue, barrier)
    )

    cons_proc.start()
    prod_proc.start()

    cons_proc.join()
    prod_proc.join()

    prod_ns, prod_count = result_queue.get()
    cons_ns, cons_count = result_queue.get()

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
        args=(path, duration_sec, batch_size, result_queue, barrier),
    )

    cons_proc.start()
    prod_proc.start()

    cons_proc.join()
    prod_proc.join()

    prod_ns, prod_count = result_queue.get()
    cons_ns, cons_count = result_queue.get()

    return prod_ns, cons_ns, prod_count, cons_count


async def benchmark_async_single(
    payload_size: int, duration_sec: float, path: str
) -> tuple[int, int, int, int]:
    """Benchmark asynchronous single message throughput across processes."""
    prod_queue: Queue = Queue()
    cons_queue: Queue = Queue()
    barrier = multiprocessing.Barrier(2)

    prod_proc = multiprocessing.Process(
        target=_run_producer_async_single,
        args=(path, payload_size, duration_sec, prod_queue, barrier),
    )
    cons_proc = multiprocessing.Process(
        target=_run_consumer_async_single,
        args=(path, duration_sec, cons_queue, barrier),
    )

    cons_proc.start()
    prod_proc.start()

    cons_proc.join()
    prod_proc.join()

    prod_ns, prod_count = prod_queue.get()
    cons_ns, cons_count = cons_queue.get()

    return prod_ns, cons_ns, prod_count, cons_count


async def benchmark_async_packed(
    payload_size: int, duration_sec: float, batch_size: int, path: str
) -> tuple[int, int, int, int]:
    """Benchmark asynchronous packed batch throughput across processes."""
    prod_queue: Queue = Queue()
    cons_queue: Queue = Queue()
    barrier = multiprocessing.Barrier(2)

    prod_proc = multiprocessing.Process(
        target=_run_producer_async_packed,
        args=(path, payload_size, duration_sec, batch_size, prod_queue, barrier),
    )
    cons_proc = multiprocessing.Process(
        target=_run_consumer_async_packed,
        args=(path, duration_sec, batch_size, cons_queue, barrier),
    )

    cons_proc.start()
    prod_proc.start()

    cons_proc.join()
    prod_proc.join()

    prod_ns, prod_count = prod_queue.get()
    cons_ns, cons_count = cons_queue.get()

    return prod_ns, cons_ns, prod_count, cons_count


def print_results(
    payload_size: int,
    producer_ns: int,
    consumer_ns: int,
    prod_count: int,
    cons_count: int,
    mode: str,
) -> None:
    """Print benchmark results."""
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

    print(f"\n{mode} - Payload: {payload_size} bytes")
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


async def run_benchmarks() -> None:
    """Run all benchmarks."""
    PAYLOAD_SIZES = [64, 256, 1024, 4096, 16384, 65536, 262144]
    DURATION_SEC = 5.0
    BATCH_SIZE = 100

    base_path = f"ipc:///tmp/ringbuffer_bench_{os.getpid()}"
    print("=" * 80)
    print("IPC Ring Buffer Performance Benchmark (Multi-Process)")
    print("=" * 80)
    print(f"Duration per test: {DURATION_SEC}s")
    print(f"Batch size (packed): {BATCH_SIZE}")
    print(f"Payload sizes: {PAYLOAD_SIZES} bytes")

    for payload_size in PAYLOAD_SIZES:
        path = f"{base_path}_{payload_size}"

        socket_path = path.replace("ipc://", "")
        if Path(socket_path).exists():
            Path(socket_path).unlink()

        try:
            prod_ns, cons_ns, prod_count, cons_count = benchmark_sync_single(
                payload_size, DURATION_SEC, path
            )
            print_results(
                payload_size, prod_ns, cons_ns, prod_count, cons_count, "Sync Single"
            )

            socket_path = path.replace("ipc://", "")
            if Path(socket_path).exists():
                Path(socket_path).unlink()

            prod_ns, cons_ns, prod_count, cons_count = benchmark_sync_packed(
                payload_size, DURATION_SEC, BATCH_SIZE, path
            )
            print_results(
                payload_size,
                prod_ns,
                cons_ns,
                prod_count,
                cons_count,
                f"Sync Packed (batch={BATCH_SIZE})",
            )

            socket_path = path.replace("ipc://", "")
            if Path(socket_path).exists():
                Path(socket_path).unlink()

            prod_ns, cons_ns, prod_count, cons_count = await benchmark_async_single(
                payload_size, DURATION_SEC, path
            )
            print_results(
                payload_size, prod_ns, cons_ns, prod_count, cons_count, "Async Single"
            )

            socket_path = path.replace("ipc://", "")
            if Path(socket_path).exists():
                Path(socket_path).unlink()

            prod_ns, cons_ns, prod_count, cons_count = await benchmark_async_packed(
                payload_size, DURATION_SEC, BATCH_SIZE, path
            )
            print_results(
                payload_size,
                prod_ns,
                cons_ns,
                prod_count,
                cons_count,
                f"Async Packed (batch={BATCH_SIZE})",
            )
        finally:
            socket_path = path.replace("ipc://", "")
            if Path(socket_path).exists():
                Path(socket_path).unlink()

    print("\n" + "=" * 80)


if __name__ == "__main__":
    asyncio.run(run_benchmarks())
