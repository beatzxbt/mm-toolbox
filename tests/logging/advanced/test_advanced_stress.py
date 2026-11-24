"""Layer 3 — Stress tests for advanced logging handlers.

Exercises the handler subsystem under high load to verify throughput,
concurrency safety, and memory stability. Load tests are marked ``slow``; exact transport-accounting regressions also
run in the ordinary suite.
"""

from __future__ import annotations

import gc
import resource
import threading

import pytest

import mm_toolbox.logging.advanced.worker as worker_module
from mm_toolbox.logging.advanced import WorkerLogger
from mm_toolbox.logging.advanced.config import LoggerConfig
from mm_toolbox.logging.advanced.handlers.base import BaseLogHandler
from mm_toolbox.logging.advanced.pylog import PyLog, PyLogLevel


class CountingHandler(BaseLogHandler):
    """Test-double handler that atomically counts received logs for stress testing."""

    def __init__(self):
        super().__init__()
        self.count = 0
        self.lock = threading.Lock()

    def push(self, logs: list[PyLog]) -> None:
        """Atomically increment the counter by the batch size.

        Args:
            logs: Batch of log entries.
        """
        with self.lock:
            self.count += len(logs)


def _worker_push_logs(
    handler: CountingHandler, num_logs: int, batch_size: int = 1000
) -> None:
    """Push logs to *handler* in batches from a worker thread.

    Args:
        handler: Target handler.
        num_logs: Total number of logs to push.
        batch_size: Number of logs per ``push()`` call.
    """
    config = LoggerConfig(str_format="%(message)s")
    handler.add_primary_config(config)
    for i in range(0, num_logs, batch_size):
        batch = [
            PyLog(j, b"name", PyLogLevel.INFO, b"msg")
            for j in range(i, min(i + batch_size, num_logs))
        ]
        handler.push(batch)


@pytest.mark.slow
class TestStressHandlers:
    """Layer 3 — Stress and load tests for logging handlers."""

    def test_high_throughput_single_worker(self):
        """Given one worker pushing 50 000 logs, every single log is received."""
        handler = CountingHandler()
        num_logs = 50_000

        worker = threading.Thread(target=_worker_push_logs, args=(handler, num_logs))
        worker.start()
        worker.join(timeout=60.0)

        assert worker.is_alive() is False
        assert handler.count == num_logs
        handler.close()

    def test_high_throughput_multiple_workers(self):
        """Given ten workers each pushing 10 000 logs, the total received equals 100 000.

        This catches race conditions in handler buffer or accounting logic.
        """
        handler = CountingHandler()
        num_workers = 10
        logs_per_worker = 10_000
        workers: list[threading.Thread] = []

        for _ in range(num_workers):
            t = threading.Thread(
                target=_worker_push_logs, args=(handler, logs_per_worker)
            )
            workers.append(t)
            t.start()

        for t in workers:
            t.join(timeout=30.0)

        for t in workers:
            assert t.is_alive() is False

        assert handler.count == num_workers * logs_per_worker
        handler.close()

    def test_burst_load(self):
        """Given a single 1 000 000-log batch, it is processed without error or timeout."""
        handler = CountingHandler()
        num_logs = 1_000_000

        logs = [PyLog(i, b"name", PyLogLevel.INFO, b"msg") for i in range(num_logs)]
        handler.push(logs)

        assert handler.count == num_logs
        handler.close()

    def test_memory_stability(self):
        """Given 10 × 10 000-log batches, RSS growth stays below 5×.

        A large growth would indicate a memory leak in the handler's internal
        buffers or future-tracking lists.
        """
        handler = CountingHandler()
        num_logs = 10_000

        logs = [PyLog(i, b"name", PyLogLevel.INFO, b"msg") for i in range(num_logs)]
        handler.push(logs)
        gc.collect()
        mem_after_warmup = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        for _ in range(9):
            logs = [PyLog(i, b"name", PyLogLevel.INFO, b"msg") for i in range(num_logs)]
            handler.push(logs)

        gc.collect()
        mem_after_main = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        if mem_after_warmup > 0:
            growth = (mem_after_main - mem_after_warmup) / mem_after_warmup
            assert growth < 5.0

        handler.close()


class FakeTransport:
    num_rings = 2
    last_insert_dropped = 0
    failed_inserts = 0

    def __init__(self, mode):
        self.mode = mode
        self.closed = False

    def insert(self, payload):
        if self.mode == "error":
            raise OSError("injected insert error")
        if self.mode == "rejected":
            self.failed_inserts += 1
            return False
        self.last_insert_dropped = 3 if self.mode == "overwrite" else 0
        return True

    def close(self):
        self.closed = True
        if self.mode == "close_error":
            raise OSError("injected close error")


@pytest.mark.parametrize(
    "mode,rejected,errors",
    [("rejected", 1, 0), ("error", 1, 1), ("close_error", 0, 1), ("oversized", 1, 0)],
)
def test_worker_rejections_and_errors_remain_visible_after_shutdown(
    tmp_path, monkeypatch, mode, rejected, errors
):
    transport = FakeTransport(mode)
    monkeypatch.setattr(worker_module, "ShmMpscProducer", lambda **kwargs: transport)
    config = LoggerConfig(
        path=str(tmp_path / "worker.shm"),
        flush_interval_s=60,
        max_batch_messages=1,
        max_batch_bytes=16 if mode == "oversized" else 1024,
    )
    worker = WorkerLogger(config=config, name="fault-worker")
    try:
        worker.info(b"x" * 16)
    finally:
        worker.shutdown()
    expected = {
        "rejected_messages": rejected,
        "transport_errors": errors,
        "overwritten_batches": 0,
    }
    assert worker.get_transport_stats() == expected
    worker.shutdown()
    assert worker.get_transport_stats() == expected
    assert transport.closed


def test_sampled_overwrite_diagnostics_preserve_exact_batch_counts(
    tmp_path, monkeypatch, capsys
):
    transport = FakeTransport("overwrite")
    monkeypatch.setattr(worker_module, "ShmMpscProducer", lambda **kwargs: transport)
    config = LoggerConfig(
        path=str(tmp_path / "worker.shm"), flush_interval_s=60, max_batch_messages=1
    )
    worker = WorkerLogger(config=config, name="saturated-worker")
    try:
        for _ in range(16):
            worker.info(b"message")
    finally:
        worker.shutdown()
    assert worker.get_transport_stats() == {
        "rejected_messages": 0,
        "transport_errors": 0,
        "overwritten_batches": 48,
    }
    warnings = [
        line for line in capsys.readouterr().err.splitlines() if "SHM overwrite" in line
    ]
    assert len(warnings) == 5
    assert "dropped 48 older log batches" in warnings[-1]
