"""Stress tests for advanced logging handlers.

These tests exercise the handler subsystem under high load to verify
throughput, concurrency safety, and memory stability.
"""

from __future__ import annotations

import gc
import resource
import threading

import pytest

from mm_toolbox.logging.advanced.config import LoggerConfig
from mm_toolbox.logging.advanced.handlers.base import BaseLogHandler
from mm_toolbox.logging.advanced.pylog import PyLog, PyLogLevel

pytestmark = pytest.mark.slow


class CountingHandler(BaseLogHandler):
    """A simple handler that counts received logs for stress testing."""

    def __init__(self):
        """Initialize the counting handler."""
        super().__init__()
        self.count = 0
        self.lock = threading.Lock()

    def push(self, logs: list[PyLog]) -> None:
        """Count the number of logs received.

        Args:
            logs: Batch of log entries.
        """
        with self.lock:
            self.count += len(logs)


def _worker_push_logs(
    handler: CountingHandler, num_logs: int, batch_size: int = 1000
) -> None:
    """Push logs to a handler in batches from a worker thread.

    Args:
        handler: Target handler.
        num_logs: Total number of logs to push.
        batch_size: Number of logs per push call.
    """
    config = LoggerConfig(str_format="%(message)s")
    handler.add_primary_config(config)
    for i in range(0, num_logs, batch_size):
        batch = [
            PyLog(j, b"name", PyLogLevel.INFO, b"msg")
            for j in range(i, min(i + batch_size, num_logs))
        ]
        handler.push(batch)


class TestStressHandlers:
    """Stress and load tests for logging handlers."""

    def test_high_throughput_single_worker(self):
        """One worker pushing 100K logs; all must be received."""
        handler = CountingHandler()
        num_logs = 100_000

        worker = threading.Thread(target=_worker_push_logs, args=(handler, num_logs))
        worker.start()
        worker.join(timeout=30.0)

        assert worker.is_alive() is False
        assert handler.count == num_logs
        handler.close()

    def test_high_throughput_multiple_workers(self):
        """Ten workers each pushing 10K logs; all must be received."""
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
        """One worker sends 1M logs as fast as possible."""
        handler = CountingHandler()
        num_logs = 1_000_000

        logs = [PyLog(i, b"name", PyLogLevel.INFO, b"msg") for i in range(num_logs)]
        handler.push(logs)

        assert handler.count == num_logs
        handler.close()

    def test_memory_stability(self):
        """Push 10K logs ten times and verify modest memory growth."""
        handler = CountingHandler()
        num_logs = 10_000

        # Warmup batch
        logs = [PyLog(i, b"name", PyLogLevel.INFO, b"msg") for i in range(num_logs)]
        handler.push(logs)
        gc.collect()
        mem_after_warmup = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        # Main batches
        for _ in range(9):
            logs = [PyLog(i, b"name", PyLogLevel.INFO, b"msg") for i in range(num_logs)]
            handler.push(logs)

        gc.collect()
        mem_after_main = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        if mem_after_warmup > 0:
            growth = (mem_after_main - mem_after_warmup) / mem_after_warmup
            assert growth < 5.0

        handler.close()
