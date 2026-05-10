"""Layer 3 — Integration tests for the advanced logging system.

Validates end-to-end multi-process logging via shared-memory IPC:
``WorkerLogger`` processes enqueue binary messages; ``MasterLogger``
dequques, formats, and delivers them to handlers. Covers many-to-one
workloads, high throughput, per-level filtering, large (1 MB) payloads,
mixed severities, and aggressive flush intervals.
"""

import multiprocessing
import os
import time
from queue import Empty, Queue
from pathlib import Path

import pytest

from mm_toolbox.logging.advanced.config import LoggerConfig
from mm_toolbox.logging.advanced.handlers.base import BaseLogHandler
from mm_toolbox.logging.advanced.handlers.file import FileLogHandler
from mm_toolbox.logging.advanced.master import MasterLogger
from mm_toolbox.logging.advanced.pylog import PyLogLevel
from mm_toolbox.logging.advanced.worker import WorkerLogger

pytestmark = pytest.mark.timeout(10, method="thread")


class MockHandler(BaseLogHandler):
    """Test-double handler that buffers received logs in a thread-safe queue."""

    def __init__(self):
        super().__init__()
        self.received_logs = Queue()

    def push(self, logs):
        """Enqueue every log from the batch so tests can drain them later."""
        for log in logs:
            self.received_logs.put(log)


def _drain_logs(queue: Queue, expected: int, timeout_s: float = 5.0) -> list:
    """Poll *queue* until *expected* items arrive or *timeout_s* elapses.

    Args:
        queue: Source queue.
        expected: Number of items to wait for.
        timeout_s: Maximum wait time in seconds.

    Returns:
        List of dequeued items.
    """
    received = []
    deadline = time.monotonic() + timeout_s
    while len(received) < expected and time.monotonic() < deadline:
        remaining = max(deadline - time.monotonic(), 0.0)
        try:
            received.append(queue.get(timeout=min(0.1, remaining)))
        except Empty:
            pass
    return received


def _wait_for_file_lines(path, expected: int, timeout_s: float = 5.0) -> list[str]:
    """Poll *path* until it contains at least *expected* lines.

    Args:
        path: Path to the log file.
        expected: Minimum number of lines required.
        timeout_s: Maximum wait time in seconds.

    Returns:
        List of lines read from the file.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.exists():
            lines = path.read_text().splitlines()
            if len(lines) >= expected:
                return lines
        time.sleep(0.05)
    return path.read_text().splitlines() if path.exists() else []


@pytest.fixture
def ipc_path(tmp_path: Path):
    """Return a factory that builds unique IPC paths per test process."""
    def _make(name: str) -> str:
        suffix = f"{name}_{os.getpid()}"
        path = tmp_path / suffix
        return str(path)

    return _make


def worker_process(
    path,
    name,
    num_logs,
    level=PyLogLevel.INFO,
    base_level=PyLogLevel.INFO,
):
    """Worker target that logs *num_logs* messages at *level* via a ``WorkerLogger``."""
    config = LoggerConfig(path=path, base_level=base_level)
    logger = WorkerLogger(config=config, name=name)
    for i in range(num_logs):
        log_func = {
            PyLogLevel.TRACE: logger.trace,
            PyLogLevel.DEBUG: logger.debug,
            PyLogLevel.INFO: logger.info,
            PyLogLevel.WARNING: logger.warning,
            PyLogLevel.ERROR: logger.error,
        }[level]
        log_func(msg_bytes=f"Log {i} from {name}".encode("utf-8"))
    logger.shutdown()


def worker_large_msg(path, name, msg_size):
    """Worker target that sends a single *msg_size*-byte payload."""
    config = LoggerConfig(path=path)
    logger = WorkerLogger(config=config, name=name)
    large_msg = b"x" * msg_size
    logger.info(msg_bytes=large_msg)
    logger.shutdown()


def worker_mixed_levels(path, name):
    """Worker target that emits one message at every severity level."""
    config = LoggerConfig(path=path, base_level=PyLogLevel.TRACE)
    logger = WorkerLogger(config=config, name=name)
    logger.trace(msg_bytes=b"Trace msg")
    logger.debug(msg_bytes=b"Debug msg")
    logger.info(msg_bytes=b"Info msg")
    logger.warning(msg_bytes=b"Warning msg")
    logger.error(msg_bytes=b"Error msg")
    logger.shutdown()


class TestIntegration:
    """Layer 3 — End-to-end multi-process integration tests."""

    @pytest.mark.parametrize(
        "num_workers, num_logs_per_worker", [(1, 10), (5, 10), (25, 10)]
    )
    def test_multiple_workers(
        self, num_workers, num_logs_per_worker, ipc_path, tmp_path
    ):
        """Given N workers each emitting M logs, the master receives N×M lines."""
        log_file = tmp_path / "integration_multi.txt"
        path = ipc_path("test_integration")
        config = LoggerConfig(path=path)
        file_handler = FileLogHandler(str(log_file), create=True)
        master = MasterLogger(config=config, log_handlers=[file_handler])
        file_handler.add_primary_config(config)

        processes = []
        for i in range(num_workers):
            p = multiprocessing.Process(
                target=worker_process,
                args=(path, f"Worker_{i}", num_logs_per_worker),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        lines = _wait_for_file_lines(log_file, num_workers * num_logs_per_worker)
        master.shutdown()

        assert len(lines) == num_workers * num_logs_per_worker

        worker_logs = {f"Worker_{i}": 0 for i in range(num_workers)}
        for line in lines:
            assert "INFO" in line
            for worker_name in worker_logs:
                if f" {worker_name} -" in line:
                    worker_logs[worker_name] += 1
                    break

        for count in worker_logs.values():
            assert count == num_logs_per_worker

    def test_high_throughput(self, ipc_path, tmp_path):
        """Given 10 workers × 1000 logs, all 10 000 lines are delivered within 8 s."""
        log_file = tmp_path / "test_high_throughput.txt"
        path = ipc_path("test_high_throughput")
        config = LoggerConfig(path=path, flush_interval_s=0.1)
        file_handler = FileLogHandler(str(log_file), create=True)
        master = MasterLogger(config=config, log_handlers=[file_handler])
        file_handler.add_primary_config(config)

        num_workers = 10
        num_logs_per_worker = 1000

        processes = []
        for i in range(num_workers):
            p = multiprocessing.Process(
                target=worker_process,
                args=(path, f"Worker_{i}", num_logs_per_worker),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        lines = _wait_for_file_lines(
            log_file,
            num_workers * num_logs_per_worker,
            timeout_s=8.0,
        )
        master.shutdown()

        assert len(lines) == num_workers * num_logs_per_worker

    @pytest.mark.parametrize(
        "level",
        [
            PyLogLevel.TRACE,
            PyLogLevel.DEBUG,
            PyLogLevel.INFO,
            PyLogLevel.WARNING,
            PyLogLevel.ERROR,
        ],
    )
    def test_different_levels(self, level, ipc_path, tmp_path):
        """Given a worker emitting at a single level, only that level appears in the output."""
        log_file = tmp_path / f"test_levels_{level.name}.txt"
        path = ipc_path(f"test_levels_{level}")
        config = LoggerConfig(
            path=path,
            base_level=PyLogLevel.TRACE,
            str_format="%(levelname)s: %(message)s",
        )
        file_handler = FileLogHandler(str(log_file), create=True)
        master = MasterLogger(config=config, log_handlers=[file_handler])
        file_handler.add_primary_config(config)

        p = multiprocessing.Process(
            target=worker_process,
            args=(path, "Worker", 5, level, PyLogLevel.TRACE),
        )
        p.start()
        p.join()

        lines = _wait_for_file_lines(log_file, 5)
        master.shutdown()

        assert len(lines) == 5
        for line in lines:
            assert line.startswith(f"{level.name}:")

    def test_large_messages(self, ipc_path):
        """Given a 1 MB payload, the master receives it intact without truncation."""
        path = ipc_path("test_large")
        config = LoggerConfig(path=path)
        mock_handler = MockHandler()
        master = MasterLogger(config=config, log_handlers=[mock_handler])
        mock_handler.add_primary_config(config)

        msg_size = 1024 * 1024
        p = multiprocessing.Process(
            target=worker_large_msg, args=(path, "Worker", msg_size)
        )
        p.start()
        p.join()

        received = _drain_logs(mock_handler.received_logs, 1)
        master.shutdown()

        assert len(received) == 1
        assert len(received[0][3]) == msg_size

    def test_mixed_levels(self, ipc_path):
        """Given a worker emitting all five levels, the master receives all five."""
        path = ipc_path("test_mixed")
        config = LoggerConfig(path=path, base_level=PyLogLevel.TRACE)
        mock_handler = MockHandler()
        master = MasterLogger(config=config, log_handlers=[mock_handler])
        mock_handler.add_primary_config(config)

        p = multiprocessing.Process(target=worker_mixed_levels, args=(path, "Worker"))
        p.start()
        p.join()

        received = _drain_logs(mock_handler.received_logs, 5)
        master.shutdown()

        received_levels = set()
        for _, _, level, _ in received:
            received_levels.add(level)

        assert len(received_levels) == 5

    def test_with_file_handler(self, tmp_path, ipc_path):
        """Given a file handler, logs are written line-by-line in arrival order."""
        log_file = tmp_path / "test.txt"
        path = ipc_path("test_file")
        config = LoggerConfig(path=path, str_format="%(levelname)s: %(message)s")
        file_handler = FileLogHandler(str(log_file), create=True)
        master = MasterLogger(config=config, log_handlers=[file_handler])
        file_handler.add_primary_config(config)

        p = multiprocessing.Process(target=worker_process, args=(path, "Worker", 3))
        p.start()
        p.join()

        lines = _wait_for_file_lines(log_file, 3)
        master.shutdown()

        assert len(lines) == 3
        for i, line in enumerate(lines):
            assert line.strip() == f"INFO: Log {i} from Worker"

    def test_short_flush_many_logs(self, ipc_path, tmp_path):
        """Given a 0.01-second flush interval, 1000 logs are delivered promptly."""
        log_file = tmp_path / "test_flush.txt"
        path = ipc_path("test_flush")
        config = LoggerConfig(path=path, flush_interval_s=0.01)
        file_handler = FileLogHandler(str(log_file), create=True)
        master = MasterLogger(config=config, log_handlers=[file_handler])
        file_handler.add_primary_config(config)

        num_logs = 1000
        p = multiprocessing.Process(
            target=worker_process, args=(path, "Worker", num_logs)
        )
        p.start()
        p.join()

        lines = _wait_for_file_lines(log_file, num_logs, timeout_s=8.0)
        master.shutdown()

        assert len(lines) == num_logs
