import multiprocessing
import os
import time
from queue import Empty, Queue
from pathlib import Path
import hashlib

import pytest
import zmq

from mm_toolbox.logging.advanced.config import LoggerConfig
from mm_toolbox.logging.advanced.handlers.base import BaseLogHandler
from mm_toolbox.logging.advanced.handlers.file import FileLogHandler
from mm_toolbox.logging.advanced.master import MasterLogger
from mm_toolbox.logging.advanced.pylog import PyLogLevel
from mm_toolbox.logging.advanced.worker import WorkerLogger

pytestmark = pytest.mark.timeout(10, method="thread")


class MockHandler(BaseLogHandler):
    def __init__(self):
        super().__init__()
        self.received_logs = Queue()

    def push(self, logs):
        for log in logs:
            self.received_logs.put(log)


def _drain_logs(queue: Queue, expected: int, timeout_s: float = 5.0) -> list:
    """Drain logs until expected count or timeout."""
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
    """Wait for a file to contain at least expected lines."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.exists():
            lines = path.read_text().splitlines()
            if len(lines) >= expected:
                return lines
        time.sleep(0.05)
    return path.read_text().splitlines() if path.exists() else []


def _ipc_available(path: Path) -> bool:
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PULL)
    addr = f"ipc://{path}"
    try:
        sock.bind(addr)
        sock.unbind(addr)
        return True
    except zmq.ZMQError:
        return False
    finally:
        sock.close(0)
        if path.exists():
            path.unlink()


@pytest.fixture(scope="session", autouse=True)
def _require_ipc_support() -> None:
    base_dir = Path.cwd() / ".ipc"
    base_dir.mkdir(exist_ok=True)
    probe_path = base_dir / "ipc_probe"
    if not _ipc_available(probe_path):
        pytest.skip("IPC transport is not available in this environment")


@pytest.fixture
def ipc_path():
    def _make(name: str) -> str:
        base_dir = Path.cwd() / ".ipc"
        base_dir.mkdir(exist_ok=True)
        suffix = f"{name}_{os.getpid()}"
        path = base_dir / suffix

        max_len = getattr(zmq, "IPC_PATH_MAX_LEN", 103)
        if len(str(path)) > max_len:
            digest = hashlib.sha1(suffix.encode("utf-8")).hexdigest()[:12]
            path = base_dir / digest

        return f"ipc://{path}"

    return _make


def worker_process(path, name, num_logs, level=PyLogLevel.INFO):
    config = LoggerConfig(path=path)
    logger = WorkerLogger(config=config, name=name)
    for i in range(num_logs):
        log_func = {
            PyLogLevel.TRACE: logger.trace,
            PyLogLevel.DEBUG: logger.debug,
            PyLogLevel.INFO: logger.info,
            PyLogLevel.WARNING: logger.warning,
            PyLogLevel.ERROR: logger.error,
        }[level]
        log_func(f"Log {i} from {name}")
    logger.shutdown()


def worker_large_msg(path, name, msg_size):
    config = LoggerConfig(path=path)
    logger = WorkerLogger(config=config, name=name)
    large_msg = b"x" * msg_size
    logger.info(msg_bytes=large_msg)
    logger.shutdown()


def worker_mixed_levels(path, name):
    config = LoggerConfig(path=path)
    logger = WorkerLogger(config=config, name=name)
    logger.trace("Trace msg")
    logger.debug("Debug msg")
    logger.info("Info msg")
    logger.warning("Warning msg")
    logger.error("Error msg")
    logger.shutdown()


class TestIntegration:
    @pytest.mark.parametrize(
        "num_workers, num_logs_per_worker", [(1, 10), (5, 10), (25, 10)]
    )
    def test_multiple_workers(self, num_workers, num_logs_per_worker, ipc_path):
        path = ipc_path("test_integration")
        config = LoggerConfig(path=path)
        mock_handler = MockHandler()
        master = MasterLogger(config=config, log_handlers=[mock_handler])
        mock_handler.add_primary_config(config)

        processes = []
        for i in range(num_workers):
            p = multiprocessing.Process(
                target=worker_process,
                args=(path, f"Worker_{i}", num_logs_per_worker),
            )
            p.start()
            processes.append(p)

        # Wait for all workers to finish
        for p in processes:
            p.join()

        received = _drain_logs(
            mock_handler.received_logs,
            num_workers * num_logs_per_worker,
        )
        master.shutdown()

        assert len(received) == num_workers * num_logs_per_worker

        # Check contents
        worker_logs = {f"Worker_{i}".encode(): 0 for i in range(num_workers)}
        for log in received:
            assert log.level == PyLogLevel.INFO
            worker_logs[log.name] += 1

        for count in worker_logs.values():
            assert count == num_logs_per_worker

    def test_high_throughput(self, ipc_path):
        path = ipc_path("test_high_throughput")
        config = LoggerConfig(path=path, flush_interval_s=0.1)
        mock_handler = MockHandler()
        master = MasterLogger(config=config, log_handlers=[mock_handler])
        mock_handler.add_primary_config(config)

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

        received = _drain_logs(
            mock_handler.received_logs,
            num_workers * num_logs_per_worker,
            timeout_s=8.0,
        )
        master.shutdown()

        assert len(received) == num_workers * num_logs_per_worker

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
    def test_different_levels(self, level, ipc_path):
        path = ipc_path(f"test_levels_{level}")
        config = LoggerConfig(
            path=path, base_level=PyLogLevel.TRACE
        )  # Set low to capture all
        mock_handler = MockHandler()
        master = MasterLogger(config=config, log_handlers=[mock_handler])
        mock_handler.add_primary_config(config)

        p = multiprocessing.Process(
            target=worker_process, args=(path, "Worker", 5, level)
        )
        p.start()
        p.join()

        received = _drain_logs(mock_handler.received_logs, 5)
        master.shutdown()

        assert len(received) == 5
        for _, _, recv_level, _ in received:
            assert recv_level == level

    def test_large_messages(self, ipc_path):
        path = ipc_path("test_large")
        config = LoggerConfig(path=path)
        mock_handler = MockHandler()
        master = MasterLogger(config=config, log_handlers=[mock_handler])
        mock_handler.add_primary_config(config)

        msg_size = 1024 * 1024  # 1MB
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

        assert len(received_levels) == 5  # All levels

    def test_with_file_handler(self, tmp_path, ipc_path):
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

    def test_short_flush_many_logs(self, ipc_path):
        path = ipc_path("test_flush")
        config = LoggerConfig(path=path, flush_interval_s=0.01)  # Very short
        mock_handler = MockHandler()
        master = MasterLogger(config=config, log_handlers=[mock_handler])
        mock_handler.add_primary_config(config)

        num_logs = 1000
        p = multiprocessing.Process(
            target=worker_process, args=(path, "Worker", num_logs)
        )
        p.start()
        p.join()

        received = _drain_logs(mock_handler.received_logs, num_logs, timeout_s=8.0)
        master.shutdown()

        assert len(received) == num_logs
