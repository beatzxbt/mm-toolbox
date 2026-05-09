import multiprocessing
import os
import threading
import time
from pathlib import Path
from queue import Empty, Queue

import pytest

from mm_toolbox.logging.advanced.config import LoggerConfig
from mm_toolbox.logging.advanced.handlers.base import BaseLogHandler
from mm_toolbox.logging.advanced.handlers.file import FileLogHandler
from mm_toolbox.logging.advanced.master import MasterLogger
from mm_toolbox.logging.advanced.pylog import PyLogLevel
from mm_toolbox.logging.advanced.worker import WorkerLogger
from mm_toolbox.ringbuffer.shm.mpsc import ShmMpscProducer


def _create_shm_ring(
    path: str, capacity_bytes: int = 65536, num_rings: int = 1
) -> None:
    """Create a shared-memory ring at ``path`` for consumers/producers to attach."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    producer = ShmMpscProducer(
        path=path,
        capacity_bytes=capacity_bytes,
        num_rings=num_rings,
        create=True,
        unlink_on_close=False,
    )
    producer.close()


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


@pytest.fixture
def shm_path(tmp_path: Path):
    def _make(name: str) -> str:
        path = tmp_path / f"{name}_{os.getpid()}"
        return str(path)

    return _make


def worker_process(
    path, name, num_logs, level=PyLogLevel.INFO, base_level=PyLogLevel.INFO
):
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
        log_func(f"Log {i} from {name}".encode())
    logger.shutdown()


def worker_large_msg(path, name, msg_size):
    config = LoggerConfig(path=path)
    logger = WorkerLogger(config=config, name=name)
    large_msg = b"x" * msg_size
    logger.info(msg_bytes=large_msg)
    logger.shutdown()


def worker_mixed_levels(path, name):
    config = LoggerConfig(path=path, base_level=PyLogLevel.TRACE)
    logger = WorkerLogger(config=config, name=name)
    logger.trace(b"Trace msg")
    logger.debug(b"Debug msg")
    logger.info(b"Info msg")
    logger.warning(b"Warning msg")
    logger.error(b"Error msg")
    logger.shutdown()


def child_task(path):
    config = LoggerConfig(path=path)
    child_worker = WorkerLogger(config=config, name="Child")
    child_worker.info(b"child log")
    child_worker.shutdown()


pytestmark = pytest.mark.timeout(10, method="thread")


class TestIntegration:
    @pytest.fixture(autouse=True)
    def _clear_worker_singleton(self):
        from mm_toolbox.logging.advanced.worker import _worker_logger_local

        if hasattr(_worker_logger_local, "logger"):
            delattr(_worker_logger_local, "logger")
        yield
        if hasattr(_worker_logger_local, "logger"):
            delattr(_worker_logger_local, "logger")

    @pytest.mark.parametrize(
        "num_workers, num_logs_per_worker", [(1, 10), (5, 10), (25, 10)]
    )
    def test_multiple_workers(self, num_workers, num_logs_per_worker, shm_path):
        path = shm_path("test_integration")
        config = LoggerConfig(path=path)
        _create_shm_ring(
            path,
            capacity_bytes=config.shm_capacity_bytes,
            num_rings=max(num_workers, 1),
        )
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

        for p in processes:
            p.join()

        received = _drain_logs(
            mock_handler.received_logs,
            num_workers * num_logs_per_worker,
        )
        master.shutdown()

        assert len(received) == num_workers * num_logs_per_worker

        worker_logs = {f"Worker_{i}".encode(): 0 for i in range(num_workers)}
        for log in received:
            assert log.level == PyLogLevel.INFO
            worker_logs[log.name] += 1

        for count in worker_logs.values():
            assert count == num_logs_per_worker

    def test_high_throughput(self, shm_path):
        path = shm_path("test_high_throughput")
        num_workers = 10
        config = LoggerConfig(path=path, flush_interval_s=0.1)
        _create_shm_ring(
            path, capacity_bytes=config.shm_capacity_bytes, num_rings=num_workers
        )
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
    def test_different_levels(self, level, shm_path):
        path = shm_path(f"test_levels_{level}")
        _create_shm_ring(path)
        config = LoggerConfig(path=path, base_level=PyLogLevel.TRACE)
        mock_handler = MockHandler()
        master = MasterLogger(config=config, log_handlers=[mock_handler])
        mock_handler.add_primary_config(config)

        p = multiprocessing.Process(
            target=worker_process,
            args=(path, "Worker", 5, level, PyLogLevel.TRACE),
        )
        p.start()
        p.join()

        received = _drain_logs(mock_handler.received_logs, 5)
        master.shutdown()

        assert len(received) == 5
        for _, _, recv_level, _ in received:
            assert recv_level == level

    def test_large_messages(self, shm_path):
        path = shm_path("test_large")
        config = LoggerConfig(path=path)
        _create_shm_ring(path, capacity_bytes=config.shm_capacity_bytes)
        mock_handler = MockHandler()
        master = MasterLogger(config=config, log_handlers=[mock_handler])
        mock_handler.add_primary_config(config)

        msg_size = 512 * 1024  # 512KB to fit within batch capacity
        p = multiprocessing.Process(
            target=worker_large_msg, args=(path, "Worker", msg_size)
        )
        p.start()
        p.join()

        received = _drain_logs(mock_handler.received_logs, 1)
        master.shutdown()

        assert len(received) == 1
        assert len(received[0][3]) == msg_size

    def test_mixed_levels(self, shm_path):
        path = shm_path("test_mixed")
        _create_shm_ring(path)
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

    def test_with_file_handler(self, tmp_path, shm_path):
        log_file = tmp_path / "test.txt"
        path = shm_path("test_file")
        _create_shm_ring(path)
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

    def test_short_flush_many_logs(self, shm_path):
        path = shm_path("test_flush")
        config = LoggerConfig(path=path, flush_interval_s=0.01)  # Very short
        _create_shm_ring(path, capacity_bytes=config.shm_capacity_bytes)
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

    def test_backpressure_ring_full(self, shm_path):
        path = shm_path("test_backpressure")
        # Tiny ring: 4KB
        _create_shm_ring(path, capacity_bytes=4096, num_rings=1)
        config = LoggerConfig(path=path, flush_interval_s=0.01)
        mock_handler = MockHandler()
        master = MasterLogger(config=config, log_handlers=[mock_handler])
        mock_handler.add_primary_config(config)

        # Create worker in a separate thread to avoid singleton conflict
        worker_result = []

        def flood_logs():
            worker = WorkerLogger(config=config, name="FloodWorker")
            try:
                for i in range(10000):
                    worker.info(f"log {i}".encode())
                worker_result.append(worker.is_running())
            finally:
                worker.shutdown()

        t = threading.Thread(target=flood_logs)
        t.start()
        t.join()
        assert worker_result and worker_result[0]
        master.shutdown()

    def test_rapid_create_shutdown(self, shm_path):
        for i in range(100):
            path = shm_path(f"test_rapid_{i}")
            _create_shm_ring(path, capacity_bytes=65536, num_rings=1)
            config = LoggerConfig(path=path, flush_interval_s=0.001)
            mock_handler = MockHandler()
            master = MasterLogger(config=config, log_handlers=[mock_handler])
            mock_handler.add_primary_config(config)

            # Create worker in a separate thread to avoid singleton conflict
            # with the master's internal worker.
            worker_result = []

            def create_and_log():
                worker = WorkerLogger(config=config, name=f"Worker_{i}")
                worker.info(b"hello")
                worker.shutdown()
                worker_result.append(True)

            t = threading.Thread(target=create_and_log)
            t.start()
            t.join()

            assert worker_result
            master.shutdown()
            if os.path.exists(path):
                os.unlink(path)

    def test_shutdown_while_logging(self, shm_path):
        path = shm_path("test_shutdown_while_logging")
        _create_shm_ring(path)
        config = LoggerConfig(path=path, flush_interval_s=0.01)
        worker = WorkerLogger(config=config, name="Logger")

        def log_continuously():
            while worker.is_running():
                worker.info(b"spam")

        t = threading.Thread(target=log_continuously)
        t.start()
        time.sleep(0.05)
        worker.shutdown()
        t.join(timeout=2.0)
        assert not t.is_alive()

    def test_multiple_masters_same_path(self, shm_path):
        path = shm_path("test_multi_master")
        _create_shm_ring(path)
        config = LoggerConfig(path=path)
        master1 = MasterLogger(config=config, log_handlers=[MockHandler()])
        try:
            with pytest.raises(
                RuntimeError, match="Only one WorkerLogger allowed per thread"
            ):
                MasterLogger(config=config, log_handlers=[MockHandler()])
        finally:
            master1.shutdown()

    def test_fork_safety(self, shm_path):
        path = shm_path("test_fork")
        _create_shm_ring(path)
        config = LoggerConfig(path=path)
        worker = WorkerLogger(config=config, name="Parent")
        worker.info(b"parent log")
        worker.shutdown()

        p = multiprocessing.Process(target=child_task, args=(path,))
        p.start()
        p.join(timeout=5.0)
        assert p.exitcode == 0
        if os.path.exists(path):
            os.unlink(path)
