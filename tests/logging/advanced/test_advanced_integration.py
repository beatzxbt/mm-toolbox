import multiprocessing
import os
import time
from queue import Queue

import pytest

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
    time.sleep(0.5)
    logger.shutdown()


def worker_large_msg(path, name, msg_size):
    config = LoggerConfig(path=path)
    logger = WorkerLogger(config=config, name=name)
    large_msg = b"x" * msg_size
    logger.info(msg_bytes=large_msg)
    time.sleep(0.5)
    logger.shutdown()


def worker_mixed_levels(path, name):
    config = LoggerConfig(path=path)
    logger = WorkerLogger(config=config, name=name)
    logger.trace("Trace msg")
    logger.debug("Debug msg")
    logger.info("Info msg")
    logger.warning("Warning msg")
    logger.error("Error msg")
    time.sleep(0.5)
    logger.shutdown()


class TestIntegration:
    @pytest.mark.parametrize(
        "num_workers, num_logs_per_worker", [(1, 10), (5, 10), (25, 10)]
    )
    def test_multiple_workers(self, num_workers, num_logs_per_worker):
        path = f"ipc:///tmp/test_integration_{os.getpid()}"
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

        # Give master time to process
        time.sleep(1)
        master.shutdown()

        # Collect received logs
        received = []
        while not mock_handler.received_logs.empty():
            received.append(mock_handler.received_logs.get())

        assert len(received) == num_workers * num_logs_per_worker

        # Check contents
        worker_logs = {f"Worker_{i}".encode(): 0 for i in range(num_workers)}
        for log in received:
            assert log.level == PyLogLevel.INFO
            worker_logs[log.name] += 1

        for count in worker_logs.values():
            assert count == num_logs_per_worker

    def test_high_throughput(self):
        path = f"ipc:///tmp/test_high_throughput_{os.getpid()}"
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

        time.sleep(2)  # Extra time for high load
        master.shutdown()

        received_count = 0
        while not mock_handler.received_logs.empty():
            mock_handler.received_logs.get()
            received_count += 1

        assert received_count == num_workers * num_logs_per_worker

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
    def test_different_levels(self, level):
        path = f"ipc:///tmp/test_levels_{os.getpid()}_{level}"
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

        time.sleep(1)
        master.shutdown()

        received = []
        while not mock_handler.received_logs.empty():
            received.append(mock_handler.received_logs.get())

        assert len(received) == 5
        for _, _, recv_level, _ in received:
            assert recv_level == level

    def test_large_messages(self):
        path = f"ipc:///tmp/test_large_{os.getpid()}"
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

        time.sleep(1)
        master.shutdown()

        received = []
        while not mock_handler.received_logs.empty():
            received.append(mock_handler.received_logs.get())

        assert len(received) == 1
        assert len(received[0][3]) == msg_size

    def test_mixed_levels(self):
        path = f"ipc:///tmp/test_mixed_{os.getpid()}"
        config = LoggerConfig(path=path, base_level=PyLogLevel.TRACE)
        mock_handler = MockHandler()
        master = MasterLogger(config=config, log_handlers=[mock_handler])
        mock_handler.add_primary_config(config)

        p = multiprocessing.Process(target=worker_mixed_levels, args=(path, "Worker"))
        p.start()
        p.join()

        time.sleep(1)
        master.shutdown()

        received_levels = set()
        while not mock_handler.received_logs.empty():
            _, _, level, _ = mock_handler.received_logs.get()
            received_levels.add(level)

        assert len(received_levels) == 5  # All levels

    def test_with_file_handler(self, tmp_path):
        log_file = tmp_path / "test.txt"
        path = f"ipc:///tmp/test_file_{os.getpid()}"
        config = LoggerConfig(path=path, str_format="%(levelname)s: %(message)s")
        file_handler = FileLogHandler(str(log_file), create=True)
        master = MasterLogger(config=config, log_handlers=[file_handler])
        file_handler.add_primary_config(config)

        p = multiprocessing.Process(target=worker_process, args=(path, "Worker", 3))
        p.start()
        p.join()

        time.sleep(1)
        master.shutdown()

        with open(log_file) as f:
            lines = f.readlines()
            assert len(lines) == 3
            for i, line in enumerate(lines):
                assert line.strip() == f"INFO: Log {i} from Worker"

    def test_short_flush_many_logs(self):
        path = f"ipc:///tmp/test_flush_{os.getpid()}"
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

        time.sleep(1)
        master.shutdown()

        received_count = 0
        while not mock_handler.received_logs.empty():
            mock_handler.received_logs.get()
            received_count += 1

        assert received_count == num_logs
