import os
import threading
import time

import pytest

from mm_toolbox.logging.advanced.config import LoggerConfig
from mm_toolbox.logging.advanced.pylog import PyLogLevel
from mm_toolbox.logging.advanced.worker import WorkerLogger
from mm_toolbox.ringbuffer.shm.mpsc import ShmMpscProducer


def _create_shm_ring(
    path: str, capacity_bytes: int = 65536, num_rings: int = 1
) -> None:
    """Create a shared-memory ring at ``path`` for workers to attach."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    producer = ShmMpscProducer(
        path=path,
        capacity_bytes=capacity_bytes,
        num_rings=num_rings,
        create=True,
        unlink_on_close=False,
    )
    producer.close()


class TestWorkerLogger:
    @pytest.fixture(autouse=True)
    def _clear_worker_singleton(self):
        from mm_toolbox.logging.advanced.worker import _worker_logger_local

        if hasattr(_worker_logger_local, "logger"):
            delattr(_worker_logger_local, "logger")
        yield
        if hasattr(_worker_logger_local, "logger"):
            delattr(_worker_logger_local, "logger")

    @pytest.fixture
    def shm_path(self, tmp_path):
        path = str(tmp_path / "test_ring.shm")
        _create_shm_ring(path)
        yield path
        if os.path.exists(path):
            os.unlink(path)

    @pytest.fixture
    def default_config(self, shm_path):
        return LoggerConfig(path=shm_path)

    def test_init_default(self, default_config):
        logger = WorkerLogger(config=default_config)
        assert logger.is_running()
        assert logger.get_name().startswith("WORKER")
        assert logger.get_config() == default_config
        logger.shutdown()

    @pytest.mark.parametrize("name", ["test_logger", "custom_name", ""])
    def test_init_with_name(self, default_config, name):
        logger = WorkerLogger(config=default_config, name=name)
        expected_name = name if name else f"WORKER{os.getpid()}"
        assert logger.get_name() == expected_name
        logger.shutdown()

    def test_init_invalid_config_type(self):
        with pytest.raises(TypeError):
            WorkerLogger(config="not a config")  # type: ignore

    def test_init_invalid_name_type(self, default_config):
        with pytest.raises(TypeError):
            WorkerLogger(config=default_config, name=123)  # type: ignore

    @pytest.mark.parametrize(
        "method, args",
        [
            ("trace", {"msg_bytes": b"trace bytes"}),
            ("debug", {"msg_bytes": b"debug bytes"}),
            ("info", {"msg_bytes": b"info bytes"}),
            ("warning", {"msg_bytes": b"warning bytes"}),
            ("error", {"msg_bytes": b"error bytes"}),
        ],
    )
    def test_log_methods(self, default_config, method, args):
        logger = WorkerLogger(config=default_config)
        log_func = getattr(logger, method)
        log_func(**args)  # Should not raise
        logger.shutdown()

    def test_shutdown(self, default_config):
        logger = WorkerLogger(config=default_config)
        assert logger.is_running()
        logger.shutdown()
        assert not logger.is_running()

    def test_double_shutdown(self, default_config):
        logger = WorkerLogger(config=default_config)
        logger.shutdown()
        logger.shutdown()  # Should not raise
        assert not logger.is_running()

    def test_log_after_shutdown(self, default_config):
        logger = WorkerLogger(config=default_config)
        logger.shutdown()
        logger.info(b"msg after shutdown")  # Should not add to batch, but no error

    def test_concurrent_logging(self, default_config):
        worker = WorkerLogger(config=default_config)
        errors = []
        threads = []

        def log_many():
            try:
                for _ in range(1000):
                    worker.info(b"concurrent log")
            except Exception as e:
                errors.append(e)

        for _ in range(10):
            t = threading.Thread(target=log_many)
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        assert not errors
        worker.shutdown()

    def test_size_based_flush_trigger_messages(self, default_config):
        config = LoggerConfig(
            path=default_config.path,
            max_batch_messages=5,
            max_batch_bytes=1048576,
            flush_interval_s=10.0,
        )
        worker = WorkerLogger(config=config)

        for i in range(4):
            worker.info(b"msg")
            assert worker._num_pending_logs == i + 1

        worker.info(b"trigger msg")
        assert worker._num_pending_logs == 0  # Flushed

        worker.shutdown()

    def test_size_based_flush_trigger_bytes(self, default_config):
        config = LoggerConfig(
            path=default_config.path,
            max_batch_messages=10000,
            max_batch_bytes=100,
            flush_interval_s=10.0,
        )
        worker = WorkerLogger(config=config)

        # Each empty log is 13 bytes (8+1+4+0).
        # 7 logs = 91 bytes, 8th = 104 bytes -> triggers flush.
        for i in range(7):
            worker.info(b"")
            assert worker._num_pending_logs == i + 1

        worker.info(b"")
        assert worker._num_pending_logs == 0

        worker.shutdown()

    def test_empty_bytes_message(self, default_config):
        worker = WorkerLogger(config=default_config)
        worker.info(b"")  # Should not crash
        worker.shutdown()

    def test_large_message(self, default_config):
        worker = WorkerLogger(config=default_config)
        large_msg = b"x" * (1024 * 1024)
        try:
            worker.info(large_msg)
        except MemoryError:
            pass  # Expected if message exceeds fixed batch capacity
        assert worker.is_running()
        worker.shutdown()

    def test_singleton_guard(self, default_config):
        worker1 = WorkerLogger(config=default_config, name="First")
        try:
            with pytest.raises(
                RuntimeError, match="Only one WorkerLogger allowed per thread"
            ):
                WorkerLogger(config=default_config, name="Second")
        finally:
            worker1.shutdown()

    def test_log_level_filtering(self, default_config):
        config = LoggerConfig(
            path=default_config.path,
            base_level=PyLogLevel.WARNING,
        )
        worker = WorkerLogger(config=config)

        worker.trace(b"trace")
        worker.debug(b"debug")
        worker.info(b"info")
        assert worker._num_pending_logs == 0

        worker.warning(b"warning")
        assert worker._num_pending_logs == 1

        worker.shutdown()

    def test_exception_in_flush_loop(self, default_config):
        config = LoggerConfig(path=default_config.path, flush_interval_s=0.05)
        worker = WorkerLogger(config=config)
        worker.info(b"test msg")

        class MockTransport:
            def insert_char(self, *args):
                raise RuntimeError("transport error")

            def insert(self, *args):
                raise RuntimeError("transport error")

            def close(self):
                pass

        worker._transport = MockTransport()

        # Wait for background thread to attempt flush and recover
        time.sleep(0.15)

        assert worker._num_pending_logs == 0
        assert worker.is_running()
        worker.shutdown()
