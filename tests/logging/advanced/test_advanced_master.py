import os
import struct
import time

import pytest

from mm_toolbox.logging.advanced.config import LoggerConfig
from mm_toolbox.logging.advanced.handlers.base import BaseLogHandler
from mm_toolbox.logging.advanced.handlers.file import FileLogHandler
from mm_toolbox.logging.advanced.master import MasterLogger
from mm_toolbox.logging.advanced.pylog import PyLogLevel
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
    def push(self, logs):
        pass  # Mock implementation


class CapturingHandler(BaseLogHandler):
    def __init__(self):
        super().__init__()
        self.logs = []

    def push(self, logs):
        self.logs.extend(logs)


class BadHandler(BaseLogHandler):
    def push(self, logs):
        raise RuntimeError("bad handler")


pytestmark = pytest.mark.timeout(10, method="thread")


class TestMasterLogger:
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
        logger = MasterLogger(config=default_config)
        assert logger.is_running()
        assert logger.get_config() == default_config
        logger.shutdown()

    def test_init_with_handlers(self, default_config, tmp_path):
        handlers = [MockHandler(), FileLogHandler(str(tmp_path / "test.txt"))]
        logger = MasterLogger(config=default_config, log_handlers=handlers)
        assert logger.is_running()
        logger.shutdown()

    def test_init_invalid_config_type(self):
        with pytest.raises(TypeError):
            MasterLogger(config="not a config")  # type: ignore

    def test_init_invalid_handlers_type(self, default_config):
        with pytest.raises(TypeError):
            MasterLogger(config=default_config, log_handlers="not a list")  # type: ignore

    def test_init_invalid_handler_class(self, default_config):
        class InvalidHandler:
            pass

        with pytest.raises(TypeError):
            MasterLogger(config=default_config, log_handlers=[InvalidHandler()])  # type: ignore

    @pytest.mark.parametrize(
        "method, args",
        [
            ("trace", {"msg_str": "trace msg"}),
            ("debug", {"msg_str": "debug msg"}),
            ("info", {"msg_str": "info msg"}),
            ("warning", {"msg_str": "warning msg"}),
            ("error", {"msg_str": "error msg"}),
            ("trace", {"msg_bytes": b"trace bytes"}),
            ("debug", {"msg_bytes": b"debug bytes"}),
            ("info", {"msg_bytes": b"info bytes"}),
            ("warning", {"msg_bytes": b"warning bytes"}),
            ("error", {"msg_bytes": b"error bytes"}),
        ],
    )
    def test_log_methods(self, default_config, method, args):
        logger = MasterLogger(config=default_config)
        log_func = getattr(logger, method)
        log_func(**args)  # Should not raise
        logger.shutdown()

    def test_log_with_both_str_and_bytes(self, default_config):
        logger = MasterLogger(config=default_config)
        with pytest.raises(TypeError):  # Assuming implementation doesn't allow both
            logger.info(msg_str="str", msg_bytes=b"bytes")
        logger.shutdown()

    def test_shutdown(self, default_config):
        logger = MasterLogger(config=default_config)
        assert logger.is_running()
        logger.shutdown()
        assert not logger.is_running()

    def test_double_shutdown(self, default_config):
        logger = MasterLogger(config=default_config)
        logger.shutdown()
        logger.shutdown()  # Should not raise
        assert not logger.is_running()

    def test_log_after_shutdown(self, default_config):
        logger = MasterLogger(config=default_config)
        logger.shutdown()
        logger.info("msg after shutdown")  # Should not add to batch, but no error

    def test_decode_truncated_message(self, default_config):
        master = MasterLogger(config=default_config)
        try:
            # Header says data_len=1000 but buffer is only 20 bytes
            msg = struct.pack("<BQI", 0, 0, 1000) + b"\x00" * 7
            with pytest.raises(ValueError):
                master._decode_worker_message(msg)
        finally:
            master.shutdown()

    def test_decode_overflow_data_len(self, default_config):
        master = MasterLogger(config=default_config)
        try:
            # data_start=13, data_len=0xFFFFFFFF (would overflow)
            msg = struct.pack("<BQI", 0, 0, 0xFFFFFFFF)
            with pytest.raises(ValueError):
                master._decode_worker_message(msg)
        finally:
            master.shutdown()

    def test_decode_impossible_num_logs(self, default_config):
        master = MasterLogger(config=default_config)
        try:
            # Valid header, data_len=12, worker_name="test", num_logs=0xFFFFFFFF
            msg = struct.pack("<BQI", 0, 0, 12)
            msg += struct.pack("<I", 4) + b"test"  # worker_name_len + name
            msg += struct.pack("<I", 0xFFFFFFFF)  # impossible num_logs
            with pytest.raises(ValueError):
                master._decode_worker_message(msg)
        finally:
            master.shutdown()

    def test_decode_invalid_level_fallback(self, default_config):
        master = MasterLogger(config=default_config)
        try:
            msg = struct.pack("<BQI", 0, 0, 29)
            msg += struct.pack("<I", 4) + b"test"  # worker name
            msg += struct.pack("<I", 1)  # num_logs = 1
            msg += struct.pack("<QB", 0, 99)  # timestamp + invalid level
            msg += struct.pack("<I", 4) + b"msg1"  # message
            logs = master._decode_worker_message(msg)
            assert len(logs) == 1
            assert logs[0].level == PyLogLevel.INFO
        finally:
            master.shutdown()

    def test_decode_empty_batch(self, default_config):
        master = MasterLogger(config=default_config)
        try:
            msg = struct.pack("<BQI", 0, 0, 12)
            msg += struct.pack("<I", 4) + b"test"
            msg += struct.pack("<I", 0)  # num_logs = 0
            logs = master._decode_worker_message(msg)
            assert logs == []
        finally:
            master.shutdown()

    def test_handler_exception_isolation(self, default_config):
        config = LoggerConfig(path=default_config.path, flush_interval_s=0.05)
        good = CapturingHandler()
        bad = BadHandler()
        master = MasterLogger(config=config, log_handlers=[bad, good])
        try:
            master.info("test msg")
            time.sleep(0.15)
            assert len(good.logs) >= 1
        finally:
            master.shutdown()

    def test_master_internal_worker(self, default_config):
        config = LoggerConfig(path=default_config.path, flush_interval_s=0.05)
        handler = CapturingHandler()
        master = MasterLogger(config=config, log_handlers=[handler])
        try:
            master.info("internal test")
            time.sleep(0.15)
            assert len(handler.logs) >= 1
            assert any(log.message == b"internal test" for log in handler.logs)
        finally:
            master.shutdown()

    def test_consume_exception_handling(self, default_config):
        class FakeTransport:
            def consume_all(self):
                raise RuntimeError("consume error")

            def close(self):
                pass

        master = MasterLogger(config=default_config, log_handlers=[MockHandler()])

        # Wait for background thread to create transport
        for _ in range(50):
            if master._transport is not None:
                break
            time.sleep(0.001)

        master._transport = FakeTransport()
        time.sleep(0.05)
        assert master.is_running()
        master.shutdown()
