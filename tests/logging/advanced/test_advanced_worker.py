import os

import pytest

from mm_toolbox.logging.advanced.config import LoggerConfig
from mm_toolbox.logging.advanced.worker import WorkerLogger


class TestWorkerLogger:
    @pytest.fixture
    def default_config(self):
        return LoggerConfig(path=f"/tmp/test_worker_{os.getpid()}.shm")

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
        logger.info(
            msg_bytes=b"msg after shutdown"
        )  # Should not add to batch, but no error
