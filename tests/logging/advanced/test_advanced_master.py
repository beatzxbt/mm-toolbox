import pytest
pytestmark = pytest.mark.timeout(10, method="thread")

from mm_toolbox.logging.advanced.config import LoggerConfig
from mm_toolbox.logging.advanced.handlers.base import BaseLogHandler
from mm_toolbox.logging.advanced.handlers.file import FileLogHandler
from mm_toolbox.logging.advanced.master import MasterLogger


class MockHandler(BaseLogHandler):
    def push(self, logs):
        pass  # Mock implementation


class TestMasterLogger:
    @pytest.fixture
    def default_config(self):
        return LoggerConfig(path="ipc:///tmp/test_master")

    def test_init_default(self, default_config):
        logger = MasterLogger(config=default_config)
        assert logger.is_running()
        assert logger.get_config() == default_config
        logger.shutdown()

    def test_init_with_handlers(self, default_config):
        handlers = [MockHandler(), FileLogHandler("test.txt")]
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
