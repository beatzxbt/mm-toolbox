"""Layer 2 — Component tests for ``MasterLogger``.

Validates construction (config and handler validation), all severity-level
logging methods, graceful shutdown, idempotent double-shutdown, and the
no-op behaviour of logging after shutdown.
"""

from pathlib import Path

import pytest

from mm_toolbox.logging.advanced.config import LoggerConfig
from mm_toolbox.logging.advanced.handlers.file import FileLogHandler
from mm_toolbox.logging.advanced.master import MasterLogger

pytestmark = pytest.mark.timeout(10, method="thread")


class TestMasterLogger:
    """Layer 2 — ``MasterLogger`` behaviour tests."""

    @pytest.fixture
    def default_config(self, tmp_path: Path):
        """Return a default ``LoggerConfig`` backed by a temporary shared-memory path."""
        return LoggerConfig(path=str(tmp_path / "test_master.shm"))

    def test_init_default(self, default_config):
        """Given a valid config, the master starts running and exposes the config."""
        logger = MasterLogger(config=default_config)
        assert logger.is_running()
        assert logger.get_config() == default_config
        logger.shutdown()

    def test_init_with_handlers(self, default_config, tmp_path: Path):
        """Given a list of valid handlers, the master starts without error."""
        handlers = [FileLogHandler(str(tmp_path / "test.txt"))]
        logger = MasterLogger(config=default_config, log_handlers=handlers)
        assert logger.is_running()
        logger.shutdown()

    def test_init_invalid_config_type(self):
        """Given a string instead of ``LoggerConfig``, construction raises ``TypeError``."""
        with pytest.raises(TypeError):
            MasterLogger(config="not a config")  # type: ignore

    def test_init_invalid_handlers_type(self, default_config):
        """Given a non-list for ``log_handlers``, construction raises ``TypeError``."""
        with pytest.raises(TypeError):
            MasterLogger(config=default_config, log_handlers="not a list")  # type: ignore

    def test_init_invalid_handler_class(self, default_config):
        """Given a handler that is not a ``BaseLogHandler`` subclass, construction raises ``TypeError``."""
        class InvalidHandler:
            pass

        with pytest.raises(TypeError):
            MasterLogger(config=default_config, log_handlers=[InvalidHandler()])  # type: ignore

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
        """Given any severity method, calling it on a running master does not raise."""
        logger = MasterLogger(config=default_config)
        try:
            log_func = getattr(logger, method)
            log_func(**args)
        finally:
            logger.shutdown()

    def test_shutdown(self, default_config):
        """Given a running master, ``shutdown()`` stops it and ``is_running()`` becomes False."""
        logger = MasterLogger(config=default_config)
        try:
            assert logger.is_running()
        finally:
            logger.shutdown()
        assert not logger.is_running()

    def test_double_shutdown(self, default_config):
        """Given a master already shut down, a second ``shutdown()`` is a no-op."""
        logger = MasterLogger(config=default_config)
        logger.shutdown()
        logger.shutdown()
        assert not logger.is_running()

    def test_log_after_shutdown(self, default_config):
        """Given a shut-down master, ``info()`` is silently ignored and does not raise."""
        logger = MasterLogger(config=default_config)
        logger.shutdown()
        logger.info(
            msg_bytes=b"msg after shutdown"
        )
