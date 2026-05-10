"""Layer 2 — Component tests for ``WorkerLogger``.

Validates construction (config type, name handling), all severity-level
logging methods, graceful shutdown, idempotent double-shutdown, and the
no-op behaviour of logging after shutdown.
"""

import os
from pathlib import Path

import pytest

from mm_toolbox.logging.advanced.config import LoggerConfig
from mm_toolbox.logging.advanced.worker import WorkerLogger


class TestWorkerLogger:
    """Layer 2 — ``WorkerLogger`` behaviour tests."""

    @pytest.fixture
    def default_config(self, tmp_path: Path):
        """Return a default ``LoggerConfig`` backed by a temporary shared-memory path."""
        return LoggerConfig(path=str(tmp_path / f"test_worker_{os.getpid()}.shm"))

    def test_init_default(self, default_config):
        """Given a valid config, the worker starts running with an auto-generated name."""
        logger = WorkerLogger(config=default_config)
        assert logger.is_running()
        assert logger.get_name().startswith("WORKER")
        assert logger.get_config() == default_config
        logger.shutdown()

    @pytest.mark.parametrize("name", ["test_logger", "custom_name", ""])
    def test_init_with_name(self, default_config, name):
        """Given a custom name, it is stored; an empty string falls back to ``WORKER<pid>``."""
        logger = WorkerLogger(config=default_config, name=name)
        expected_name = name if name else f"WORKER{os.getpid()}"
        assert logger.get_name() == expected_name
        logger.shutdown()

    def test_init_invalid_config_type(self):
        """Given a string instead of ``LoggerConfig``, construction raises ``TypeError``."""
        with pytest.raises(TypeError):
            WorkerLogger(config="not a config")  # type: ignore

    def test_init_invalid_name_type(self, default_config):
        """Given an ``int`` for ``name``, construction raises ``TypeError``."""
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
        """Given any severity method, calling it on a running worker does not raise."""
        logger = WorkerLogger(config=default_config)
        log_func = getattr(logger, method)
        log_func(**args)
        logger.shutdown()

    def test_shutdown(self, default_config):
        """Given a running worker, ``shutdown()`` stops it and ``is_running()`` becomes False."""
        logger = WorkerLogger(config=default_config)
        assert logger.is_running()
        logger.shutdown()
        assert not logger.is_running()

    def test_double_shutdown(self, default_config):
        """Given a worker already shut down, a second ``shutdown()`` is a no-op."""
        logger = WorkerLogger(config=default_config)
        logger.shutdown()
        logger.shutdown()
        assert not logger.is_running()

    def test_log_after_shutdown(self, default_config):
        """Given a shut-down worker, ``info()`` is silently ignored and does not raise."""
        logger = WorkerLogger(config=default_config)
        logger.shutdown()
        logger.info(msg_bytes=b"msg after shutdown")
