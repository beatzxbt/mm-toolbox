"""Layer 2 — Component tests for ``MasterLogger``.

Validates construction (config and handler validation), all severity-level
logging methods, graceful shutdown, idempotent double-shutdown, and the
no-op behaviour of logging after shutdown.
"""

from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

from mm_toolbox.logging.advanced.config import LoggerConfig
from mm_toolbox.logging.advanced.handlers.base import BaseLogHandler
from mm_toolbox.logging.advanced.handlers.file import FileLogHandler
from mm_toolbox.logging.advanced.master import MasterLogger
from mm_toolbox.logging.advanced.worker import WorkerLogger
import mm_toolbox.logging.advanced.master as master_module

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
        logger.info(msg_bytes=b"msg after shutdown")

    def test_shutdown_deadline_keeps_transport_mapped_while_handler_is_active(
        self, tmp_path: Path, monkeypatch
    ):
        """A stalled dispatch cannot make shutdown unmap transport in use by its thread."""

        class BlockingHandler(BaseLogHandler):
            def __init__(self):
                super().__init__()
                self.started = threading.Event()
                self.release = threading.Event()

            def push(self, logs):
                self.started.set()
                self.release.wait(2.0)

        joined = []

        class ObservedThread(threading.Thread):
            def join(self, timeout=None):
                joined.append(timeout)
                # Keep the stalled drain alive without a wall-clock assertion.
                return super().join(timeout=0)

        monkeypatch.setattr(
            master_module,
            "threading",
            SimpleNamespace(Thread=ObservedThread, Event=threading.Event),
        )
        monkeypatch.setattr(
            master_module, "time", SimpleNamespace(monotonic=lambda: 10.0)
        )

        path = tmp_path / "blocked_master.shm"
        config = LoggerConfig(path=str(path), flush_interval_s=0.01)
        handler = BlockingHandler()
        logger = MasterLogger(config=config, log_handlers=[handler])
        worker = WorkerLogger(config=config, name="blocked-worker")
        try:
            worker.info(msg_bytes=b"wait for blocking handler")
            worker.shutdown()
            assert handler.started.wait(2.0)
            assert path.exists()

            logger.shutdown(timeout_s=0.05)
            assert joined == [pytest.approx(0.05)]
            assert path.exists()
        finally:
            handler.release.set()
            worker.shutdown()

        cleanup_deadline = time.monotonic() + 2.0
        while path.exists() and time.monotonic() < cleanup_deadline:
            time.sleep(0.01)
        assert not path.exists()

    def test_receive_loss_before_consumer_attachment_is_reported(
        self, tmp_path, monkeypatch, capsys
    ):
        """The master starts its drop baseline at mapping creation, not attachment."""
        attached, release = threading.Event(), threading.Event()
        original_consumer = master_module.ShmMpscConsumer

        def delayed_consumer(**kwargs):
            attached.set()
            assert release.wait(2)
            return original_consumer(**kwargs)

        monkeypatch.setattr(master_module, "ShmMpscConsumer", delayed_consumer)
        config = LoggerConfig(
            path=str(tmp_path / "startup_loss.shm"),
            shm_capacity_bytes=4096,
            max_batch_messages=1,
        )
        logger = MasterLogger(config=config)
        worker = None
        try:
            assert attached.wait(1)
            worker = WorkerLogger(config=config, name="early-worker")
            for _ in range(100):
                worker.info(b"x" * 64)
            worker.shutdown()
            dropped = worker.get_transport_stats()["overwritten_batches"]
            assert dropped > 0
        finally:
            release.set()
            if worker is not None:
                worker.shutdown()
            logger.shutdown()
        assert (
            f"[MasterLogger] SHM transport dropped {dropped} log batches (cumulative {dropped})"
            in capsys.readouterr().err
        )
