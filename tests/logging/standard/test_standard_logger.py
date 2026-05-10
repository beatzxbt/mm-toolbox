"""Layer 2 — Component tests for the standard ``Logger`` implementation.

Covers message logging at every severity level, stdout suppression, handler
resilience (one failing handler must not block others), buffer management
(size threshold, interval, explicit flush), shutdown semantics, and format-
string expansion.
"""

import pytest
import time

from mm_toolbox.logging.standard.config import LoggerConfig, LogLevel
from mm_toolbox.logging.standard.handlers import BaseLogHandler
from mm_toolbox.logging.standard.logger import Logger


class RecordingHandler(BaseLogHandler):
    """Test-double handler that records every pushed buffer for assertions."""

    def __init__(self, should_raise: bool = False) -> None:
        super().__init__()
        self.should_raise = should_raise
        self.invocations: list[tuple[str, ...]] = []
        self.closed = False

    def push(self, buffer: list[str]) -> None:
        """Store the buffer or raise if simulating a faulty handler."""
        if self.should_raise:
            raise RuntimeError("intentional handler failure")
        self.invocations.append(tuple(buffer))

    def close(self) -> None:
        """Mark as closed and delegate to the base implementation."""
        self.closed = True
        super().close()


class TestLoggerLoggingBehavior:
    """Layer 2 — Core logging behaviour at every severity level."""

    def test_info_message_flushed(self) -> None:
        """Given a threshold of 1, an ``info()`` call is immediately flushed to the handler."""
        handler = RecordingHandler()
        config = LoggerConfig(
            base_level=LogLevel.INFO,
            do_stdout=False,
            flush_on_size=True,
            flush_size_threshold=1,
        )
        logger = Logger(name="logger-basic", config=config, handlers=[handler])
        logger.info("hello world")
        logger.shutdown()

        assert handler.invocations
        assert any(
            "hello world" in entry for call in handler.invocations for entry in call
        )

    def test_warning_message_flushed(self) -> None:
        """Given a threshold of 1, a ``warning()`` call is immediately flushed."""
        handler = RecordingHandler()
        config = LoggerConfig(
            base_level=LogLevel.WARNING,
            do_stdout=False,
            flush_on_size=True,
            flush_size_threshold=1,
        )
        logger = Logger(config=config, handlers=[handler])
        logger.warning("warn message")
        logger.shutdown()

        assert any(
            "warn message" in entry for call in handler.invocations for entry in call
        )

    def test_error_message_flushed(self) -> None:
        """Given a threshold of 1, an ``error()`` call is immediately flushed."""
        handler = RecordingHandler()
        config = LoggerConfig(
            base_level=LogLevel.ERROR,
            do_stdout=False,
            flush_on_size=True,
            flush_size_threshold=1,
        )
        logger = Logger(config=config, handlers=[handler])
        logger.error("error message")
        logger.shutdown()

        assert any(
            "error message" in entry for call in handler.invocations for entry in call
        )

    def test_all_levels_in_sequence(self) -> None:
        """Given TRACE level and a threshold of 5, all five severities appear in one batch."""
        handler = RecordingHandler()
        config = LoggerConfig(
            base_level=LogLevel.TRACE,
            do_stdout=False,
            flush_on_size=True,
            flush_size_threshold=5,
        )
        logger = Logger(config=config, handlers=[handler])
        logger.trace("t")
        logger.debug("d")
        logger.info("i")
        logger.warning("w")
        logger.error("e")
        logger.shutdown()

        all_msgs = "\n".join(entry for call in handler.invocations for entry in call)
        assert "t" in all_msgs and "d" in all_msgs and "i" in all_msgs
        assert "w" in all_msgs and "e" in all_msgs

    def test_level_filter_and_runtime_change(self) -> None:
        """Given INFO level, DEBUG messages are dropped; after lowering to DEBUG, they appear."""
        handler = RecordingHandler()
        config = LoggerConfig(
            do_stdout=False, flush_on_size=True, flush_size_threshold=1
        )
        logger = Logger(config=config, handlers=[handler])

        logger.debug("filtered debug")
        assert handler.invocations == []

        logger.set_log_level(LogLevel.DEBUG)
        logger.debug("visible debug")
        logger.shutdown()

        all_messages = "\n".join(
            entry for call in handler.invocations for entry in call
        )
        assert "visible debug" in all_messages
        assert "filtered debug" not in all_messages

    def test_trace_level_logging(self) -> None:
        """Given TRACE level, ``trace()`` messages are flushed like any other level."""
        handler = RecordingHandler()
        config = LoggerConfig(
            base_level=LogLevel.TRACE,
            do_stdout=False,
            flush_on_size=True,
            flush_size_threshold=1,
        )
        logger = Logger(config=config, handlers=[handler])
        logger.trace("trace me")
        logger.shutdown()

        assert any(
            "trace me" in entry for call in handler.invocations for entry in call
        )

    def test_multiple_messages_batch_together(self) -> None:
        """Given a threshold of 3, the first two messages stay buffered and the third triggers flush."""
        handler = RecordingHandler()
        config = LoggerConfig(
            base_level=LogLevel.INFO,
            do_stdout=False,
            flush_on_size=True,
            flush_size_threshold=3,
        )
        logger = Logger(config=config, handlers=[handler])
        logger.info("m1")
        logger.info("m2")
        assert handler.invocations == []  # Not flushed yet
        logger.info("m3")  # Hits threshold
        logger.shutdown()

        assert len(handler.invocations) == 1
        assert len(handler.invocations[0]) == 3
        assert any("m1" in entry for entry in handler.invocations[0])
        assert any("m2" in entry for entry in handler.invocations[0])
        assert any("m3" in entry for entry in handler.invocations[0])


class TestLoggerStdoutBehavior:
    """Layer 2 — Console-output suppression tests."""

    def test_stdout_enabled_prints(self, capsys: pytest.CaptureFixture) -> None:
        """Given ``do_stdout=True``, the message appears on standard output."""
        handler = RecordingHandler()
        config = LoggerConfig(
            base_level=LogLevel.INFO,
            do_stdout=True,
            flush_on_size=True,
            flush_size_threshold=1,
        )
        logger = Logger(config=config, handlers=[handler])
        logger.info("stdout message")
        logger.shutdown()

        captured = capsys.readouterr()
        assert "stdout message" in captured.out

    def test_stdout_disabled_suppresses_output(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        """Given ``do_stdout=False``, nothing is written to standard output."""
        handler = RecordingHandler()
        config = LoggerConfig(
            do_stdout=False, flush_on_size=True, flush_size_threshold=1
        )
        logger = Logger(config=config, handlers=[handler])
        logger.info("silent message")
        logger.shutdown()

        captured = capsys.readouterr()
        assert captured.out == ""


class TestLoggerErrorHandling:
    """Layer 2 — Handler-fault isolation tests."""

    def test_handler_exception_does_not_block_others(self) -> None:
        """Given one failing and one healthy handler, the healthy handler still receives the message."""
        failing = RecordingHandler(should_raise=True)
        healthy = RecordingHandler()
        config = LoggerConfig(
            do_stdout=False, flush_on_size=True, flush_size_threshold=1
        )
        logger = Logger(config=config, handlers=[failing, healthy])
        logger.info("resilient message")
        logger.shutdown()

        assert healthy.invocations
        assert any(
            "resilient message" in entry
            for call in healthy.invocations
            for entry in call
        )


class TestLoggerShutdownBehavior:
    """Layer 2 — Graceful shutdown and context-manager tests."""

    def test_shutdown_flushes_pending_buffer(self) -> None:
        """Given an unflushed buffer, ``shutdown()`` forces delivery before closing handlers."""
        handler = RecordingHandler()
        config = LoggerConfig(
            do_stdout=False,
            flush_on_size=True,
            flush_size_threshold=10,
        )
        logger = Logger(config=config, handlers=[handler])
        logger.info("needs shutdown flush")
        assert handler.invocations == []
        logger.shutdown()

        assert handler.invocations
        assert any(
            "needs shutdown flush" in entry
            for call in handler.invocations
            for entry in call
        )
        assert handler.closed

    def test_context_manager_auto_shutdown(self) -> None:
        """Given a logger used as a context manager, handlers are closed on exit."""
        handler = RecordingHandler()
        config = LoggerConfig(
            do_stdout=False, flush_on_size=True, flush_size_threshold=1
        )
        with Logger(config=config, handlers=[handler]) as logger:
            logger.info("context message")

        assert handler.closed
        assert any(
            "context message" in entry for call in handler.invocations for entry in call
        )


class TestLoggerBufferManagement:
    """Layer 2 — Buffer-flush strategy tests."""

    def test_buffer_flushes_on_size_threshold(self) -> None:
        """Given a threshold of 2, the second message triggers an immediate flush."""
        handler = RecordingHandler()
        config = LoggerConfig(
            do_stdout=False,
            flush_on_size=True,
            flush_size_threshold=2,
        )
        logger = Logger(config=config, handlers=[handler])
        logger.info("m0")
        logger.info("m1")  # Triggers flush
        logger.shutdown()

        assert len(handler.invocations) == 1
        assert len(handler.invocations[0]) == 2
        assert any("m0" in entry for entry in handler.invocations[0])
        assert any("m1" in entry for entry in handler.invocations[0])

    def test_multiple_handlers_receive_same_payload(self) -> None:
        """Given two handlers, both receive the identical formatted payload."""
        handler1 = RecordingHandler()
        handler2 = RecordingHandler()
        config = LoggerConfig(
            do_stdout=False, flush_on_size=True, flush_size_threshold=1
        )
        logger = Logger(config=config, handlers=[handler1, handler2])
        logger.info("multi handler")
        logger.shutdown()

        assert handler1.invocations
        assert handler2.invocations
        assert any(
            "multi handler" in entry for call in handler1.invocations for entry in call
        )
        assert any(
            "multi handler" in entry for call in handler2.invocations for entry in call
        )

    def test_explicit_flush(self) -> None:
        """Given an unflushed buffer, ``flush()`` delivers it without waiting for the threshold."""
        handler = RecordingHandler()
        config = LoggerConfig(
            do_stdout=False,
            flush_on_size=True,
            flush_size_threshold=100,
        )
        logger = Logger(config=config, handlers=[handler])
        logger.info("before flush")
        assert handler.invocations == []
        logger.flush()
        logger.shutdown()

        assert len(handler.invocations) == 1
        assert any("before flush" in entry for entry in handler.invocations[0])

    def test_flush_on_interval(self) -> None:
        """Given ``flush_on_interval=True`` and a 0.05-second interval, messages flush after the interval elapses."""
        handler = RecordingHandler()
        config = LoggerConfig(
            do_stdout=False,
            flush_on_size=False,
            flush_on_interval=True,
            flush_interval_s=0.05,
        )
        logger = Logger(config=config, handlers=[handler])
        logger.info("interval message")
        time.sleep(0.1)
        logger.info("trigger flush")
        logger.shutdown()

        assert handler.invocations
        all_msgs = "\n".join(entry for call in handler.invocations for entry in call)
        assert "interval message" in all_msgs

    def test_flush_on_size_disabled(self) -> None:
        """Given both size and interval flushing disabled, messages remain buffered until ``flush()`` is called."""
        handler = RecordingHandler()
        config = LoggerConfig(
            do_stdout=False,
            flush_on_size=False,
            flush_on_interval=False,
        )
        logger = Logger(config=config, handlers=[handler])
        logger.info("m1")
        logger.info("m2")
        logger.info("m3")
        assert handler.invocations == []
        logger.flush()
        logger.shutdown()

        assert len(handler.invocations) == 1
        assert len(handler.invocations[0]) == 3


class TestLoggerProperties:
    """Layer 1 — Accessor primitive tests."""

    def test_is_running_reflects_state(self) -> None:
        """Given a running logger, ``is_running()`` is True; after shutdown it is False."""
        handler = RecordingHandler()
        logger = Logger(handlers=[handler])
        assert logger.is_running() is True
        logger.shutdown()
        assert logger.is_running() is False

    def test_get_name_returns_name(self) -> None:
        """Given a name, ``get_name()`` returns it exactly."""
        logger = Logger(name="my-logger")
        assert logger.get_name() == "my-logger"

    def test_get_config_returns_config(self) -> None:
        """Given a config object, ``get_config()`` returns the identical reference."""
        config = LoggerConfig(base_level=LogLevel.DEBUG)
        logger = Logger(config=config)
        assert logger.get_config() is config
        assert logger.get_config().base_level == LogLevel.DEBUG


class TestLoggerValidation:
    """Layer 1 — Constructor-validation primitive tests."""

    def test_invalid_handler_type_raises(self) -> None:
        """Given a non-``BaseLogHandler`` object, construction raises TypeError."""
        with pytest.raises(TypeError, match="BaseLogHandler"):
            Logger(handlers=["not-a-handler"])

    def test_empty_logger_no_handlers(self) -> None:
        """Given no handlers, logging is a no-op and does not raise."""
        config = LoggerConfig(
            do_stdout=False, flush_on_size=True, flush_size_threshold=1
        )
        logger = Logger(config=config)
        logger.info("no handlers")
        logger.shutdown()


class TestLoggerEdgeCases:
    """Layer 2 — Edge-case behaviour tests."""

    def test_log_after_shutdown_is_noop(self) -> None:
        """Given a shut-down logger, subsequent ``info()`` calls do not produce new output."""
        handler = RecordingHandler()
        config = LoggerConfig(
            do_stdout=False, flush_on_size=True, flush_size_threshold=1
        )
        logger = Logger(config=config, handlers=[handler])
        logger.info("before shutdown")
        logger.shutdown()

        invocations_after_first_shutdown = list(handler.invocations)
        logger.info("after shutdown")
        logger.shutdown()

        assert handler.invocations == invocations_after_first_shutdown

    def test_double_shutdown_safe(self) -> None:
        """Given a logger already shut down, a second ``shutdown()`` does not raise."""
        handler = RecordingHandler()
        config = LoggerConfig(
            do_stdout=False, flush_on_size=True, flush_size_threshold=1
        )
        logger = Logger(config=config, handlers=[handler])
        logger.info("msg")
        logger.shutdown()
        logger.shutdown()

    def test_auto_flush_on_exit_false(self) -> None:
        """Given ``auto_flush_on_exit=False``, the logger can still be created and shut down cleanly."""
        config = LoggerConfig(auto_flush_on_exit=False)
        logger = Logger(config=config)
        logger.info("test")
        logger.shutdown()

    def test_format_string_expansion(self) -> None:
        """Given a custom format string, all placeholders (asctime, levelname, name, message) are expanded."""
        handler = RecordingHandler()
        config = LoggerConfig(
            do_stdout=False,
            flush_on_size=True,
            flush_size_threshold=1,
            str_format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
        logger = Logger(name="test-name", config=config, handlers=[handler])
        logger.info("test-msg")
        logger.shutdown()

        msg = handler.invocations[0][0]
        assert "test-name" in msg
        assert "INFO" in msg
        assert "test-msg" in msg
        assert "T" in msg
