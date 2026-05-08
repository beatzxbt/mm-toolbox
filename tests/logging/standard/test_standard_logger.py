"""Tests for the standard logger implementation."""

import pytest

from mm_toolbox.logging.standard.config import LoggerConfig, LogLevel
from mm_toolbox.logging.standard.handlers import BaseLogHandler
from mm_toolbox.logging.standard.logger import Logger


class RecordingHandler(BaseLogHandler):
    """Handler that records payloads for assertions."""

    def __init__(self, should_raise: bool = False) -> None:
        super().__init__()
        self.should_raise = should_raise
        self.invocations: list[tuple[str, ...]] = []
        self.closed = False

    def push(self, buffer: list[str]) -> None:
        if self.should_raise:
            raise RuntimeError("intentional handler failure")
        self.invocations.append(tuple(buffer))

    def close(self) -> None:
        self.closed = True
        super().close()


class TestLoggerLoggingBehavior:
    def test_info_message_flushed(self) -> None:
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
    def test_stdout_enabled_prints(self, capsys: pytest.CaptureFixture) -> None:
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
    def test_handler_exception_does_not_block_others(self) -> None:
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
    def test_shutdown_flushes_pending_buffer(self) -> None:
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
    def test_buffer_flushes_on_size_threshold(self) -> None:
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
        handler = RecordingHandler()
        config = LoggerConfig(
            do_stdout=False,
            flush_on_size=False,
            flush_on_interval=True,
            flush_interval_s=0.05,
        )
        logger = Logger(config=config, handlers=[handler])
        logger.info("interval message")
        # Simulate time passing
        import time

        time.sleep(0.1)
        logger.info("trigger flush")
        logger.shutdown()

        # The second message should trigger interval check and flush both
        assert handler.invocations
        all_msgs = "\n".join(entry for call in handler.invocations for entry in call)
        assert "interval message" in all_msgs

    def test_flush_on_size_disabled(self) -> None:
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
    def test_is_running_reflects_state(self) -> None:
        handler = RecordingHandler()
        logger = Logger(handlers=[handler])
        assert logger.is_running() is True
        logger.shutdown()
        assert logger.is_running() is False

    def test_get_name_returns_name(self) -> None:
        logger = Logger(name="my-logger")
        assert logger.get_name() == "my-logger"

    def test_get_config_returns_config(self) -> None:
        config = LoggerConfig(base_level=LogLevel.DEBUG)
        logger = Logger(config=config)
        assert logger.get_config() is config
        assert logger.get_config().base_level == LogLevel.DEBUG


class TestLoggerValidation:
    def test_invalid_handler_type_raises(self) -> None:
        with pytest.raises(TypeError, match="BaseLogHandler"):
            Logger(handlers=["not-a-handler"])

    def test_empty_logger_no_handlers(self) -> None:
        config = LoggerConfig(
            do_stdout=False, flush_on_size=True, flush_size_threshold=1
        )
        logger = Logger(config=config)
        logger.info("no handlers")
        logger.shutdown()
        # Should not raise; just no output


class TestLoggerEdgeCases:
    def test_log_after_shutdown_is_noop(self) -> None:
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
        handler = RecordingHandler()
        config = LoggerConfig(
            do_stdout=False, flush_on_size=True, flush_size_threshold=1
        )
        logger = Logger(config=config, handlers=[handler])
        logger.info("msg")
        logger.shutdown()
        logger.shutdown()  # Should not raise

    def test_auto_flush_on_exit_false(self) -> None:
        # Simply verify the logger can be created and used without error
        config = LoggerConfig(auto_flush_on_exit=False)
        logger = Logger(config=config)
        logger.info("test")
        logger.shutdown()

    def test_format_string_expansion(self) -> None:
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
        # asctime should produce an ISO8601-like string with T
        assert "T" in msg
