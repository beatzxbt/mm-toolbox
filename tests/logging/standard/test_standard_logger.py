"""Tests for the standard logger implementation."""

import asyncio
import time

import pytest

from mm_toolbox.logging.standard.config import LoggerConfig, LogLevel
from mm_toolbox.logging.standard.handlers import BaseLogHandler
from mm_toolbox.logging.standard.logger import Logger


class RecordingHandler(BaseLogHandler):
    """Handler that records payloads for assertions."""

    def __init__(self, delay: float = 0.0, should_raise: bool = False) -> None:
        super().__init__()
        self.delay = delay
        self.should_raise = should_raise
        self.invocations: list[tuple[str, ...]] = []
        self.closed = False

    async def push(self, buffer: list[str]) -> None:  # noqa: D401
        if self.should_raise:
            raise RuntimeError("intentional handler failure")
        if self.delay:
            await asyncio.sleep(self.delay)
        self.invocations.append(tuple(buffer))

    async def aclose(self) -> None:
        self.closed = True
        await super().aclose()


class TestLoggerLoggingBehavior:
    """Test core logging behaviour and level filtering."""

    def test_info_message_flushed(self, wait_for) -> None:
        handler = RecordingHandler()
        config = LoggerConfig(
            base_level=LogLevel.INFO,
            do_stdout=False,
            flush_interval_s=0.05,
            buffer_size=4,
        )
        logger = Logger(name="logger-basic", config=config, handlers=[handler])
        try:
            logger.info("hello world")
            assert wait_for(lambda: handler.invocations)
        finally:
            asyncio.run(logger.shutdown())

        assert handler.invocations
        assert any(
            "hello world" in entry for call in handler.invocations for entry in call
        )

    def test_level_filter_and_runtime_change(self, wait_for) -> None:
        handler = RecordingHandler()
        config = LoggerConfig(do_stdout=False, flush_interval_s=0.05)
        logger = Logger(config=config, handlers=[handler])
        try:
            logger.debug("filtered debug")
            assert not wait_for(lambda: handler.invocations, timeout_s=0.5)

            logger.set_log_level(LogLevel.DEBUG)
            logger.debug("visible debug")
            assert wait_for(lambda: handler.invocations)
        finally:
            asyncio.run(logger.shutdown())

        all_messages = "\n".join(
            entry for call in handler.invocations for entry in call
        )
        assert "visible debug" in all_messages
        assert "filtered debug" not in all_messages

    def test_trace_level_logging(self, wait_for) -> None:
        handler = RecordingHandler()
        config = LoggerConfig(
            base_level=LogLevel.TRACE,
            do_stdout=False,
            flush_interval_s=0.05,
        )
        logger = Logger(config=config, handlers=[handler])
        try:
            logger.trace("trace me")
            assert wait_for(lambda: handler.invocations)
        finally:
            asyncio.run(logger.shutdown())

        assert any(
            "trace me" in entry for call in handler.invocations for entry in call
        )


class TestLoggerStdoutBehavior:
    """Test stdout mirroring behaviour of the logger."""

    def test_stdout_enabled_prints(
        self, monkeypatch: pytest.MonkeyPatch, wait_for
    ) -> None:
        printed: list[str] = []
        monkeypatch.setattr("builtins.print", lambda msg: printed.append(msg))

        handler = RecordingHandler()
        config = LoggerConfig(
            base_level=LogLevel.INFO,
            do_stdout=True,
            flush_interval_s=0.05,
        )
        logger = Logger(config=config, handlers=[handler])
        try:
            logger.info("stdout message")
            assert wait_for(lambda: any("stdout message" in msg for msg in printed))
        finally:
            asyncio.run(logger.shutdown())

        assert any("stdout message" in msg for msg in printed)

    def test_stdout_disabled_suppresses_print(
        self, monkeypatch: pytest.MonkeyPatch, wait_for
    ) -> None:
        printed: list[str] = []
        monkeypatch.setattr("builtins.print", lambda msg: printed.append(msg))

        handler = RecordingHandler()
        config = LoggerConfig(do_stdout=False, flush_interval_s=0.05)
        logger = Logger(config=config, handlers=[handler])
        try:
            logger.info("silent message")
            assert not wait_for(
                lambda: any("silent message" in msg for msg in printed),
                timeout_s=0.5,
            )
        finally:
            asyncio.run(logger.shutdown())

        assert all("silent message" not in msg for msg in printed)


class TestLoggerErrorHandling:
    """Test robustness when handlers misbehave."""

    def test_handler_exception_does_not_block_others(self, wait_for) -> None:
        failing = RecordingHandler(should_raise=True)
        healthy = RecordingHandler()
        config = LoggerConfig(do_stdout=False, flush_interval_s=0.05)
        logger = Logger(config=config, handlers=[failing, healthy])
        try:
            logger.info("resilient message")
            assert wait_for(lambda: healthy.invocations)
        finally:
            asyncio.run(logger.shutdown())

        assert healthy.invocations
        assert any(
            "resilient message" in entry
            for call in healthy.invocations
            for entry in call
        )


class TestLoggerShutdownBehavior:
    """Test graceful shutdown semantics."""

    def test_shutdown_flushes_pending_buffer(self) -> None:
        handler = RecordingHandler(delay=0.05)
        config = LoggerConfig(do_stdout=False, flush_interval_s=5.0)
        logger = Logger(config=config, handlers=[handler])
        try:
            logger.info("needs shutdown flush")
        finally:
            asyncio.run(logger.shutdown())

        assert handler.invocations
        assert handler.closed
        assert any(
            "needs shutdown flush" in entry
            for call in handler.invocations
            for entry in call
        )


class TestLoggerBufferManagement:
    """Test buffer growth and batching behaviour."""

    def test_buffer_expands_when_capacity_exceeded(self, wait_for) -> None:
        handler = RecordingHandler()
        config = LoggerConfig(do_stdout=False, flush_interval_s=0.2, buffer_size=1)
        logger = Logger(config=config, handlers=[handler])
        try:
            logger.info("m0")
            logger.info("m1")
            logger.info("m2")
            assert wait_for(lambda: handler.invocations)
        finally:
            asyncio.run(logger.shutdown())

        combined = "\n".join(entry for call in handler.invocations for entry in call)
        assert "m0" in combined and "m1" in combined and "m2" in combined

    def test_multiple_handlers_receive_same_payload(self, wait_for) -> None:
        slow = RecordingHandler(delay=0.05)
        fast = RecordingHandler()
        config = LoggerConfig(do_stdout=False, flush_interval_s=0.05)
        logger = Logger(config=config, handlers=[slow, fast])
        try:
            start = time.time()
            logger.info("concurrent handlers")
            assert wait_for(lambda: slow.invocations and fast.invocations)
        finally:
            asyncio.run(logger.shutdown())
        duration = time.time() - start

        assert slow.invocations and fast.invocations
        assert any(
            "concurrent handlers" in entry
            for call in fast.invocations
            for entry in call
        )
        assert any(
            "concurrent handlers" in entry
            for call in slow.invocations
            for entry in call
        )
        # Ensure flush latency roughly bounded by the slow handler (not additive)
        assert duration < 0.5
