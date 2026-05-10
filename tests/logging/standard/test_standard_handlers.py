"""Layer 2 — Component tests for standard logger handler infrastructure.

Covers ``BaseLogHandler`` lifecycle (open/close), error handling (stderr +
custom callback), and concrete subclasses: ``FileLogHandler`` (creation,
append, multi-flush), ``DiscordLogHandler`` (webhook URL validation), and
``TelegramLogHandler`` (config storage).
"""

import os
import tempfile

import pytest

from mm_toolbox.logging.standard.handlers import (
    BaseLogHandler,
    DiscordLogHandler,
    FileLogHandler,
    TelegramLogHandler,
)


class DummyPayloadHandler(BaseLogHandler):
    """A test-double handler that records every pushed buffer for assertions."""

    def __init__(self) -> None:
        super().__init__()
        self.received: list[tuple[str, ...]] = []
        self.closed = False

    def push(self, buffer: list[str]) -> None:
        """Store the buffer so tests can inspect it later."""
        self.received.append(tuple(buffer))

    def close(self) -> None:
        """Mark as closed and delegate to the base implementation."""
        self.closed = True
        super().close()


class TestBaseLogHandler:
    """Layer 1 — ``BaseLogHandler`` primitive lifecycle tests."""

    def test_open_sets_flag(self):
        """Given a fresh handler, ``open()`` sets the internal ``_is_open`` flag."""
        handler = DummyPayloadHandler()
        assert not handler._is_open
        handler.open()
        assert handler._is_open

    def test_close_sets_flag(self):
        """Given an open handler, ``close()`` clears ``_is_open`` and marks the handler closed."""
        handler = DummyPayloadHandler()
        handler.open()
        handler.close()
        assert handler.closed
        assert not handler._is_open

    def test_push_receives_buffer(self):
        """Given an open handler, ``push()`` delivers the exact buffer contents."""
        handler = DummyPayloadHandler()
        handler.open()
        handler.push(["msg1", "msg2"])
        assert handler.received == [("msg1", "msg2")]


class TestBaseLogHandlerErrorHandling:
    """Layer 2 — Error-handling behaviour of the base handler."""

    def test_handle_exception_writes_to_stderr(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        """Given no custom error handler, an exception is written to stderr with context."""
        handler = DummyPayloadHandler()
        exc = RuntimeError("test error")
        handler._handle_exception(exc, "push")

        captured = capsys.readouterr()
        assert "DummyPayloadHandler" in captured.err
        assert "push" in captured.err
        assert "test error" in captured.err

    def test_handle_exception_calls_custom_callback(self) -> None:
        """Given a custom error handler, it is invoked with the exception and context string."""
        handler = DummyPayloadHandler()
        called_with = []
        handler.set_error_handler(lambda exc, ctx: called_with.append((exc, ctx)))

        exc = ValueError("custom error")
        handler._handle_exception(exc, "flush")

        assert len(called_with) == 1
        assert called_with[0][0] is exc
        assert called_with[0][1] == "flush"

    def test_handle_exception_callback_failure_fallback(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        """Given a custom handler that itself raises, the original exception still reaches stderr."""
        handler = DummyPayloadHandler()
        handler.set_error_handler(
            lambda exc, ctx: (_ for _ in ()).throw(RuntimeError("callback fail"))
        )

        exc = ValueError("original error")
        handler._handle_exception(exc, "push")

        captured = capsys.readouterr()
        assert "DummyPayloadHandler" in captured.err
        assert "original error" in captured.err


class TestFileLogHandler:
    """Layer 2 — ``FileLogHandler`` concrete behaviour."""

    def test_file_handler_requires_txt_extension(self) -> None:
        """Given a non-``.txt`` path, construction raises ValueError."""
        with pytest.raises(ValueError):
            FileLogHandler("/tmp/log.dat")

    def test_file_handler_creates_file(self) -> None:
        """Given ``create=True`` and a missing file, the file is created on ``open()``."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            path = tmp.name
        os.remove(path)

        handler = FileLogHandler(path, create=True)
        handler.open()
        assert os.path.exists(path)
        handler.close()
        os.remove(path)

    def test_file_handler_appends_to_file(self) -> None:
        """Given an existing file with content, new logs are appended without truncation."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
            tmp.write("existing line\n")
            path = tmp.name

        handler = FileLogHandler(path)
        handler.open()
        handler.push(["msg1", "msg2"])
        handler.close()

        with open(path) as f:
            content = f.read()

        assert "existing line" in content
        assert "msg1" in content
        assert "msg2" in content
        os.remove(path)

    def test_file_handler_multiple_flushes(self) -> None:
        """Given multiple ``push()`` calls, each batch appears as a separate line."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            path = tmp.name

        handler = FileLogHandler(path)
        handler.open()
        handler.push(["batch1"])
        handler.push(["batch2", "batch3"])
        handler.close()

        with open(path) as f:
            lines = f.read().strip().split("\n")

        assert lines == ["batch1", "batch2", "batch3"]
        os.remove(path)

    def test_file_handler_open_close_lifecycle(self) -> None:
        """Given a handler, ``open()`` acquires the file handle and ``close()`` releases it."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            path = tmp.name

        handler = FileLogHandler(path)
        assert not handler._is_open
        handler.open()
        assert handler._is_open
        assert handler._file is not None
        handler.close()
        assert not handler._is_open
        assert handler._file is None
        os.remove(path)


class TestDiscordLogHandler:
    """Layer 1 — ``DiscordLogHandler`` primitive validation tests."""

    def test_discord_validates_webhook_url(self) -> None:
        """Given an invalid or non-Discord URL, construction raises ValueError."""
        with pytest.raises(ValueError):
            DiscordLogHandler("https://example.com/webhook")

        with pytest.raises(ValueError):
            DiscordLogHandler("not-a-url")

        handler = DiscordLogHandler("https://discord.com/api/webhooks/123/abc")
        assert handler.url == "https://discord.com/api/webhooks/123/abc"


class TestTelegramLogHandler:
    """Layer 1 — ``TelegramLogHandler`` primitive storage tests."""

    def test_telegram_stores_config(self) -> None:
        """Given a token and chat ID, both are retained and the URL contains the token."""
        handler = TelegramLogHandler("my-token", "my-chat-id")
        assert handler.chat_id == "my-chat-id"
        assert "my-token" in handler.url
