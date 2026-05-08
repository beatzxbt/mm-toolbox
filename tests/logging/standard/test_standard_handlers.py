"""Tests covering standard logger handler infrastructure."""

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
    def __init__(self) -> None:
        super().__init__()
        self.received: list[tuple[str, ...]] = []
        self.closed = False

    def push(self, buffer: list[str]) -> None:
        self.received.append(tuple(buffer))

    def close(self) -> None:
        self.closed = True
        super().close()


class TestBaseLogHandler:
    def test_open_sets_flag(self):
        handler = DummyPayloadHandler()
        assert not handler._is_open
        handler.open()
        assert handler._is_open

    def test_close_sets_flag(self):
        handler = DummyPayloadHandler()
        handler.open()
        handler.close()
        assert handler.closed
        assert not handler._is_open

    def test_push_receives_buffer(self):
        handler = DummyPayloadHandler()
        handler.open()
        handler.push(["msg1", "msg2"])
        assert handler.received == [("msg1", "msg2")]


class TestBaseLogHandlerErrorHandling:
    def test_handle_exception_writes_to_stderr(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        handler = DummyPayloadHandler()
        exc = RuntimeError("test error")
        handler._handle_exception(exc, "push")

        captured = capsys.readouterr()
        assert "DummyPayloadHandler" in captured.err
        assert "push" in captured.err
        assert "test error" in captured.err

    def test_handle_exception_calls_custom_callback(self) -> None:
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
    def test_file_handler_requires_txt_extension(self) -> None:
        with pytest.raises(ValueError):
            FileLogHandler("/tmp/log.dat")

    def test_file_handler_creates_file(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            path = tmp.name
        os.remove(path)

        handler = FileLogHandler(path, create=True)
        handler.open()
        assert os.path.exists(path)
        handler.close()
        os.remove(path)

    def test_file_handler_appends_to_file(self) -> None:
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
    def test_discord_validates_webhook_url(self) -> None:
        with pytest.raises(ValueError):
            DiscordLogHandler("https://example.com/webhook")

        with pytest.raises(ValueError):
            DiscordLogHandler("not-a-url")

        # Valid URL should not raise
        handler = DiscordLogHandler("https://discord.com/api/webhooks/123/abc")
        assert handler.url == "https://discord.com/api/webhooks/123/abc"


class TestTelegramLogHandler:
    def test_telegram_stores_config(self) -> None:
        handler = TelegramLogHandler("my-token", "my-chat-id")
        assert handler.chat_id == "my-chat-id"
        assert "my-token" in handler.url
