"""Layer 2 — Component tests for standard logger handler infrastructure.

Covers ``BaseLogHandler`` lifecycle (open/close), error handling (stderr +
custom callback), and concrete subclasses: ``FileLogHandler`` (creation,
append, multi-flush), ``DiscordLogHandler`` (webhook URL validation),
chunking, push delivery, and error handling), and
``TelegramLogHandler`` (config storage, chunking, push delivery, and
error handling).
"""

import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

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


class TestDiscordChunking:
    """Layer 1 — ``DiscordLogHandler._chunk_messages`` primitive tests."""

    @pytest.fixture
    def discord_handler(self):
        """Return a configured DiscordLogHandler.

        Returns:
            DiscordLogHandler: Handler instance for chunking tests.
        """
        return DiscordLogHandler("https://discord.com/api/webhooks/123/abc")

    def test_chunk_empty_input(self, discord_handler):
        """Given an empty list, ``_chunk_messages`` returns an empty list."""
        chunks = discord_handler._chunk_messages([], max_chars=2000)
        assert chunks == []

    def test_chunk_single_message(self, discord_handler):
        """Given a single message, it is returned as one chunk."""
        chunks = discord_handler._chunk_messages(["hello"], max_chars=2000)
        assert chunks == ["hello"]

    def test_chunk_exact_boundary(self, discord_handler):
        """Given messages that exactly fit the limit, they form a single chunk."""
        msg1 = "a" * 999
        msg2 = "b" * 1000
        # 999 + 1 (newline) + 1000 = 2000
        chunks = discord_handler._chunk_messages([msg1, msg2], max_chars=2000)
        assert len(chunks) == 1
        assert chunks[0] == f"{msg1}\n{msg2}"

    def test_chunk_boundary_plus_one(self, discord_handler):
        """Given messages that exceed the limit by one char, they split into two chunks."""
        msg1 = "a" * 999
        msg2 = "b" * 1001
        # 999 + 1 + 1001 = 2001 > 2000
        chunks = discord_handler._chunk_messages([msg1, msg2], max_chars=2000)
        assert len(chunks) == 2
        assert chunks[0] == msg1
        assert chunks[1] == msg2

    def test_chunk_multi_chunk(self, discord_handler):
        """Given many messages, they are grouped into multiple chunks."""
        messages = [f"message_number_{i}_with_some_padding" for i in range(100)]
        chunks = discord_handler._chunk_messages(messages, max_chars=2000)
        assert len(chunks) > 1
        # Reconstruct and verify all messages are preserved
        reconstructed = []
        for chunk in chunks:
            reconstructed.extend(chunk.split("\n"))
        assert reconstructed == messages

    def test_chunk_single_message_over_max(self, discord_handler):
        """Given a single message exceeding max_chars, it still forms its own chunk."""
        msg = "x" * 3000
        chunks = discord_handler._chunk_messages([msg], max_chars=2000)
        assert chunks == [msg]


class TestDiscordPush:
    """Layer 2 — ``DiscordLogHandler.push`` and ``_push_chunks`` composite tests."""

    @pytest.fixture
    def mock_session(self):
        """Return a mocked aiohttp ClientSession for push tests.

        Returns:
            MagicMock: Configured mock session context manager.
        """
        mock_response = MagicMock()
        mock_response.status = 200

        mock_post_cm = AsyncMock()
        mock_post_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post_cm.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_post_cm)

        mock_session_cm = AsyncMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cm.__aexit__ = AsyncMock(return_value=False)

        return mock_session_cm

    def test_push_success(self, mock_session, capsys):
        """Given valid messages, ``push`` sends each chunk successfully."""
        with patch(
            "mm_toolbox.logging.standard.handlers.discord.aiohttp.ClientSession",
            return_value=mock_session,
        ):
            handler = DiscordLogHandler("https://discord.com/api/webhooks/123/abc")
            handler.open()
            handler.push(["test message"])
            handler.close()

        captured = capsys.readouterr()
        assert "Discord webhook returned" not in captured.err

    def test_push_http_error(self, capsys):
        """Given a 500 response, ``push`` logs the error via ``_handle_exception``."""
        mock_response = MagicMock()
        mock_response.status = 500

        mock_post_cm = AsyncMock()
        mock_post_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post_cm.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_post_cm)

        mock_session_cm = AsyncMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cm.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "mm_toolbox.logging.standard.handlers.discord.aiohttp.ClientSession",
            return_value=mock_session_cm,
        ):
            handler = DiscordLogHandler("https://discord.com/api/webhooks/123/abc")
            handler.open()
            handler.push(["test message"])
            handler.close()

        captured = capsys.readouterr()
        assert "Discord webhook returned 500" in captured.err

    def test_push_exception(self, capsys):
        """Given a connection exception, ``push`` logs the error via ``_handle_exception``."""
        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=Exception("connection failed"))

        mock_session_cm = AsyncMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cm.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "mm_toolbox.logging.standard.handlers.discord.aiohttp.ClientSession",
            return_value=mock_session_cm,
        ):
            handler = DiscordLogHandler("https://discord.com/api/webhooks/123/abc")
            handler.open()
            handler.push(["test message"])
            handler.close()

        captured = capsys.readouterr()
        assert "connection failed" in captured.err


class TestTelegramLogHandler:
    """Layer 1 — ``TelegramLogHandler`` primitive storage tests."""

    def test_telegram_stores_config(self) -> None:
        """Given a token and chat ID, both are retained and the URL contains the token."""
        handler = TelegramLogHandler("my-token", "my-chat-id")
        assert handler.chat_id == "my-chat-id"
        assert "my-token" in handler.url


class TestTelegramChunking:
    """Layer 1 — ``TelegramLogHandler._chunk_messages`` primitive tests."""

    @pytest.fixture
    def telegram_handler(self):
        """Return a configured TelegramLogHandler.

        Returns:
            TelegramLogHandler: Handler instance for chunking tests.
        """
        return TelegramLogHandler("my-token", "my-chat-id")

    def test_chunk_empty_input(self, telegram_handler):
        """Given an empty list, ``_chunk_messages`` returns an empty list."""
        chunks = telegram_handler._chunk_messages([], max_chars=4096)
        assert chunks == []

    def test_chunk_single_message(self, telegram_handler):
        """Given a single message, it is returned as one chunk."""
        chunks = telegram_handler._chunk_messages(["hello"], max_chars=4096)
        assert chunks == ["hello"]

    def test_chunk_exact_boundary(self, telegram_handler):
        """Given messages that exactly fit the limit, they form a single chunk."""
        msg1 = "a" * 2047
        msg2 = "b" * 2048
        # 2047 + 1 + 2048 = 4096
        chunks = telegram_handler._chunk_messages([msg1, msg2], max_chars=4096)
        assert len(chunks) == 1
        assert chunks[0] == f"{msg1}\n{msg2}"

    def test_chunk_boundary_plus_one(self, telegram_handler):
        """Given messages that exceed the limit by one char, they split into two chunks."""
        msg1 = "a" * 2047
        msg2 = "b" * 2049
        # 2047 + 1 + 2049 = 4097 > 4096
        chunks = telegram_handler._chunk_messages([msg1, msg2], max_chars=4096)
        assert len(chunks) == 2
        assert chunks[0] == msg1
        assert chunks[1] == msg2

    def test_chunk_multi_chunk(self, telegram_handler):
        """Given many messages, they are grouped into multiple chunks."""
        messages = [f"message_number_{i}_with_some_padding" for i in range(200)]
        chunks = telegram_handler._chunk_messages(messages, max_chars=4096)
        assert len(chunks) > 1
        reconstructed = []
        for chunk in chunks:
            reconstructed.extend(chunk.split("\n"))
        assert reconstructed == messages

    def test_chunk_single_message_over_max(self, telegram_handler):
        """Given a single message exceeding max_chars, it still forms its own chunk."""
        msg = "x" * 5000
        chunks = telegram_handler._chunk_messages([msg], max_chars=4096)
        assert chunks == [msg]


class TestTelegramPush:
    """Layer 2 — ``TelegramLogHandler.push`` and ``_push_chunks`` composite tests."""

    @pytest.fixture
    def mock_session(self):
        """Return a mocked aiohttp ClientSession for push tests.

        Returns:
            MagicMock: Configured mock session context manager.
        """
        mock_response = MagicMock()
        mock_response.status = 200

        mock_post_cm = AsyncMock()
        mock_post_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post_cm.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_post_cm)

        mock_session_cm = AsyncMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cm.__aexit__ = AsyncMock(return_value=False)

        return mock_session_cm

    def test_push_success(self, mock_session, capsys):
        """Given valid messages, ``push`` sends each chunk successfully."""
        with patch(
            "mm_toolbox.logging.standard.handlers.telegram.aiohttp.ClientSession",
            return_value=mock_session,
        ):
            handler = TelegramLogHandler("my-token", "my-chat-id")
            handler.open()
            handler.push(["test message"])
            handler.close()

        captured = capsys.readouterr()
        assert "Telegram API returned" not in captured.err

    def test_push_http_error(self, capsys):
        """Given a 500 response, ``push`` logs the error via ``_handle_exception``."""
        mock_response = MagicMock()
        mock_response.status = 500

        mock_post_cm = AsyncMock()
        mock_post_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post_cm.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_post_cm)

        mock_session_cm = AsyncMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cm.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "mm_toolbox.logging.standard.handlers.telegram.aiohttp.ClientSession",
            return_value=mock_session_cm,
        ):
            handler = TelegramLogHandler("my-token", "my-chat-id")
            handler.open()
            handler.push(["test message"])
            handler.close()

        captured = capsys.readouterr()
        assert "Telegram API returned 500" in captured.err

    def test_push_exception(self, capsys):
        """Given a connection exception, ``push`` logs the error via ``_handle_exception``."""
        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=Exception("connection failed"))

        mock_session_cm = AsyncMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cm.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "mm_toolbox.logging.standard.handlers.telegram.aiohttp.ClientSession",
            return_value=mock_session_cm,
        ):
            handler = TelegramLogHandler("my-token", "my-chat-id")
            handler.open()
            handler.push(["test message"])
            handler.close()

        captured = capsys.readouterr()
        assert "connection failed" in captured.err
