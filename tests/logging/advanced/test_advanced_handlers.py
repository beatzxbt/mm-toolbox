"""Layer 2 — Component tests for advanced logger handlers.

Covers ``BaseLogHandler`` abstract enforcement, lazy initialisation (JSON
encoder, HTTP session, event loop), future lifecycle and trimming, error
handling, thread-safe event-loop access, and concrete subclasses:
``FileLogHandler`` (creation, append, permission errors, disk-full handling),
``DiscordLogHandler`` (chunking, rate limiting, POST payload), and
``TelegramLogHandler`` (chunking, per-message POST). Also covers the internal
``_RateLimiter`` token-bucket behaviour.
"""

import asyncio
import json
import os
import shutil
import tempfile
import threading
from concurrent.futures import Future
from unittest.mock import AsyncMock, patch

import aiohttp
import pytest

from mm_toolbox.logging.advanced.config import LoggerConfig
from mm_toolbox.logging.advanced.handlers.base import BaseLogHandler, _RateLimiter
from mm_toolbox.logging.advanced.handlers.discord import DiscordLogHandler
from mm_toolbox.logging.advanced.handlers.file import FileLogHandler
from mm_toolbox.logging.advanced.handlers.telegram import TelegramLogHandler
from mm_toolbox.logging.advanced.pylog import PyLog, PyLogLevel


class TestBaseLogHandler:
    """Layer 1 — ``BaseLogHandler`` primitive tests."""

    def test_abstract_push(self):
        """Given direct instantiation of the abstract base, ``TypeError`` is raised."""
        with pytest.raises(TypeError):
            BaseLogHandler()  # type: ignore

    def test_format_log_requires_config(self):
        """Given a handler without a primary config, ``format_log`` raises ``RuntimeError``."""
        handler = FileLogHandler("test.txt")
        with pytest.raises(RuntimeError):
            handler.format_log(PyLog(1234567890, b"name", PyLogLevel.INFO, b"msg"))

    def test_format_log_accepts_memoryview_fields(self):
        """Given ``memoryview`` name and message fields, formatting succeeds."""
        handler = FileLogHandler("test.txt")
        config = LoggerConfig(str_format="%(name)s %(message)s")
        handler.add_primary_config(config)
        log = PyLog(
            1234567890,
            memoryview(b"name"),
            PyLogLevel.INFO,
            memoryview(b"msg"),
        )
        assert handler.format_log(log) == "name msg"

    def test_lazy_encode_json(self):
        """Given first use of ``encode_json``, the encoder is created lazily and works."""
        handler = FileLogHandler("test.txt")
        assert handler._encode_json is None
        encoder = handler.encode_json
        assert callable(encoder)
        assert handler._encode_json is not None
        assert encoder({"test": 1}) == b'{"test":1}'

    def test_lazy_http_session(self):
        """Given first use of ``http_session``, an ``aiohttp.ClientSession`` is created lazily."""
        handler = FileLogHandler("test.txt")
        assert handler._http_session is None
        session = handler.http_session
        assert isinstance(session, aiohttp.ClientSession)
        assert handler._http_session is not None

    def test_lazy_ev_loop(self):
        """Given first use of ``ev_loop``, an asyncio event loop is created lazily."""
        handler = FileLogHandler("test.txt")
        assert handler._ev_loop is None
        loop = handler.ev_loop
        assert isinstance(loop, asyncio.AbstractEventLoop)
        assert handler._ev_loop is not None

    def test_close_waits_for_futures(self):
        """Given a pending future, ``close()`` waits for it to complete before returning."""
        handler = FileLogHandler("test.txt")
        fut = handler._run_coro(asyncio.sleep(0.1))
        handler._track_future(fut)
        handler.close(timeout_s=1.0)
        assert fut.done()
        assert handler._loop_thread is None or not handler._loop_thread.is_alive()

    def test_future_trim_at_4096(self):
        """Given more than 4096 tracked futures, the list is trimmed to 2048 to bound memory."""
        handler = FileLogHandler("test.txt")
        for _ in range(4097):
            mock_fut = Future()
            handler._track_future(mock_fut)
        assert len(handler._futures) == 2048
        for _ in range(1000):
            mock_fut = Future()
            handler._track_future(mock_fut)
        assert len(handler._futures) <= 4096
        handler._futures.clear()

    def test_on_future_done_captures_exception(self):
        """Given a future that raises, the error handler receives the exception and context."""
        handler = FileLogHandler("test.txt")
        errors = []

        def callback(exc, ctx):
            errors.append((exc, ctx))

        handler.set_error_handler(callback)
        fut = Future()
        handler._track_future(fut)
        fut.set_exception(RuntimeError("test error"))
        assert len(errors) == 1
        assert isinstance(errors[0][0], RuntimeError)
        assert errors[0][1] == "handler task"

    def test_handle_exception_custom_callback(self):
        """Given a custom error handler, it is invoked with the exception and context."""
        handler = FileLogHandler("test.txt")
        errors = []

        def callback(exc, ctx):
            errors.append((exc, ctx))

        handler.set_error_handler(callback)
        exc = RuntimeError("test")
        handler._handle_exception(exc, "test_ctx")
        assert len(errors) == 1
        assert errors[0][0] is exc
        assert errors[0][1] == "test_ctx"

    def test_handle_exception_stderr_fallback(self, capsys):
        """Given no custom error handler, exceptions are written to stderr with context."""
        handler = FileLogHandler("test.txt")
        exc = RuntimeError("stderr test")
        handler._handle_exception(exc, "fallback")
        captured = capsys.readouterr()
        assert "stderr test" in captured.err
        assert "fallback" in captured.err
        assert "FileLogHandler" in captured.err

    def test_normalize_log_bytes_bytearray(self):
        """Given a ``bytearray``, it is normalised to ``bytes``."""
        result = BaseLogHandler._normalize_log_bytes(bytearray(b"hello"))
        assert result == b"hello"
        assert isinstance(result, bytes)

    def test_normalize_log_bytes_invalid_type(self):
        """Given an ``int``, normalisation raises ``TypeError``."""
        with pytest.raises(TypeError):
            BaseLogHandler._normalize_log_bytes(123)

    def test_ev_loop_thread_safe(self):
        """Given concurrent threads accessing ``ev_loop``, both receive the same loop instance."""
        handler = FileLogHandler("test.txt")
        loops = []

        def get_loop():
            loops.append(handler.ev_loop)

        t1 = threading.Thread(target=get_loop)
        t2 = threading.Thread(target=get_loop)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert len(loops) == 2
        assert loops[0] is loops[1]
        handler.close()


class TestFileLogHandler:
    """Layer 2 — ``FileLogHandler`` concrete behaviour tests."""

    @pytest.fixture
    def temp_file(self):
        """Yield a temporary ``.txt`` file path and clean it up after the test."""
        fd, path = tempfile.mkstemp(suffix=".txt")
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.remove(path)

    def test_init_valid(self, temp_file):
        """Given a valid path, the handler stores it and does not auto-create."""
        handler = FileLogHandler(temp_file)
        assert handler.filepath == temp_file
        assert not handler.create

    def test_init_create(self, temp_file):
        """Given ``create=True`` and a missing file, the file is created immediately."""
        os.remove(temp_file)
        _ = FileLogHandler(temp_file, create=True)
        assert os.path.exists(temp_file)

    def test_init_invalid_extension(self):
        """Given a non-``.txt`` extension, construction raises ``ValueError``."""
        with pytest.raises(ValueError):
            FileLogHandler("invalid.log")

    def test_push_writes_to_file(self, temp_file):
        """Given a handler with config, ``push()`` appends formatted logs to the file."""
        handler = FileLogHandler(temp_file)
        config = LoggerConfig()
        handler.add_primary_config(config)

        logs = [PyLog(1234567890, b"name", PyLogLevel.INFO, b"test message")]
        handler.push(logs)

        with open(temp_file) as f:
            content = f.read()
            assert "test message" in content

    def test_push_multiple_logs(self, temp_file):
        """Given multiple logs, each appears on its own line."""
        handler = FileLogHandler(temp_file)
        config = LoggerConfig(str_format="%(message)s")
        handler.add_primary_config(config)

        logs = [
            PyLog(1, b"name", PyLogLevel.INFO, b"msg1"),
            PyLog(2, b"name", PyLogLevel.WARNING, b"msg2"),
        ]
        handler.push(logs)

        with open(temp_file) as f:
            content = f.read().strip().split("\n")
            assert content == ["msg1", "msg2"]

    def test_create_false_file_missing(self):
        """Given ``create=False`` and a missing file, ``push()`` does not create it."""
        invalid_path = "/tmp/non_existent.txt"
        if os.path.exists(invalid_path):
            os.remove(invalid_path)
        handler = FileLogHandler(invalid_path, create=False)
        config = LoggerConfig()
        handler.add_primary_config(config)
        handler.push([PyLog(1, b"name", PyLogLevel.INFO, b"msg")])
        assert not os.path.exists(invalid_path)

    def test_permission_denied(self, temp_file):
        """Given a ``PermissionError`` on open, the handler logs the error and skips writing."""
        handler = FileLogHandler(temp_file)
        config = LoggerConfig()
        handler.add_primary_config(config)

        orig_open = open

        def deny_append(file: str, mode: str = "r", *args, **kwargs):
            if file == temp_file and mode == "a":
                raise PermissionError("permission denied")
            return orig_open(file, mode, *args, **kwargs)

        with patch("builtins.open", side_effect=deny_append):
            handler.push([PyLog(1, b"name", PyLogLevel.INFO, b"msg")])
        with open(temp_file) as f:
            assert f.read() == ""

    def test_create_with_directory(self):
        """Given a nested path, missing parent directories are created automatically."""
        with tempfile.TemporaryDirectory() as tmpdir:
            shutil.rmtree(tmpdir)
            path = os.path.join(tmpdir, "subdir", "test.txt")
            handler = FileLogHandler(path, create=True)
            assert os.path.exists(os.path.dirname(path))
            handler.close()

    def test_push_creates_file_if_missing(self):
        """Given ``create=True``, ``push()`` creates the file on first write."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.txt")
            handler = FileLogHandler(path, create=True)
            config = LoggerConfig(str_format="%(message)s")
            handler.add_primary_config(config)
            logs = [PyLog(1, b"name", PyLogLevel.INFO, b"msg")]
            handler.push(logs)
            assert os.path.exists(path)
            with open(path) as f:
                assert "msg" in f.read()
            handler.close()

    def test_push_disk_full(self, temp_file):
        """Given an ``OSError`` (disk full), the handler logs the error and does not crash."""
        handler = FileLogHandler(temp_file)
        config = LoggerConfig(str_format="%(message)s")
        handler.add_primary_config(config)
        with patch("builtins.open", side_effect=OSError("No space left")):
            handler.push([PyLog(1, b"name", PyLogLevel.INFO, b"msg")])
        handler.close()


class TestDiscordLogHandler:
    """Layer 2 — ``DiscordLogHandler`` concrete behaviour tests."""

    def test_init_valid(self):
        """Given a valid Discord webhook URL, construction succeeds."""
        url = "https://discord.com/api/webhooks/123/abc"
        handler = DiscordLogHandler(url)
        assert handler.url == url

    def test_init_invalid_url(self):
        """Given an invalid URL, construction raises ``ValueError``."""
        with pytest.raises(ValueError):
            DiscordLogHandler("invalid url")

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.post", new_callable=AsyncMock)
    async def test_push(self, mock_post):
        """Given a single log, ``push()`` schedules an async POST without blocking."""
        handler = DiscordLogHandler("https://discord.com/api/webhooks/123/abc")
        logs = [PyLog(1234567890, b"name", PyLogLevel.INFO, b"msg")]
        handler.push(logs)

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.post", new_callable=AsyncMock)
    async def test_push_multiple(self, mock_post):
        """Given two logs, they are concatenated with newlines and sent in one POST."""
        url = "https://discord.com/api/webhooks/123/abc"
        handler = DiscordLogHandler(url)
        config = LoggerConfig(str_format="%(message)s")
        handler.add_primary_config(config)

        logs = [
            PyLog(1, b"name", PyLogLevel.INFO, b"msg1"),
            PyLog(2, b"name", PyLogLevel.INFO, b"msg2"),
        ]
        handler.push(logs)

        await asyncio.sleep(0.1)

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == url
        assert call_args[1]["headers"] == {"Content-Type": "application/json"}
        data = json.loads(call_args[1]["data"])
        assert data["content"] == "msg1\nmsg2"

    def test_discord_chunking(self):
        """Given a 2500-character message, it is split into 1800-character chunks."""
        text = "a" * 2500
        chunks = DiscordLogHandler._chunk(text, 1800)
        assert len(chunks) == 2
        assert len(chunks[0]) == 1800
        assert len(chunks[1]) == 700

    def test_discord_rate_limiter(self):
        """Given construction, the internal rate limiter is configured for Discord (2.5/s, burst 5)."""
        handler = DiscordLogHandler("https://discord.com/api/webhooks/123/abc")
        assert handler._limiter._rate == 2.5
        assert handler._limiter._capacity == 5


class TestTelegramLogHandler:
    """Layer 2 — ``TelegramLogHandler`` concrete behaviour tests."""

    def test_init_valid(self):
        """Given a token and chat ID, the URL and chat ID are stored correctly."""
        handler = TelegramLogHandler("bot_token", "chat_id")
        assert handler.chat_id == "chat_id"
        assert handler.url.startswith("https://api.telegram.org/bot")

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.post", new_callable=AsyncMock)
    async def test_push(self, mock_post):
        """Given a single log, ``push()`` schedules an async POST with the correct payload."""
        handler = TelegramLogHandler("token", "chat")
        config = LoggerConfig(str_format="%(message)s")
        handler.add_primary_config(config)

        logs = [PyLog(1, b"name", PyLogLevel.INFO, b"test msg")]
        handler.push(logs)

        await asyncio.sleep(0.1)

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == handler.url
        data = json.loads(call_args[1]["data"])
        assert data["chat_id"] == "chat"
        assert data["text"] == "test msg"

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.post", new_callable=AsyncMock)
    async def test_push_multiple(self, mock_post):
        """Given two logs, each is sent in a separate POST because Telegram does not support batch text."""
        handler = TelegramLogHandler("token", "chat")
        config = LoggerConfig(str_format="%(message)s")
        handler.add_primary_config(config)
        mock_post.return_value.read = AsyncMock()

        logs = [
            PyLog(1, b"name", PyLogLevel.INFO, b"msg1"),
            PyLog(2, b"name", PyLogLevel.INFO, b"msg2"),
        ]
        handler.push(logs)

        for _ in range(50):
            if mock_post.call_count >= 2:
                break
            await asyncio.sleep(0.01)

        assert mock_post.call_count == 2
        calls = mock_post.call_args_list
        for i, call in enumerate(calls, 1):
            data = json.loads(call[1]["data"])
            assert data["text"] == f"msg{i}"

    def test_telegram_chunking(self):
        """Given a 4000-character message, it is split into 3500-character chunks."""
        text = "b" * 4000
        chunks = TelegramLogHandler._chunk(text, 3500)
        assert len(chunks) == 2
        assert len(chunks[0]) == 3500
        assert len(chunks[1]) == 500

    def test_telegram_rate_limiter(self):
        """Given construction, the internal rate limiter is configured for Telegram (1.0/s, burst 20)."""
        handler = TelegramLogHandler("token", "chat")
        assert handler._limiter._rate == 1.0
        assert handler._limiter._capacity == 20


class TestRateLimiter:
    """Layer 1 — ``_RateLimiter`` token-bucket primitive tests."""

    @pytest.mark.asyncio
    async def test_rate_limiter_basic(self):
        """Given a bucket of 5 tokens at 10/s, the 6th acquire waits because the bucket is empty."""
        limiter = _RateLimiter(10.0, 5)
        for _ in range(5):
            await limiter.acquire(1)
        start = asyncio.get_running_loop().time()
        await limiter.acquire(1)
        elapsed = asyncio.get_running_loop().time() - start
        assert elapsed > 0

    @pytest.mark.asyncio
    async def test_rate_limiter_burst(self):
        """Given a bucket of 10 tokens at 1/s, all 10 are acquired instantly (burst)."""
        limiter = _RateLimiter(1.0, 10)
        start = asyncio.get_running_loop().time()
        for _ in range(10):
            await limiter.acquire(1)
        elapsed = asyncio.get_running_loop().time() - start
        assert elapsed < 0.1
