import asyncio
import json
import os
import tempfile
from unittest.mock import AsyncMock, patch

import aiohttp
import pytest

from mm_toolbox.logging.advanced.config import LoggerConfig
from mm_toolbox.logging.advanced.handlers.base import BaseLogHandler
from mm_toolbox.logging.advanced.handlers.discord import DiscordLogHandler
from mm_toolbox.logging.advanced.handlers.file import FileLogHandler
from mm_toolbox.logging.advanced.handlers.telegram import TelegramLogHandler
from mm_toolbox.logging.advanced.pylog import PyLog, PyLogLevel


class TestBaseLogHandler:
    def test_abstract_push(self):
        with pytest.raises(TypeError):
            BaseLogHandler()  # type: ignore # Cannot instantiate abstract class

    def test_format_log_requires_config(self):
        handler = FileLogHandler("test.txt")  # Concrete subclass
        with pytest.raises(RuntimeError):
            handler.format_log(PyLog(1234567890, b"name", PyLogLevel.INFO, b"msg"))

    def test_format_log_accepts_memoryview_fields(self):
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
        handler = FileLogHandler("test.txt")
        assert handler._encode_json is None
        encoder = handler.encode_json
        assert callable(encoder)
        assert handler._encode_json is not None
        # Test functionality
        assert encoder({"test": 1}) == b'{"test":1}'

    def test_lazy_http_session(self):
        handler = FileLogHandler("test.txt")
        assert handler._http_session is None
        session = handler.http_session
        assert isinstance(session, aiohttp.ClientSession)
        assert handler._http_session is not None

    def test_lazy_ev_loop(self):
        handler = FileLogHandler("test.txt")
        assert handler._ev_loop is None
        loop = handler.ev_loop
        assert isinstance(loop, asyncio.AbstractEventLoop)
        assert handler._ev_loop is not None


class TestFileLogHandler:
    @pytest.fixture
    def temp_file(self):
        fd, path = tempfile.mkstemp(suffix=".txt")
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.remove(path)

    def test_init_valid(self, temp_file):
        handler = FileLogHandler(temp_file)
        assert handler.filepath == temp_file
        assert not handler.create

    def test_init_create(self, temp_file):
        os.remove(temp_file)  # Ensure doesn't exist
        _ = FileLogHandler(temp_file, create=True)
        assert os.path.exists(temp_file)

    def test_init_invalid_extension(self):
        with pytest.raises(ValueError):
            FileLogHandler("invalid.log")

    def test_push_writes_to_file(self, temp_file):
        handler = FileLogHandler(temp_file)
        logs = [PyLog(1234567890, b"name", PyLogLevel.INFO, b"test message")]
        handler.push(
            logs
        )  # But format requires config, wait this might fail without config

        # To test properly, add config
        config = LoggerConfig()
        handler.add_primary_config(config)

        handler.push(logs)

        with open(temp_file) as f:
            content = f.read()
            assert "test message" in content  # Already str

    def test_push_multiple_logs(self, temp_file):
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
            assert content == ["msg1", "msg2"]  # str

    def test_create_false_file_missing(self):
        invalid_path = "/tmp/non_existent.txt"
        if os.path.exists(invalid_path):
            os.remove(invalid_path)
        handler = FileLogHandler(invalid_path, create=False)
        config = LoggerConfig()
        handler.add_primary_config(config)
        handler.push([PyLog(1, b"name", PyLogLevel.INFO, b"msg")])
        # Should print error but not raise; check no file created
        assert not os.path.exists(invalid_path)

    def test_permission_denied(self, temp_file):
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
        # Should print error
        with open(temp_file) as f:
            assert f.read() == ""  # Nothing written


class TestDiscordLogHandler:
    def test_init_valid(self):
        url = "https://discord.com/api/webhooks/123/abc"
        handler = DiscordLogHandler(url)
        assert handler.url == url

    def test_init_invalid_url(self):
        with pytest.raises(ValueError):
            DiscordLogHandler("invalid url")

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.post", new_callable=AsyncMock)
    async def test_push(self, mock_post):
        handler = DiscordLogHandler("https://discord.com/api/webhooks/123/abc")
        logs = [PyLog(1234567890, b"name", PyLogLevel.INFO, b"msg")]
        handler.push(logs)  # Creates task, but for test we can await if needed

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.post", new_callable=AsyncMock)
    async def test_push_multiple(self, mock_post):
        url = "https://discord.com/api/webhooks/123/abc"
        handler = DiscordLogHandler(url)
        config = LoggerConfig(str_format="%(message)s")
        handler.add_primary_config(config)

        logs = [
            PyLog(1, b"name", PyLogLevel.INFO, b"msg1"),
            PyLog(2, b"name", PyLogLevel.INFO, b"msg2"),
        ]
        handler.push(logs)

        # Since it's create_task, we need to run the loop briefly
        await asyncio.sleep(0.1)

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == url
        assert call_args[1]["headers"] == {"Content-Type": "application/json"}
        data = json.loads(call_args[1]["data"])
        assert data["content"] == "msg1\nmsg2"


class TestTelegramLogHandler:
    def test_init_valid(self):
        handler = TelegramLogHandler("bot_token", "chat_id")
        assert handler.chat_id == "chat_id"
        assert handler.url.startswith("https://api.telegram.org/bot")

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.post", new_callable=AsyncMock)
    async def test_push(self, mock_post):
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
