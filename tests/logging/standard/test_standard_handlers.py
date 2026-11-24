"""Layer 2 — Component tests for standard logger handler infrastructure.

Covers ``BaseLogHandler`` lifecycle (open/close), error handling (stderr +
custom callback), and concrete subclasses: ``FileLogHandler`` (creation,
append, multi-flush), ``DiscordLogHandler`` (webhook URL validation),
chunking, push delivery, and error handling), and
``TelegramLogHandler`` (config storage, chunking, push delivery, and
error handling).
"""

import asyncio
import json
from typing import cast
import os
import queue
import threading
import tempfile
from mm_toolbox.time import time_monotonic_ns
from unittest.mock import patch

import pytest

from mm_toolbox.logging.standard.handlers import (
    BaseLogHandler,
    DiscordLogHandler,
    FileLogHandler,
    TelegramLogHandler,
)


class _FakeResponse:
    def __init__(self, status: int = 200, headers: dict[str, str] | None = None):
        self.status = status
        self.headers = headers or {}

    async def read(self) -> bytes:
        return b"{}"


class _FakeRequestContext:
    def __init__(self, response: _FakeResponse):
        self.response = response

    async def __aenter__(self) -> _FakeResponse:
        return self.response

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class _FakeClientSession:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.closed = False
        self.close_calls = 0
        self.request_loops: list[asyncio.AbstractEventLoop] = []
        self.close_loop: asyncio.AbstractEventLoop | None = None
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def post(self, url: str, **kwargs):
        self.request_loops.append(asyncio.get_running_loop())
        self.calls.append((url, kwargs))
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        return _FakeRequestContext(response)

    async def close(self) -> None:
        self.close_calls += 1
        self.close_loop = asyncio.get_running_loop()
        self.closed = True


class DummyPayloadHandler(BaseLogHandler):
    """A test-double handler that records every pushed buffer for assertions."""

    def __init__(self) -> None:
        super().__init__()
        self.received: list[tuple[str, ...]] = []
        self.closed = False

    def push(self, buffer: list[str]) -> None:
        """Store the buffer so tests can inspect it later."""
        self.received.append(tuple(buffer))

    def close(self, timeout_s: float = 2.0) -> None:
        """Mark as closed and delegate to the base implementation."""
        self.closed = True
        super().close(timeout_s=timeout_s)


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
        """Given a single message exceeding max_chars, it is split safely."""
        msg = "x" * 3000
        chunks = discord_handler._chunk_messages([msg], max_chars=2000)
        assert [len(chunk) for chunk in chunks] == [2000, 1000]


class TestDiscordPush:
    """Layer 2 — bounded, async-safe Discord delivery tests."""

    def test_push_success(self, capsys):
        session = _FakeClientSession([_FakeResponse()])
        with patch(
            "mm_toolbox.logging.standard.handlers.base.aiohttp.ClientSession",
            return_value=session,
        ):
            handler = DiscordLogHandler("https://discord.com/api/webhooks/123/abc")
            handler.open()
            handler.push(["test message"])
            handler.close()

        assert len(session.calls) == 1
        assert json.loads(cast(bytes, session.calls[0][1]["data"])) == {
            "content": "test message"
        }
        assert "Discord webhook returned" not in capsys.readouterr().err

    def test_push_retries_429_then_delivers(self):
        session = _FakeClientSession(
            [
                _FakeResponse(429, {"Retry-After": "0"}),
                _FakeResponse(),
            ]
        )
        with patch(
            "mm_toolbox.logging.standard.handlers.base.aiohttp.ClientSession",
            return_value=session,
        ):
            handler = DiscordLogHandler("https://discord.com/api/webhooks/123/abc")
            handler.push(["test message"])
            handler.close()
        assert len(session.calls) == 2

    def test_push_http_error_retries_are_bounded_and_observable(self):
        session = _FakeClientSession([_FakeResponse(500) for _ in range(3)])
        errors = []
        with patch(
            "mm_toolbox.logging.standard.handlers.base.aiohttp.ClientSession",
            return_value=session,
        ):
            handler = DiscordLogHandler("https://discord.com/api/webhooks/123/abc")
            handler.set_error_handler(lambda exc, ctx: errors.append((exc, ctx)))
            handler.push(["test message"])
            handler.close()

        assert len(session.calls) == 3
        assert isinstance(errors[0][0], RuntimeError)
        assert "returned 500" in str(errors[0][0])
        assert errors[0][1] == "remote delivery"

    @pytest.mark.asyncio
    async def test_push_from_running_event_loop_is_nonblocking(self):
        session = _FakeClientSession([_FakeResponse()])
        with patch(
            "mm_toolbox.logging.standard.handlers.base.aiohttp.ClientSession",
            return_value=session,
        ):
            handler = DiscordLogHandler("https://discord.com/api/webhooks/123/abc")
            handler.open()
            started = time_monotonic_ns() / 1e9
            handler.push(["async-loop message"])
            elapsed = (time_monotonic_ns() / 1e9) - started
            assert elapsed < 0.1
            handler.close()
        assert len(session.calls) == 1

    @pytest.mark.asyncio
    async def test_delivery_queue_is_bounded(self, monkeypatch):
        handler = DiscordLogHandler("https://discord.com/api/webhooks/123/abc")
        handler._delivery_queue = queue.Queue(maxsize=1)
        started = threading.Event()
        release = threading.Event()
        errors = []

        async def blocked_delivery(item):
            started.set()
            await asyncio.to_thread(release.wait)

        monkeypatch.setattr(handler, "_deliver_chunk", blocked_delivery)
        monkeypatch.setattr(
            handler,
            "_chunk_messages",
            lambda messages, max_chars: ["one", "two", "three"],
        )
        handler.set_error_handler(lambda exc, ctx: errors.append((exc, ctx)))
        handler.push(["batch"])
        assert started.wait(1.0)
        assert any(isinstance(exc, OverflowError) for exc, _ in errors)
        release.set()
        handler.close()


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
        """Given a single message exceeding max_chars, it is split safely."""
        msg = "x" * 5000
        chunks = telegram_handler._chunk_messages([msg], max_chars=4096)
        assert [len(chunk) for chunk in chunks] == [4096, 904]


class TestTelegramPush:
    """Layer 2 — bounded Telegram delivery and request errors."""

    def test_push_success_and_multiple_logs(self):
        session = _FakeClientSession([_FakeResponse(), _FakeResponse()])
        with patch(
            "mm_toolbox.logging.standard.handlers.base.aiohttp.ClientSession",
            return_value=session,
        ):
            handler = TelegramLogHandler("my-token", "my-chat-id")
            handler.open()
            handler.push(["msg1", "msg2"])
            handler.close()

        assert [
            json.loads(cast(bytes, call[1]["data"]))["text"] for call in session.calls
        ] == ["msg1\nmsg2"]
        assert all(
            json.loads(cast(bytes, call[1]["data"]))["chat_id"] == "my-chat-id"
            for call in session.calls
        )

    def test_push_http_error_retries_are_bounded_and_observable(self):
        session = _FakeClientSession([_FakeResponse(500) for _ in range(3)])
        errors = []
        with patch(
            "mm_toolbox.logging.standard.handlers.base.aiohttp.ClientSession",
            return_value=session,
        ):
            handler = TelegramLogHandler("my-token", "my-chat-id")
            handler.set_error_handler(lambda exc, ctx: errors.append((exc, ctx)))
            handler.push(["test message"])
            handler.close(timeout_s=4.0)

        assert len(session.calls) == 3
        assert "returned 500" in str(errors[0][0])
        assert errors[0][1] == "remote delivery"


class _Standard(BaseLogHandler):
    def push(self, buffer):
        pass


def test_standard_immediate_close_drains_every_queued_delivery():
    handler = _Standard()
    delivered = []

    async def deliver(item):
        delivered.append(item)

    for item in range(20):
        assert handler._enqueue_delivery(item, deliver)
    handler.close()
    assert delivered == list(range(20))
    assert handler._delivery_thread is not None
    assert not handler._delivery_thread.is_alive()


def test_standard_close_has_one_deadline_and_reports_queued_loss():
    handler = _Standard()
    errors = []
    entered = threading.Event()
    cancelled = threading.Event()
    discarded = threading.Event()
    release_discard_callback = threading.Event()

    def on_error(error, context):
        errors.append(error)
        if "discarded" in str(error):
            discarded.set()
            release_discard_callback.wait(timeout=2.0)

    handler.set_error_handler(on_error)

    async def never(item):
        entered.set()
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    for item in range(20):
        handler._enqueue_delivery(item, never)
    assert entered.wait(timeout=1)
    thread = handler._delivery_thread
    assert thread is not None
    original_join = thread.join
    clock_reads = 0

    def monotonic_at_deadline():
        nonlocal clock_reads
        clock_reads += 1
        return 10.0 if clock_reads == 1 else 10.1

    with (
        patch(
            "mm_toolbox.logging.standard.handlers.base.time_monotonic_ns",
            new=lambda: int(monotonic_at_deadline() * 1e9),
        ),
        patch.object(thread, "join", wraps=original_join) as thread_join,
    ):
        try:
            handler.close(timeout_s=0.1)
            assert thread_join.call_args.kwargs["timeout"] == pytest.approx(0.0)
            assert cancelled.wait(timeout=2.0)
            assert discarded.wait(timeout=2.0)
        finally:
            release_discard_callback.set()
            original_join(timeout=2.0)

    assert handler._delivery_thread is not None
    assert not handler._delivery_thread.is_alive()
    assert errors
    assert any("discarded" in str(error) for error in errors)
    assert handler._delivery_queue.empty()
    assert cancelled.is_set()


@pytest.mark.parametrize("service", ["discord", "telegram"])
def test_remote_session_reused_until_close_and_recreated_on_reopen(service):
    sessions: list[_FakeClientSession] = []
    owner_loops: list[asyncio.AbstractEventLoop] = []

    def create_session(**kwargs):
        owner_loops.append(asyncio.get_running_loop())
        session = _FakeClientSession([_FakeResponse(), _FakeResponse()])
        sessions.append(session)
        return session

    handler = (
        DiscordLogHandler("https://discord.com/api/webhooks/test")
        if service == "discord"
        else TelegramLogHandler("token", "chat")
    )
    with patch(
        "mm_toolbox.logging.standard.handlers.base.aiohttp.ClientSession",
        side_effect=create_session,
    ) as factory:
        for cycle in range(2):
            handler.open()
            handler.push([f"{cycle}:first"])
            handler.push([f"{cycle}:second"])
            handler.close(timeout_s=3.0)
            handler.close()
            assert factory.call_count == cycle + 1
            session = sessions[cycle]
            assert len(session.calls) == 2
            assert session.request_loops == [owner_loops[cycle]] * 2
            assert session.close_loop is owner_loops[cycle]
            assert session.closed
            assert session.close_calls == 1
            assert handler._http_session is None
            assert owner_loops[cycle].is_closed()
    assert owner_loops[0] is not owner_loops[1]


def test_remote_session_closed_after_delivery_deadline():
    entered = threading.Event()

    class StalledResponse(_FakeResponse):
        async def read(self) -> bytes:
            entered.set()
            await asyncio.Future()
            return b""

    session = _FakeClientSession([StalledResponse(status=500)])
    handler = DiscordLogHandler("https://discord.com/api/webhooks/test")
    errors: list[BaseException] = []
    handler.set_error_handler(lambda exc, context: errors.append(exc))
    with patch(
        "mm_toolbox.logging.standard.handlers.base.aiohttp.ClientSession",
        return_value=session,
    ):
        handler.push(["blocked"])
        assert entered.wait(1.0)
        thread = handler._delivery_thread
        assert thread is not None
        try:
            handler.close(timeout_s=0.05)
        finally:
            thread.join(timeout=1.0)
    assert not thread.is_alive()
    assert session.closed
    assert session.close_calls == 1
    assert session.close_loop is session.request_loops[0]
    assert session.close_loop.is_closed()
    assert handler._http_session is None
    assert any(isinstance(exc, TimeoutError) for exc in errors)


def test_remote_session_close_error_does_not_leave_loop_open():
    class BrokenCloseSession(_FakeClientSession):
        async def close(self) -> None:
            await super().close()
            raise OSError("injected session close failure")

    session = BrokenCloseSession([_FakeResponse()])
    handler = DiscordLogHandler("https://discord.com/api/webhooks/test")
    errors: list[tuple[BaseException, str]] = []
    handler.set_error_handler(lambda exc, context: errors.append((exc, context)))
    with patch(
        "mm_toolbox.logging.standard.handlers.base.aiohttp.ClientSession",
        return_value=session,
    ):
        handler.push(["message"])
        handler.close()
        handler.close()
    assert len(session.calls) == 1
    assert session.close_calls == 1
    assert session.close_loop is not None
    assert session.close_loop.is_closed()
    assert handler._http_session is None
    assert len(errors) == 1
    assert str(errors[0][0]) == "injected session close failure"
    assert errors[0][1] == "session close"
