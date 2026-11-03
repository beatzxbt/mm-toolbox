"""The same remote-delivery rules apply to both logger implementations."""

import asyncio
from collections.abc import Coroutine
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock

import aiohttp
import pytest

import mm_toolbox.logging._remote as remote
from mm_toolbox.logging.advanced.handlers.base import BaseLogHandler as AdvancedBase
from mm_toolbox.logging.standard.handlers.base import BaseLogHandler as StandardBase
from mm_toolbox.rate_limiter import RateLimiter


class _Standard(StandardBase):
    def push(self, buffer: list[str]) -> None:
        pass


class _Advanced(AdvancedBase):
    def push(self, logs) -> None:
        pass


class _Response:
    def __init__(self, status=200, headers=None, body=b"{}"):
        self.status = status
        self.headers = headers or {}
        self.body = body

    async def read(self) -> bytes:
        return self.body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class _Session:
    closed = False

    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls: list[bytes] = []

    def post(self, url: str, *, headers: dict[str, str], data: bytes):
        self.calls.append(data)
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        return response

    async def close(self) -> None:
        self.closed = True


def _operation(
    handler: _Standard | _Advanced, session: _Session
) -> Coroutine[object, None, None]:
    if isinstance(handler, _Standard):
        return handler._post_with_retries(
            session,
            "https://unused",
            headers={},
            payload={"text": "hello"},
            service="test",
        )
    handler._http_session = session
    return handler._post_json_with_retries(
        "https://unused", {"text": "hello"}, limiter=None, service="test"
    )


@pytest.mark.parametrize("handler_type", [_Standard, _Advanced])
@pytest.mark.parametrize(
    "status", [200, 204, 400, 401, 403, 404, 429, 500, 502, 503, 504]
)
@pytest.mark.asyncio
async def test_status_policy_and_bounded_retries(handler_type, status, monkeypatch):
    sleep = AsyncMock()
    monkeypatch.setattr(remote.asyncio, "sleep", sleep)
    session = _Session([_Response(status) for _ in range(3)])
    handler = handler_type()
    try:
        if 200 <= status < 300:
            await _operation(handler, session)
            assert len(session.calls) == 1
            sleep.assert_not_awaited()
        else:
            with pytest.raises(RuntimeError, match=f"returned {status}"):
                await _operation(handler, session)
            retryable = status == 429 or status >= 500
            assert len(session.calls) == (3 if retryable else 1)
            assert [call.args[0] for call in sleep.await_args_list] == (
                [0.1, 0.2] if retryable else []
            )
        assert all(payload == b'{"text":"hello"}' for payload in session.calls)
    finally:
        await session.close()
        handler.close()


@pytest.mark.parametrize("handler_type", [_Standard, _Advanced])
@pytest.mark.parametrize("status", [429, 503])
@pytest.mark.parametrize(
    "headers,body", [({"Retry-After": "60"}, b"{}"), ({}, b'{"retry_after":60}')]
)
@pytest.mark.asyncio
async def test_long_retry_after_fails_without_early_retry(
    handler_type, status, headers, body
):
    handler = handler_type()
    session = _Session([_Response(status, headers, body)])
    try:
        with pytest.raises(RuntimeError, match="exceeds the remaining delivery budget"):
            await _operation(handler, session)
        assert len(session.calls) == 1
    finally:
        await session.close()
        handler.close()


@pytest.mark.parametrize("handler_type", [_Standard, _Advanced])
@pytest.mark.asyncio
async def test_transient_network_error_then_success(handler_type, monkeypatch):
    sleep = AsyncMock()
    monkeypatch.setattr(remote.asyncio, "sleep", sleep)
    session = _Session([aiohttp.ClientConnectionError("offline"), _Response()])
    handler = handler_type()
    try:
        await _operation(handler, session)
        assert len(session.calls) == 2
        sleep.assert_awaited_once_with(0.1)
    finally:
        await session.close()
        handler.close()


@pytest.mark.asyncio
async def test_retry_budget_includes_time_spent_waiting_and_in_requests(monkeypatch):
    clock = iter([0, 0, 9_900_000_001])
    monkeypatch.setattr(remote, "time_monotonic_ns", lambda: next(clock))
    sleep = AsyncMock()
    monkeypatch.setattr(remote.asyncio, "sleep", sleep)
    session = _Session([_Response(500)])
    with pytest.raises(TimeoutError, match="10-second budget"):
        await remote.post_json_with_retries(
            session, "unused", {}, headers={}, limiter=None, service="test"
        )
    assert len(session.calls) == 1
    sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_delivery_waits_for_internal_rate_reservation(monkeypatch):
    limiter = Mock(spec=RateLimiter)
    limiter.try_consume.side_effect = [Mock(allowed=False), Mock(allowed=True)]
    sleep = AsyncMock()
    monkeypatch.setattr(remote.asyncio, "sleep", sleep)
    session = _Session([_Response()])
    await remote.post_json_with_retries(
        session, "unused", {}, headers={}, limiter=limiter, service="test"
    )
    assert limiter.try_consume.call_count == 2
    assert len(session.calls) == 1
    sleep.assert_awaited_once_with(0.01)


@pytest.mark.asyncio
async def test_cancelled_rate_wait_never_sends_request(monkeypatch):
    limiter = Mock(spec=RateLimiter)
    limiter.try_consume.return_value = Mock(allowed=False)
    monkeypatch.setattr(
        remote.asyncio, "sleep", AsyncMock(side_effect=asyncio.CancelledError)
    )
    session = _Session([])
    with pytest.raises(asyncio.CancelledError):
        await remote.post_json_with_retries(
            session, "unused", {}, headers={}, limiter=limiter, service="test"
        )
    assert not session.calls


def test_http_date_uses_fractional_internal_wall_clock(monkeypatch):
    now = (
        int(datetime(2026, 11, 3, 12, 0, tzinfo=UTC).timestamp()) * 1_000_000_000
        + 250_000_000
    )
    monkeypatch.setattr(remote, "time_ns", lambda: now)
    assert remote.retry_after_seconds("Tue, 03 Nov 2026 12:00:01 GMT") == 0.75
    assert remote.retry_after_seconds("Tue, 03 Nov 2026 11:59:59 GMT") == 0.0


@pytest.mark.parametrize(
    "header,body,expected",
    [
        ("0.25", b"", 0.25),
        ("-1", b"", 0.0),
        ("broken", b'{"retry_after":1.25}', 1.25),
        (None, b'{"retry_after":0}', 0.0),
        (None, b'{"retry_after":true}', None),
        ("nan", b"", None),
        ("inf", b"", None),
        (None, b'{"retry_after":"inf"}', None),
        (None, b"[]", None),
        (None, b"null", None),
        (None, b"not json", None),
    ],
)
def test_retry_after_protocol_values(header, body, expected):
    assert remote.retry_after_seconds(header, body) == expected


@pytest.mark.parametrize(
    "messages,limit,expected",
    [
        ([], 5, []),
        ([""], 5, []),
        (["abc", "d"], 5, ["abc\nd"]),
        (["abc", "de"], 5, ["abc", "de"]),
        (["abcdefghijk"], 5, ["abcde", "fghij", "k"]),
        (["a", "", "b"], 5, ["a\n\nb"]),
        (["αβγ", "δε"], 5, ["αβγ", "δε"]),
    ],
)
def test_shared_chunk_membership(messages, limit, expected):
    assert remote.chunk_messages(messages, limit) == expected
