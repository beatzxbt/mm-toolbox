"""Shared contract tests for remote log delivery behavior."""

import asyncio

import pytest

from mm_toolbox.logging.advanced.handlers.base import BaseLogHandler as AdvancedBase
from mm_toolbox.logging.standard.handlers.base import BaseLogHandler as StandardBase


class _Standard(StandardBase):
    def push(self, buffer):
        pass

class _Advanced(AdvancedBase):
    def push(self, logs):
        pass

class _Response:
    status = 429
    headers = {"Retry-After": "60"}

    async def read(self):
        return b""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

class _Session:
    closed = False

    def __init__(self):
        self.calls = 0

    def post(self, *args, **kwargs):
        self.calls += 1
        return _Response()

    async def close(self):
        self.closed = True

@pytest.mark.parametrize("handler_type", [_Standard, _Advanced])
@pytest.mark.asyncio
async def test_long_retry_after_fails_without_early_retry(handler_type):
    handler = handler_type()
    session = _Session()
    assert handler._retry_after_seconds("60") == 60
    assert handler._retry_after_seconds(None, b'{"retry_after": 60}') == 60
    if isinstance(handler, _Standard):
        operation = handler._post_with_retries(
            session, "https://unused", headers={}, payload={}, service="test"
        )
    else:
        handler._http_session = session
        operation = handler._post_json_with_retries(
            "https://unused", {}, limiter=None, service="test"
        )
    with pytest.raises(RuntimeError, match="exceeds the remaining delivery budget"):
        await operation
    assert session.calls == 1
    if isinstance(handler, _Advanced):
        # This fake session was used on this test's loop, not an owned thread.
        await session.close()
        handler._http_session = None
    handler.close()
