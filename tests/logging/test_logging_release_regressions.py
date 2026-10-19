"""Fault tests for bounded remote delivery and cooperative shutdown."""

import asyncio
import threading
import time

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


def test_standard_immediate_close_drains_every_queued_delivery():
    handler = _Standard()
    delivered = []

    async def deliver(item):
        delivered.append(item)

    for item in range(20):
        assert handler._enqueue_delivery(item, deliver)
    handler.close()
    assert delivered == list(range(20))
    assert not handler._delivery_thread.is_alive()


def test_standard_close_has_one_deadline_and_reports_queued_loss():
    handler = _Standard()
    handler._delivery_close_timeout_s = 0.1
    errors = []
    entered = threading.Event()
    cancelled = threading.Event()
    handler.set_error_handler(lambda error, context: errors.append(error))

    async def never(item):
        entered.set()
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    for item in range(20):
        handler._enqueue_delivery(item, never)
    assert entered.wait(timeout=1)
    start = time.monotonic()
    handler.close()
    assert time.monotonic() - start < 0.5
    handler._delivery_thread.join(timeout=0.5)
    assert not handler._delivery_thread.is_alive()
    assert errors
    assert any("discarded" in str(error) for error in errors)
    assert handler._delivery_queue.empty()
    assert cancelled.is_set()


def test_advanced_close_bounds_total_wait_across_many_futures():
    handler = _Advanced()

    async def never():
        await asyncio.Future()

    futures = [handler._submit_coro(never()) for _ in range(20)]
    start = time.monotonic()
    handler.close(timeout_s=0.1)
    assert time.monotonic() - start < 0.5
    handler._loop_thread.join(timeout=0.5)
    assert not handler._loop_thread.is_alive()
    assert all(future.done() for future in futures)
    assert handler._submit_coro(never()) is None
