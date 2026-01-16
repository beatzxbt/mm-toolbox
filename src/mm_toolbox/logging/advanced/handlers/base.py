"""Base class for advanced logging handlers."""

from __future__ import annotations

import asyncio
import contextlib
import sys
import threading
import traceback
from abc import ABC, abstractmethod
from concurrent.futures import Future
from typing import TYPE_CHECKING

import aiohttp
import msgspec

from mm_toolbox.logging.advanced.config import LoggerConfig
from mm_toolbox.logging.advanced.pylog import PyLog
from mm_toolbox.time import time_iso8601

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine


class _RateLimiter:
    """Simple token bucket for asyncio contexts."""

    def __init__(self, rate_per_sec: float, burst: int) -> None:
        self._rate = max(0.001, float(rate_per_sec))
        self._capacity = max(1, int(burst))
        self._tokens = float(self._capacity)
        self._last: float | None = None

    async def acquire(self, tokens: float = 1.0) -> None:
        loop = asyncio.get_running_loop()
        now = loop.time()
        if self._last is None:
            self._last = now
        else:
            elapsed = max(0.0, now - self._last)
            self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
            self._last = now
        need = max(0.0, tokens)
        if self._tokens >= need:
            self._tokens -= need
            return
        wait_s = (need - self._tokens) / self._rate
        await asyncio.sleep(wait_s)
        # After sleep, consume
        now2 = loop.time()
        elapsed2 = max(0.0, now2 - self._last)
        self._tokens = min(self._capacity, self._tokens + elapsed2 * self._rate)
        self._last = now2
        self._tokens = max(0.0, self._tokens - need)


class BaseLogHandler(ABC):
    """Abstract base class for log handlers.

    All handlers must implement `.push(buffer)`, called from the
    main logger whenever the buffer fills up.

    Validation for any params/args should be done in '__init__'
    to catch config errors early.
    """

    def __init__(self):
        self._encode_json: Callable[[object], bytes] | None = None
        self._http_session: aiohttp.ClientSession | None = None
        self._ev_loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._primary_config: LoggerConfig | None = None
        self._futures: list[Future] = []
        self._on_error: Callable[[BaseException, str], None] | None = None

    @property
    def encode_json(self):
        """Lazily initialize the JSON encoder."""
        if self._encode_json is None:
            self._encode_json = msgspec.json.Encoder().encode
        return self._encode_json

    @property
    def http_session(self):
        """Lazily initialize the HTTP session.

        Note: Creation and usage happen on the handler's loop thread.
        """
        # Keep for backward-compat access but prefer _ensure_session in async code
        _ = self.ev_loop
        if self._http_session is None:
            # Create synchronously by delegating to the loop
            fut = self._run_coro(self._ensure_session())
            with contextlib.suppress(Exception):
                fut.result(timeout=2.0)
        return self._http_session

    @property
    def ev_loop(self):
        """Lazily create a dedicated event loop thread for the handler."""
        if self._ev_loop is None or self._ev_loop.is_closed():
            loop = asyncio.new_event_loop()
            self._ev_loop = loop

            def _runner() -> None:
                asyncio.set_event_loop(loop)
                loop.run_forever()

            self._loop_thread = threading.Thread(target=_runner, daemon=True)
            self._loop_thread.start()
        return self._ev_loop

    def _run_coro(self, coro: "Coroutine[object, None, object]") -> Future:
        loop = self.ev_loop
        return asyncio.run_coroutine_threadsafe(coro, loop)

    async def _ensure_session(self) -> None:
        if self._http_session is None or self._http_session.closed:
            self._http_session = aiohttp.ClientSession()

    @property
    def primary_config(self):
        """Get the primary config."""
        return self._primary_config

    def add_primary_config(self, config: LoggerConfig):
        """Add the primary config to the handler."""
        self._primary_config = config

    def set_error_handler(
        self, handler: "Callable[[BaseException, str], None] | None"
    ) -> None:
        """Set a handler-specific exception callback.

        Args:
            handler: Callable invoked with (exception, context). Use None to reset.
        """
        self._on_error = handler

    def _handle_exception(self, exc: BaseException, context: str) -> None:
        """Handle handler exceptions with optional custom callback.

        Args:
            exc: The exception raised by handler work.
            context: Short label describing where the error occurred.

        """
        if self._on_error is not None:
            try:
                self._on_error(exc, context)
                return
            except Exception:
                pass
        sys.stderr.write(f"[{self.__class__.__name__}] {context}: {exc}\n")
        sys.stderr.write(
            "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        )

    def format_log(self, log: PyLog) -> str:
        """Format a log message to a string."""
        if self._primary_config:
            formatted_str = self._primary_config.str_format % {
                "asctime": time_iso8601(float(log.timestamp_ns) / 1_000_000_000.0),
                "levelname": log.level.name,
                "name": log.name.decode(),
                "message": log.message.decode(),
            }
            return formatted_str
        else:
            raise RuntimeError(
                f"No primary config found for handler {self.__class__.__name__}"
            )

    @abstractmethod
    def push(self, logs: list[PyLog]):
        """Push a batch of PyLog messages to the external system.

        Args:
            logs: A batch of PyLog objects.
        """
        pass

    def _track_future(self, fut: Future) -> None:
        self._futures.append(fut)
        fut.add_done_callback(self._on_future_done)
        # Trim to avoid unbounded growth
        if len(self._futures) > 4096:
            self._futures = self._futures[-2048:]

    def _on_future_done(self, fut: Future) -> None:
        """Capture exceptions from background handler tasks.

        Args:
            fut: Future returned by run_coroutine_threadsafe.

        """
        with contextlib.suppress(Exception):
            exc = fut.exception()
            if exc is not None:
                self._handle_exception(exc, "handler task")

    def close(self, timeout_s: float = 2.0) -> None:
        """Close HTTP session and stop the handler loop."""
        # Wait for pending futures briefly
        if self._futures:
            remaining = max(0.0, timeout_s)
            per = min(0.25, remaining) if remaining > 0 else 0.0
            for fut in list(self._futures):
                try:
                    fut.result(timeout=per)
                except Exception as exc:
                    self._handle_exception(exc, "handler close")
        # Close session on the loop
        if (
            self._ev_loop
            and not self._ev_loop.is_closed()
            and self._http_session is not None
        ):
            with contextlib.suppress(Exception):
                fut = self._run_coro(self._http_session.close())
                with contextlib.suppress(Exception):
                    fut.result(timeout=timeout_s)
        # Stop loop and join thread
        if self._ev_loop and not self._ev_loop.is_closed():
            with contextlib.suppress(Exception):
                self._ev_loop.call_soon_threadsafe(self._ev_loop.stop)
        if self._loop_thread and self._loop_thread.is_alive():
            with contextlib.suppress(Exception):
                self._loop_thread.join(timeout=timeout_s)

    def __del__(self):
        with contextlib.suppress(Exception):
            self.close()
