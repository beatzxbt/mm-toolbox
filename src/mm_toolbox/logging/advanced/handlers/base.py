"""Base class for advanced logging handlers."""

from __future__ import annotations

import asyncio
import contextlib
import email.utils
import json
import sys
import threading
import time
import traceback
from abc import ABC, abstractmethod
from concurrent.futures import Future, wait
from typing import TYPE_CHECKING
from collections.abc import Mapping

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
        """Initialize the rate limiter.

        Args:
            rate_per_sec (float): Tokens replenished per second.
            burst (int): Maximum burst capacity.
        """
        self._rate = max(0.001, float(rate_per_sec))
        self._capacity = max(1, int(burst))
        self._tokens = float(self._capacity)
        self._last: float | None = None
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> None:
        """Wait until the requested number of tokens are available.

        Args:
            tokens (int): Number of tokens to acquire.
        """
        need = max(0, int(tokens))
        if need == 0:
            return
        if need > self._capacity:
            raise ValueError("requested tokens exceed rate limiter burst capacity")

        loop = asyncio.get_running_loop()
        while True:
            async with self._lock:
                now = loop.time()
                if self._last is None:
                    self._last = now
                else:
                    elapsed = max(0.0, now - self._last)
                    self._tokens = min(
                        self._capacity, self._tokens + elapsed * self._rate
                    )
                    self._last = now
                if self._tokens >= need:
                    self._tokens -= need
                    return
                wait_s = (need - self._tokens) / self._rate
            # Wait outside the lock; every waiter rechecks and reserves under
            # the lock before it can proceed.
            await asyncio.sleep(wait_s)


class BaseLogHandler(ABC):
    """Abstract base class for log handlers.

    All handlers must implement `.push(buffer)`, called from the
    main logger whenever the buffer fills up.

    Validation for any params/args should be done in '__init__'
    to catch config errors early.
    """

    def __init__(self):
        """Initialize the handler state and shared resources.

        Sets up lazy JSON encoding, a dedicated event loop thread, HTTP
        session management, and error callback tracking.

        """
        self._encode_json: Callable[[object], bytes] | None = None
        self._http_session: aiohttp.ClientSession | None = None
        self._ev_loop: asyncio.AbstractEventLoop | None = None
        self._ev_loop_lock = threading.Lock()
        self._loop_thread: threading.Thread | None = None
        self._primary_config: LoggerConfig | None = None
        self._futures: list[Future] = []
        self._futures_lock = threading.Lock()
        self._pending_slots = threading.BoundedSemaphore(256)
        self._closed = False
        self._lifecycle_lock = threading.RLock()
        self._on_error: Callable[[BaseException, str], None] | None = None

    @property
    def encode_json(self):
        """Lazily initialize the JSON encoder.

        Returns:
            Callable[[object], bytes]: JSON encoding function.
        """
        if self._encode_json is None:
            self._encode_json = msgspec.json.Encoder().encode
        return self._encode_json

    @property
    def http_session(self):
        """Lazily initialize the HTTP session.

        Note: Creation and usage happen on the handler's loop thread.

        Returns:
            aiohttp.ClientSession | None: Lazy session reference.
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
        """Lazily create a dedicated event loop thread for the handler.

        Returns:
            asyncio.AbstractEventLoop: Event loop used by this handler.
        """
        with self._ev_loop_lock:
            if self._ev_loop is None or self._ev_loop.is_closed():
                # Python 3.14: Manual loop creation is required because
                # this runs in a dedicated background thread for long-lived
                # async I/O, not the main thread where asyncio.run() is used.
                loop = asyncio.new_event_loop()
                self._ev_loop = loop

                def _runner() -> None:
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_forever()
                    finally:
                        pending = asyncio.all_tasks(loop)
                        for task in pending:
                            task.cancel()
                        if pending:
                            with contextlib.suppress(Exception):
                                loop.run_until_complete(
                                    asyncio.wait_for(
                                        asyncio.gather(
                                            *pending, return_exceptions=True
                                        ),
                                        timeout=0.25,
                                    )
                                )
                        session = self._http_session
                        if session is not None and not session.closed:
                            with contextlib.suppress(Exception):
                                loop.run_until_complete(
                                    asyncio.wait_for(session.close(), timeout=0.25)
                                )
                        loop.close()

                self._loop_thread = threading.Thread(target=_runner, daemon=True)
                self._loop_thread.start()
            return self._ev_loop

    def _run_coro(self, coro: "Coroutine[object, None, object]") -> Future:
        """Submit a coroutine to the handler's dedicated event loop.

        Args:
            coro (Coroutine[object, None, object]): Coroutine to run.

        Returns:
            Future: Future tracking the coroutine execution.

        """
        loop = self.ev_loop
        return asyncio.run_coroutine_threadsafe(coro, loop)

    def _submit_coro(self, coro: "Coroutine[object, None, object]") -> Future | None:
        """Submit remote work only while an in-flight slot is available."""
        with self._lifecycle_lock:
            if self._closed or not self._pending_slots.acquire(blocking=False):
                coro.close()
                return None
            try:
                future = self._run_coro(coro)
            except BaseException:
                self._pending_slots.release()
                coro.close()
                raise
            future.add_done_callback(self._on_scheduled_done)
            self._track_future(future)
            return future

    def _on_scheduled_done(self, fut: Future) -> None:
        with contextlib.suppress(ValueError):
            self._pending_slots.release()

    async def _ensure_session(self) -> None:
        """Ensure the HTTP session exists and is open.

        Creates a new ``aiohttp.ClientSession`` if none exists or the
        existing one has been closed.

        """
        if self._http_session is None or self._http_session.closed:
            self._http_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=3.0)
            )

    @property
    def primary_config(self):
        """Active logger configuration attached to this handler.

        Returns:
            LoggerConfig | None: Active handler config, if set.

        """
        return self._primary_config

    def add_primary_config(self, config: LoggerConfig):
        """Attach a logger configuration to this handler.

        Args:
            config (LoggerConfig): Configuration to apply to the handler.

        """
        self._primary_config = config

    def set_error_handler(
        self, handler: "Callable[[BaseException, str], None] | None"
    ) -> None:
        """Set a handler-specific exception callback.

        Args:
            handler (Callable[[BaseException, str], None] | None): Callable invoked
                with (exception, context). Use None to reset.
        """
        self._on_error = handler

    def _handle_exception(self, exc: BaseException, context: str) -> None:
        """Handle handler exceptions with optional custom callback.

        Args:
            exc (BaseException): The exception raised by handler work.
            context (str): Short label describing where the error occurred.

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

    @staticmethod
    def _normalize_log_bytes(field: object) -> bytes:
        """Normalize bytes-like log fields to immutable bytes."""
        if isinstance(field, bytes):
            return field
        if isinstance(field, memoryview):
            return field.tobytes()
        if isinstance(field, bytearray):
            return bytes(field)
        raise TypeError(
            f"Invalid log field type; expected bytes-like but got {type(field)}"
        )

    def format_log(self, log: PyLog) -> str:
        """Format a log message to a string.

        Args:
            log (PyLog): Log entry to format.

        Returns:
            str: Formatted log message.
        """
        if self._primary_config:
            name = self._normalize_log_bytes(log.name).decode()
            message = self._normalize_log_bytes(log.message).decode()
            formatted_str = self._primary_config.str_format % {
                "asctime": time_iso8601(float(log.timestamp_ns) / 1_000_000_000.0),
                "levelname": log.level.name,
                "name": name,
                "message": message,
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
            logs (list[PyLog]): A batch of PyLog objects.

        Returns:
            None: This method does not return a value.
        """
        pass

    def _track_future(self, fut: Future) -> None:
        """Track a handler future and capture completion errors.

        Args:
            fut (Future): Future returned by run_coroutine_threadsafe.

        """
        with self._futures_lock:
            self._futures.append(fut)
        fut.add_done_callback(self._on_future_done)
        # Trim to avoid unbounded growth
        with self._futures_lock:
            if len(self._futures) > 4096:
                self._futures = self._futures[-2048:]

    def _on_future_done(self, fut: Future) -> None:
        """Capture exceptions from background handler tasks.

        Args:
            fut (Future): Future returned by run_coroutine_threadsafe.

        """
        with contextlib.suppress(Exception):
            exc = fut.exception()
            if exc is not None:
                self._handle_exception(exc, "handler task")

    @staticmethod
    def _retry_after_seconds(value: str | None, body: bytes = b"") -> float | None:
        """Parse numeric or HTTP-date Retry-After values without shortening them."""
        if value:
            try:
                return max(0.0, float(value))
            except ValueError:
                try:
                    retry_at = email.utils.parsedate_to_datetime(value).timestamp()
                    return max(0.0, retry_at - time.time())
                except (TypeError, ValueError, OverflowError):
                    pass
        try:
            payload = json.loads(body)
            return max(0.0, float(payload.get("retry_after")))
        except (TypeError, ValueError, AttributeError, json.JSONDecodeError):
            return None

    async def _post_json_with_retries(
        self,
        url: str,
        payload: Mapping[str, object],
        *,
        limiter: _RateLimiter | None,
        service: str,
    ) -> None:
        """POST JSON with status validation and bounded transient retries."""
        await self._ensure_session()
        assert self._http_session is not None
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 10.0
        for attempt in range(3):
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError(f"{service} delivery exceeded its 10-second budget")
            try:
                async with asyncio.timeout(remaining):
                    if limiter is not None:
                        await limiter.acquire(1)
                    async with self._http_session.post(
                        url,
                        headers={"Content-Type": "application/json"},
                        data=self.encode_json(payload),
                    ) as response:
                        if 200 <= response.status < 300:
                            return
                        body = await response.read()
                        error = RuntimeError(f"{service} returned {response.status}")
                        retryable = response.status == 429 or response.status >= 500
                        retry_after = (
                            self._retry_after_seconds(
                                response.headers.get("Retry-After"), body
                            )
                            if response.status == 429
                            else None
                        )
            except Exception as exc:
                error = exc
                retryable = True
                retry_after = None
            if not retryable or attempt == 2:
                raise error
            remaining = deadline - loop.time()
            delay = (
                retry_after if retry_after is not None else min(0.1 * (2**attempt), 0.4)
            )
            if delay >= remaining:
                if retry_after is not None:
                    raise RuntimeError(
                        f"{service} Retry-After of {delay:.3f}s exceeds the remaining delivery budget"
                    ) from error
                raise TimeoutError(
                    f"{service} delivery exceeded its 10-second budget"
                ) from error
            await asyncio.sleep(delay)

    def close(self, timeout_s: float = 2.0) -> None:
        """Close HTTP session and stop the handler loop.

        Args:
            timeout_s (float): Max seconds to wait for pending futures.
        """
        deadline = time.monotonic() + max(0.0, timeout_s)
        with self._lifecycle_lock:
            self._closed = True
            with self._futures_lock:
                pending = [fut for fut in self._futures if not fut.done()]
        if pending:
            _, still_pending = wait(
                pending, timeout=max(0.0, deadline - time.monotonic())
            )
            for fut in still_pending:
                fut.cancel()

        loop = self._ev_loop
        if loop is not None and not loop.is_closed():
            remaining = max(0.0, deadline - time.monotonic())
            if (
                remaining > 0
                and loop.is_running()
                and self._http_session is not None
                and not self._http_session.closed
            ):
                try:
                    fut = asyncio.run_coroutine_threadsafe(
                        self._http_session.close(), loop
                    )
                    fut.result(timeout=remaining)
                except Exception as exc:
                    self._handle_exception(exc, "handler close")
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(loop.stop)
        if (
            self._loop_thread
            and self._loop_thread.is_alive()
            and self._loop_thread is not threading.current_thread()
        ):
            self._loop_thread.join(timeout=max(0.0, deadline - time.monotonic()))

    def __del__(self):
        with contextlib.suppress(Exception):
            self.close()
