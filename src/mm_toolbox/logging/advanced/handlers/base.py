"""Base class for advanced logging handlers."""

from __future__ import annotations

import asyncio
import contextlib
import sys
import threading
import traceback
from abc import ABC, abstractmethod
from concurrent.futures import Future, wait
from typing import TYPE_CHECKING
from collections.abc import Mapping

import aiohttp
import msgspec

from mm_toolbox.logging.advanced.config import LoggerConfig
from mm_toolbox.logging.advanced.pylog import PyLog, format_log
from mm_toolbox.logging._remote import (
    HTTPSession,
    CLEANUP_TIMEOUT_S,
    HTTP_TIMEOUT_S,
    MAX_PENDING_DELIVERIES,
    SHUTDOWN_TIMEOUT_S,
    post_json_with_retries,
    retry_after_seconds,
)
from mm_toolbox.rate_limiter import RateLimiter
from mm_toolbox.time import time_monotonic_ns

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine


class BaseLogHandler(ABC):
    """Abstract base class for log handlers.

    All handlers must implement `.push(buffer)`, called from the
    main logger whenever the buffer fills up.

    Validation for any params/args should be done in '__init__'
    to catch config errors early.
    """

    def __init__(self) -> None:
        """Initialize the handler state and shared resources.

        Sets up lazy JSON encoding, a dedicated event loop thread, HTTP
        session management, and error callback tracking.

        """
        self._encode_json: Callable[[object], bytes] | None = None
        self._http_session: HTTPSession | None = None
        self._ev_loop: asyncio.AbstractEventLoop | None = None
        self._ev_loop_lock = threading.Lock()
        self._loop_thread: threading.Thread | None = None
        self._primary_config: LoggerConfig | None = None
        self._futures: set[Future[object]] = set()
        self._futures_lock = threading.Lock()
        self._pending_slots = threading.BoundedSemaphore(MAX_PENDING_DELIVERIES)
        self._closed = False
        self._lifecycle_lock = threading.RLock()
        self._on_error: Callable[[BaseException, str], None] | None = None

    @property
    def encode_json(self) -> Callable[[object], bytes]:
        """Lazily initialize the JSON encoder.

        Returns:
            Callable[[object], bytes]: JSON encoding function.
        """
        if self._encode_json is None:
            self._encode_json = msgspec.json.Encoder().encode
        return self._encode_json

    @property
    def http_session(self) -> HTTPSession | None:
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
    def ev_loop(self) -> asyncio.AbstractEventLoop:
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
                                        timeout=CLEANUP_TIMEOUT_S,
                                    )
                                )
                        session = self._http_session
                        if session is not None and not session.closed:
                            with contextlib.suppress(Exception):
                                loop.run_until_complete(
                                    asyncio.wait_for(
                                        session.close(), timeout=CLEANUP_TIMEOUT_S
                                    )
                                )
                        loop.close()

                self._loop_thread = threading.Thread(target=_runner, daemon=True)
                self._loop_thread.start()
            return self._ev_loop

    def _run_coro(self, coro: "Coroutine[object, None, object]") -> Future[object]:
        """Submit a coroutine to the handler's dedicated event loop.

        Args:
            coro (Coroutine[object, None, object]): Coroutine to run.

        Returns:
            Future: Future tracking the coroutine execution.

        """
        loop = self.ev_loop
        return asyncio.run_coroutine_threadsafe(coro, loop)

    def _submit_coro(
        self, coro: "Coroutine[object, None, object]"
    ) -> Future[object] | None:
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

    def _on_scheduled_done(self, fut: Future[object]) -> None:
        with contextlib.suppress(ValueError):
            self._pending_slots.release()

    async def _ensure_session(self) -> None:
        """Ensure the HTTP session exists and is open.

        Creates a new ``aiohttp.ClientSession`` if none exists or the
        existing one has been closed.

        """
        if self._http_session is None or self._http_session.closed:
            self._http_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT_S)
            )

    @property
    def primary_config(self) -> LoggerConfig | None:
        """Active logger configuration attached to this handler.

        Returns:
            LoggerConfig | None: Active handler config, if set.

        """
        return self._primary_config

    def add_primary_config(self, config: LoggerConfig) -> None:
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

    def format_log(self, log: PyLog) -> str:
        """Format a log message to a string.

        Args:
            log (PyLog): Log entry to format.

        Returns:
            str: Formatted log message.
        """
        if self._primary_config:
            return format_log(log, self._primary_config.str_format)
        else:
            raise RuntimeError(
                f"No primary config found for handler {self.__class__.__name__}"
            )

    @abstractmethod
    def push(self, logs: list[PyLog]) -> None:
        """Push a batch of PyLog messages to the external system.

        Args:
            logs (list[PyLog]): A batch of PyLog objects.

        Returns:
            None: This method does not return a value.
        """
        pass

    def _track_future(self, fut: Future[object]) -> None:
        """Track every in-flight operation until its completion callback runs."""
        with self._futures_lock:
            self._futures.add(fut)
        fut.add_done_callback(self._on_future_done)

    def _on_future_done(self, fut: Future[object]) -> None:
        with self._futures_lock:
            self._futures.discard(fut)
        if not fut.cancelled():
            exc = fut.exception()
            if exc is not None:
                self._handle_exception(exc, "handler task")

    _retry_after_seconds = staticmethod(retry_after_seconds)

    async def _post_json_with_retries(
        self,
        url: str,
        payload: Mapping[str, object],
        *,
        limiter: RateLimiter | None,
        service: str,
    ) -> None:
        """Apply the common retry and rate policy on the handler loop."""
        await self._ensure_session()
        assert self._http_session is not None
        await post_json_with_retries(
            self._http_session,
            url,
            payload,
            headers={"Content-Type": "application/json"},
            limiter=limiter,
            service=service,
        )

    def close(self, timeout_s: float = SHUTDOWN_TIMEOUT_S) -> None:
        """Close HTTP session and stop the handler loop.

        Args:
            timeout_s (float): Max seconds to wait for pending futures.
        """
        deadline = (time_monotonic_ns() / 1e9) + max(0.0, timeout_s)
        with self._lifecycle_lock:
            self._closed = True
            with self._futures_lock:
                pending = [fut for fut in self._futures if not fut.done()]
        if pending:
            _, still_pending = wait(
                pending, timeout=max(0.0, deadline - (time_monotonic_ns() / 1e9))
            )
            for fut in still_pending:
                fut.cancel()
            if still_pending:
                self._handle_exception(
                    TimeoutError(
                        f"handler shutdown cancelled {len(still_pending)} pending remote deliveries"
                    ),
                    "close",
                )

        loop = self._ev_loop
        if loop is not None and not loop.is_closed():
            remaining = max(0.0, deadline - (time_monotonic_ns() / 1e9))
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
            self._loop_thread.join(
                timeout=max(0.0, deadline - (time_monotonic_ns() / 1e9))
            )

    def __del__(self):
        with contextlib.suppress(Exception):
            self.close()
