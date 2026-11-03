"""Base class for standard logging handlers."""

import asyncio
import queue
import sys
import threading
import traceback
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping

import aiohttp
import msgspec

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


class BaseLogHandler(ABC):
    """Abstract base class for log handlers, defining how log messages

    should be pushed to their respective destinations.
    """

    def __init__(self) -> None:
        """Initialize shared handler resources.

        Sets up lazy JSON encoding, HTTP session, and error callback slots.

        """
        self._json_encode: Callable[[object], bytes] | None = None
        self._http_session: HTTPSession | None = None
        self._limiter: RateLimiter | None = None
        self._on_error: Callable[[BaseException, str], None] | None = None
        self._is_open = False
        self._delivery_queue: queue.Queue[
            tuple[object, Callable[[object], Awaitable[None]]]
        ] = queue.Queue(maxsize=MAX_PENDING_DELIVERIES)
        self._delivery_thread: threading.Thread | None = None
        self._delivery_lock = threading.Lock()
        self._delivery_stop = threading.Event()
        self._delivery_closed = False
        self._delivery_shutdown_deadline: float | None = None
        self._delivery_loop_ref: asyncio.AbstractEventLoop | None = None
        self._delivery_shutdown_event: asyncio.Event | None = None

    @property
    def json_encode(self) -> Callable[[object], bytes]:
        """Lazy JSON encoder.

        Returns:
            Callable[[object], bytes]: msgspec JSON encode function.

        """
        if self._json_encode is None:
            self._json_encode = msgspec.json.Encoder().encode
        return self._json_encode

    def set_error_handler(
        self, handler: Callable[[BaseException, str], None] | None
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

    def open(self) -> None:
        """Called by Logger when handler is attached. Override for setup."""
        with self._delivery_lock:
            if self._delivery_thread is not None and self._delivery_thread.is_alive():
                if self._delivery_closed:
                    raise RuntimeError(
                        "cannot reopen handler while delivery worker is stopping"
                    )
                self._is_open = True
                return
            self._delivery_closed = False
            self._delivery_stop.clear()
            self._delivery_shutdown_deadline = None
        self._is_open = True

    def close(self, timeout_s: float = SHUTDOWN_TIMEOUT_S) -> None:
        """Stop remote delivery after a bounded drain period."""
        now = time_monotonic_ns() / 1e9
        with self._delivery_lock:
            if not self._delivery_closed:
                self._delivery_shutdown_deadline = now + max(0.0, timeout_s)
                self._delivery_closed = True
                self._delivery_stop.set()
            thread = self._delivery_thread
            loop = self._delivery_loop_ref
            shutdown_event = self._delivery_shutdown_event
            deadline = self._delivery_shutdown_deadline or now
        if loop is not None and shutdown_event is not None and loop.is_running():
            try:
                loop.call_soon_threadsafe(shutdown_event.set)
            except RuntimeError:
                pass
        if (
            thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=max(0.0, deadline - (time_monotonic_ns() / 1e9)))
            if thread.is_alive():
                self._handle_exception(
                    TimeoutError("remote delivery did not stop before close deadline"),
                    "close",
                )
        self._is_open = False

    def _enqueue_delivery(
        self,
        item: object,
        deliver: Callable[[object], Awaitable[None]],
    ) -> bool:
        """Queue one remote operation without blocking the logging caller.

        The bounded queue applies backpressure as an observable drop instead of
        allowing a slow endpoint to grow memory without limit.
        """
        error: BaseException | None = None
        with self._delivery_lock:
            if self._delivery_closed:
                error = RuntimeError("handler is closed")
            else:
                if (
                    self._delivery_thread is None
                    or not self._delivery_thread.is_alive()
                ):
                    self._delivery_stop.clear()
                    self._delivery_thread = threading.Thread(
                        target=self._delivery_loop,
                        name=f"{self.__class__.__name__}-delivery",
                        daemon=True,
                    )
                    self._delivery_thread.start()
                try:
                    self._delivery_queue.put_nowait((item, deliver))
                except queue.Full:
                    error = OverflowError(
                        "remote delivery queue full; message was dropped"
                    )

        if error is not None:
            # Error callbacks are user code and may reenter the logger. Never
            # call one while holding the lifecycle lock.
            self._handle_exception(error, "delivery")
            return False
        return True

    async def _consume_deliveries(
        self,
        shutdown_event: asyncio.Event,
    ) -> None:
        """Run queued requests and stop within the one close deadline."""
        while True:
            if self._delivery_stop.is_set() and (time_monotonic_ns() / 1e9) >= (
                self._delivery_shutdown_deadline or 0.0
            ):
                break
            try:
                item, deliver = self._delivery_queue.get_nowait()
            except queue.Empty:
                if self._delivery_stop.is_set():
                    break
                await asyncio.sleep(0.01)
                continue

            delivery_task = asyncio.ensure_future(deliver(item))
            shutdown_waiter = asyncio.create_task(shutdown_event.wait())
            try:
                done, _ = await asyncio.wait(
                    (delivery_task, shutdown_waiter),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if delivery_task in done:
                    await delivery_task
                else:
                    deadline = self._delivery_shutdown_deadline or (
                        time_monotonic_ns() / 1e9
                    )
                    remaining = max(0.0, deadline - (time_monotonic_ns() / 1e9))
                    if remaining:
                        await asyncio.wait_for(delivery_task, timeout=remaining)
                    else:
                        delivery_task.cancel()
                        await asyncio.gather(delivery_task, return_exceptions=True)
            except asyncio.TimeoutError:
                delivery_task.cancel()
                await asyncio.gather(delivery_task, return_exceptions=True)
                self._handle_exception(
                    TimeoutError("remote delivery exceeded handler close deadline"),
                    "close",
                )
            except asyncio.CancelledError:
                if not self._delivery_stop.is_set():
                    self._handle_exception(
                        RuntimeError("delivery task was cancelled"), "delivery"
                    )
            except BaseException as exc:
                self._handle_exception(exc, "remote delivery")
            finally:
                if not shutdown_waiter.done():
                    shutdown_waiter.cancel()
                await asyncio.gather(shutdown_waiter, return_exceptions=True)
                if not delivery_task.done():
                    delivery_task.cancel()
                    await asyncio.gather(delivery_task, return_exceptions=True)
                self._delivery_queue.task_done()

        dropped = 0
        while True:
            try:
                self._delivery_queue.get_nowait()
            except queue.Empty:
                break
            else:
                dropped += 1
                self._delivery_queue.task_done()
        if dropped:
            self._handle_exception(
                OverflowError(
                    f"handler shutdown discarded {dropped} queued remote deliveries"
                ),
                "close",
            )

    def _delivery_loop(self) -> None:
        """Run queued async requests on a handler-owned background thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        shutdown_event = asyncio.Event()
        with self._delivery_lock:
            self._delivery_loop_ref = loop
            self._delivery_shutdown_event = shutdown_event
            if self._delivery_stop.is_set():
                shutdown_event.set()
        try:
            loop.run_until_complete(self._consume_deliveries(shutdown_event))
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            try:
                loop.run_until_complete(
                    asyncio.wait_for(self._close_session(), timeout=CLEANUP_TIMEOUT_S)
                )
            except Exception as exc:
                self._handle_exception(exc, "session close")
            finally:
                loop.close()
                asyncio.set_event_loop(None)
                with self._delivery_lock:
                    self._delivery_loop_ref = None
                    self._delivery_shutdown_event = None

    async def _ensure_session(self) -> HTTPSession:
        """Create and reuse a session exclusively on the delivery worker's loop."""
        if self._http_session is None or self._http_session.closed:
            self._http_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT_S)
            )
        return self._http_session

    async def _close_session(self) -> None:
        """Release the session before its owning event loop is closed."""
        session, self._http_session = self._http_session, None
        if session is not None and not session.closed:
            await session.close()

    _retry_after_seconds = staticmethod(retry_after_seconds)

    async def _post_with_retries(
        self,
        session: HTTPSession,
        url: str,
        *,
        headers: dict[str, str],
        payload: Mapping[str, object],
        service: str,
    ) -> None:
        """Apply the common retry and rate policy on the delivery loop."""
        await post_json_with_retries(
            session,
            url,
            payload,
            headers=headers,
            limiter=self._limiter,
            service=service,
        )

    @abstractmethod
    def push(self, buffer: list[str]) -> None:
        """Flushes the given buffer of log entries in some way.

        Args:
            buffer (list[str]): The list of log messages to push.

        """
        pass
