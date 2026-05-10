"""Base class for standard logging handlers."""

import asyncio
import sys
import traceback
from abc import ABC, abstractmethod
from typing import Callable

import aiohttp
import msgspec


class BaseLogHandler(ABC):
    """Abstract base class for log handlers, defining how log messages

    should be pushed to their respective destinations.
    """

    def __init__(self):
        """Initialize shared handler resources.

        Sets up lazy JSON encoding, HTTP session, and error callback slots.

        """        
        self._json_encode = None
        self._http_session = None
        self._on_error: Callable[[BaseException, str], None] | None = None
        self._is_open = False

    @property
    def json_encode(self):
        """Lazy JSON encoder.

        Returns:
            Callable[[object], bytes]: msgspec JSON encode function.

        """
        if self._json_encode is None:
            self._json_encode = msgspec.json.Encoder().encode
        return self._json_encode

    @property
    def http_session(self):
        """Lazy aiohttp client session.

        Returns:
            aiohttp.ClientSession: Reusable HTTP session.

        """
        if self._http_session is None:
            self._http_session = aiohttp.ClientSession()
        return self._http_session

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
        self._is_open = True

    def close(self) -> None:
        """Close any async resources owned by the handler."""
        if self._http_session is not None and not self._http_session.closed:
            try:
                loop = asyncio.new_event_loop()
                try:
                    loop.run_until_complete(self._http_session.close())
                finally:
                    loop.close()
            except Exception:
                pass
        self._is_open = False

    @abstractmethod
    def push(self, buffer: list[str]) -> None:
        """Flushes the given buffer of log entries in some way.

        Args:
            buffer (list[str]): The list of log messages to push.

        """
        pass
