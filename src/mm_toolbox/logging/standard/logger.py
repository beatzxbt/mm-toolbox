"""Standard single-process logger implementation."""

import atexit
from collections.abc import Sequence
import sys
import traceback
from types import TracebackType
from typing import Self

from mm_toolbox.logging.standard.config import (
    LoggerConfig,
    LogLevel,
    TRACE_LEVEL,
    DEBUG_LEVEL,
    INFO_LEVEL,
    WARNING_LEVEL,
    ERROR_LEVEL,
)
from mm_toolbox.logging.standard.handlers import BaseLogHandler
from mm_toolbox.time.time import time_iso8601, time_monotonic_ns


class Logger:
    """A synchronous logger that buffers messages and flushes on demand.

    Interval flushing is opportunistic: elapsed time is checked only while
    processing a new log call. Call :meth:`flush` or :meth:`shutdown` to flush
    a quiet logger's pending messages explicitly.
    """

    def __init__(
        self,
        name: str = "",
        config: LoggerConfig | None = None,
        handlers: Sequence[BaseLogHandler] | None = None,
    ) -> None:
        """Initializes a Logger with specified configuration and handlers.

        Args:
            name (str): Name of the logger. Defaults to an empty string.
            config (LoggerConfig): Configuration settings for the logger.
            handlers (list[BaseLogHandler], optional): A list of handler
                objects that inherit from BaseLogHandler.

        Raises:
            TypeError: If one of the provided handlers does not inherit from
                BaseLogHandler.

        """
        self._name = name
        self._config: LoggerConfig = config if config is not None else LoggerConfig()
        self._handlers: list[BaseLogHandler] = (
            list(handlers) if handlers is not None else []
        )
        self._is_running = True
        self._buffer: list[str] = []
        self._buffer_start_time_ns = time_monotonic_ns()

        for handler in self._handlers:
            if not isinstance(handler, BaseLogHandler):
                raise TypeError(
                    "Invalid handler; handler must inherit from BaseLogHandler"
                )
            handler.open()

        atexit.register(self.shutdown)

    def _should_flush(self) -> bool:
        """Determine whether the internal buffer meets flush criteria.

        Evaluates size and interval thresholds configured on the logger.

        Returns:
            bool: True if the buffer should be flushed now.

        """
        if (
            self._config.flush_on_size
            and len(self._buffer) >= self._config.flush_size_threshold
        ):
            return True
        if self._config.flush_on_interval:
            elapsed = time_monotonic_ns() - self._buffer_start_time_ns
            if elapsed >= self._config.flush_interval_s * 1e9:
                return True
        return False

    def flush(self) -> None:
        """Flushes the log message buffer to all handlers."""
        if not self._buffer:
            return

        payload = self._buffer
        self._buffer = []
        self._buffer_start_time_ns = time_monotonic_ns()

        if self._config.do_stdout:
            try:
                out = sys.stdout
                for msg in payload:
                    out.write(msg + "\n")
                out.flush()
            except Exception:
                sys.stderr.write(traceback.format_exc())

        for handler in self._handlers:
            try:
                handler.push(payload)
            except Exception as exc:
                handler._handle_exception(exc, "push")

    def _process_log(self, level_name: str, msg: str) -> None:
        """Submits a log message to the buffer if it meets the minimum base level.

        Args:
            level_name (str): The severity label for the message.
            msg (str): The actual log message.

        """
        try:
            log_msg = self._config.str_format % {
                "asctime": time_iso8601(),
                "name": self._name,
                "levelname": level_name,
                "message": msg,
            }
            self._buffer.append(log_msg)

            if self._should_flush():
                self.flush()
        except Exception:
            sys.stderr.write(traceback.format_exc())

    def shutdown(self) -> None:
        """Flush once and close handlers, including their bounded remote drain."""
        if not self._is_running:
            return
        self._is_running = False
        try:
            self.flush()
        finally:
            for handler in self._handlers:
                try:
                    handler.close()
                except Exception as exc:
                    handler._handle_exception(exc, "close")
            atexit.unregister(self.shutdown)

    def trace(self, msg: str) -> None:
        """Send a trace-level log message.

        Args:
            msg (str): The log message text.

        """
        if self._is_running and self._config._base_level_value <= TRACE_LEVEL:
            self._process_log("TRACE", msg)

    def debug(self, msg: str) -> None:
        """Send a debug-level log message.

        Args:
            msg (str): The log message text.

        """
        if self._is_running and self._config._base_level_value <= DEBUG_LEVEL:
            self._process_log("DEBUG", msg)

    def info(self, msg: str) -> None:
        """Send an info-level log message.

        Args:
            msg (str): The log message text.

        """
        if self._is_running and self._config._base_level_value <= INFO_LEVEL:
            self._process_log("INFO", msg)

    def warning(self, msg: str) -> None:
        """Send a warning-level log message.

        Args:
            msg (str): The log message text.

        """
        if self._is_running and self._config._base_level_value <= WARNING_LEVEL:
            self._process_log("WARNING", msg)

    def error(self, msg: str) -> None:
        """Send an error-level log message.

        Args:
            msg (str): The log message text.

        """
        if self._is_running and self._config._base_level_value <= ERROR_LEVEL:
            self._process_log("ERROR", msg)

    def set_log_level(self, level: LogLevel) -> None:
        """Change the minimum log level at runtime.

        Args:
            level (LogLevel): New base log level.

        """
        self.debug(f"Changing base log level from {self._config.base_level} to {level}")
        self._config.set_base_level(level)

    def is_running(self) -> bool:
        """Return whether the logger is still active.

        Returns:
            bool: True if the logger has not been shut down.

        """
        return self._is_running

    def get_name(self) -> str:
        """Return the logger's name.

        Returns:
            str: Logger name provided at initialization.

        """
        return self._name

    def get_config(self) -> LoggerConfig:
        """Return the active logger configuration.

        Returns:
            LoggerConfig: Current configuration instance.

        """
        return self._config

    def __enter__(self) -> Self:
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Context manager exit — ensures shutdown."""
        self.shutdown()
