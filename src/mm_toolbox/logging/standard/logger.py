"""Standard single-process logger implementation."""

import atexit
import sys
import traceback

from mm_toolbox.logging.standard.config import LoggerConfig, LogLevel
from mm_toolbox.logging.standard.handlers import BaseLogHandler
from mm_toolbox.time.time import time_iso8601, time_ms


class Logger:
    """A simple synchronous single-threaded logger that buffers messages and

    pushes them to configured handlers when flush conditions are met.
    """

    def __init__(
        self,
        name: str = "",
        config: LoggerConfig | None = None,
        handlers: list[BaseLogHandler] | None = None,
    ):
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
        self._handlers: list[BaseLogHandler] = handlers if handlers is not None else []
        self._is_running = True
        self._buffer: list[str] = []
        self._buffer_start_time_ms = time_ms()

        for handler in self._handlers:
            if not isinstance(handler, BaseLogHandler):
                raise TypeError(
                    "Invalid handler; handler must inherit from BaseLogHandler"
                )
            handler.open()

        if self._config.auto_flush_on_exit:
            atexit.register(self._atexit_flush)

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
            elapsed = time_ms() - self._buffer_start_time_ms
            if elapsed >= int(self._config.flush_interval_s * 1000):
                return True
        return False

    def _flush(self) -> None:
        """Flushes the log message buffer to all handlers."""
        if not self._buffer:
            return

        payload = self._buffer
        self._buffer = []
        self._buffer_start_time_ms = time_ms()

        if self._config.do_stdout:
            out = sys.stdout
            for msg in payload:
                out.write(msg + "\n")
            out.flush()

        for handler in self._handlers:
            try:
                handler.push(payload)
            except Exception as exc:
                handler._handle_exception(exc, "push")

    def _process_log(self, level: LogLevel, msg: str) -> None:
        """Submits a log message to the buffer if it meets the minimum base level.

        Args:
            level (LogLevel): The severity level of the message.
            msg (str): The actual log message.

        """
        try:
            log_msg = self._config.str_format % {
                "asctime": time_iso8601(),
                "name": self._name,
                "levelname": level.name,
                "message": msg,
            }
            self._buffer.append(log_msg)

            if self._should_flush():
                self._flush()
        except Exception:
            sys.stderr.write(traceback.format_exc())

    def flush(self) -> None:
        """Flush the current buffer to all handlers immediately."""
        self._flush()

    def shutdown(self) -> None:
        """Shut down the logger and release all resources.

        Flushes any remaining buffered messages and closes attached handlers.

        """
        self._is_running = False
        self._flush()

        for handler in self._handlers:
            try:
                handler.close()
            except Exception:
                pass

    def _atexit_flush(self) -> None:
        """Called by atexit to flush remaining logs on interpreter shutdown.

        Does not close handlers — let the OS clean them up.
        """
        if self._buffer:
            self._flush()

    def trace(self, msg: str) -> None:
        """Send a trace-level log message.

        Args:
            msg (str): The log message text.

        """
        if self._is_running and self._config.base_level.value <= LogLevel.TRACE.value:
            self._process_log(LogLevel.TRACE, msg)

    def debug(self, msg: str) -> None:
        """Send a debug-level log message.

        Args:
            msg (str): The log message text.

        """
        if self._is_running and self._config.base_level.value <= LogLevel.DEBUG.value:
            self._process_log(LogLevel.DEBUG, msg)

    def info(self, msg: str) -> None:
        """Send an info-level log message.

        Args:
            msg (str): The log message text.

        """
        if self._is_running and self._config.base_level.value <= LogLevel.INFO.value:
            self._process_log(LogLevel.INFO, msg)

    def warning(self, msg: str) -> None:
        """Send a warning-level log message.

        Args:
            msg (str): The log message text.

        """
        if self._is_running and self._config.base_level.value <= LogLevel.WARNING.value:
            self._process_log(LogLevel.WARNING, msg)

    def error(self, msg: str) -> None:
        """Send an error-level log message.

        Args:
            msg (str): The log message text.

        """
        if self._is_running and self._config.base_level.value <= LogLevel.ERROR.value:
            self._process_log(LogLevel.ERROR, msg)

    def set_log_level(self, level: LogLevel) -> None:
        """Change the minimum log level at runtime.

        Args:
            level (LogLevel): New base log level.

        """
        self.debug(f"Changing base log level from {self._config.base_level} to {level}")
        self._config.base_level = level

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

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit — ensures shutdown."""
        self.shutdown()
