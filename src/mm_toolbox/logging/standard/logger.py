"""Standard single-process logger implementation."""

import asyncio
import contextlib
import sys
import threading
import traceback
from queue import Empty, Queue

from mm_toolbox.logging.standard.config import LoggerConfig, LogLevel
from mm_toolbox.logging.standard.handlers import BaseLogHandler
from mm_toolbox.time.time import time_iso8601, time_ms


class Logger:
    """A simple asynchronous logger that buffers messages and pushes them to.

    configured handlers at an appropriate time or based on severity.
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
            config (LoggerConfig): Configuration settings for the logger
                (base level, stdout, buffer size, etc.).
            handlers (list[BaseLogHandler], optional): A list of handler
                objects that inherit from BaseLogHandler.

        Raises:
            TypeError: If one of the provided handlers does not inherit from LogHandler.

        """
        self._name = name

        self._config: LoggerConfig = config if config is not None else LoggerConfig()

        self._handlers: list[BaseLogHandler] = handlers if handlers is not None else []

        for handler in self._handlers:
            if not isinstance(handler, BaseLogHandler):
                raise TypeError(
                    "Invalid handler; handler must inherit from BaseLogHandler"
                )

            # Mainly for forwarding the str_format to the handler for
            # formatting log messages where the final point is not a code
            # environment (eg Discord, Telegram, etc).
            handler.add_primary_config(self._config)

        self._buffer_size = 0
        self._buffer: list[str] = [""] * self._config.buffer_size
        self._buffer_start_time_ms = time_ms()

        # Thread-safe queue for cross-context operation (sync or async)
        self._msg_queue: Queue[tuple[str, LogLevel]] = Queue()
        self._is_running = True

        # Start background thread with its own event loop to ingest and flush
        self._thread = threading.Thread(target=self._thread_main, daemon=True)
        self._thread.start()

    async def _flush_buffer(self):
        """Flushes the log message buffer to all handlers."""
        if self._buffer_size == 0:
            return

        if self._config.do_stdout:
            for msg in self._buffer[: self._buffer_size]:
                print(msg)

        # Push concurrently; handlers must not bring down the logger
        payload = self._buffer[: self._buffer_size]
        try:
            tasks = [handler.push(payload) for handler in self._handlers]
        except Exception:
            # In case any handler raises during task creation
            sys.stderr.write(traceback.format_exc())
            tasks = []

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for res in results:
                if isinstance(res, Exception):
                    # Never propagate from logger
                    sys.stderr.write(traceback.format_exc())

        self._buffer_size = 0
        self._buffer_start_time_ms = time_ms()

    def _thread_main(self) -> None:
        """Background thread entry: manages an event loop and flush cadence."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        except Exception:
            # If loop cannot be created, run a degraded sync loop
            loop = None

        async def ingest_once() -> None:
            nonlocal loop
            # Drain as many messages as arrived
            drained = 0
            while True:
                try:
                    log_msg, level = self._msg_queue.get_nowait()
                except Empty:
                    break
                try:
                    if self._buffer_size >= len(self._buffer):
                        # Expand buffer safely if user under-specified buffer_size
                        self._buffer.extend([""] * len(self._buffer))
                    self._buffer[self._buffer_size] = log_msg
                    self._buffer_size += 1
                finally:
                    drained += 1
            # Time-based flush only
            if (time_ms() - self._buffer_start_time_ms) >= int(
                self._config.flush_interval_s * 1000
            ):
                await self._flush_buffer()

        try:
            if loop is None:
                # Fallback: simple polling loop without asyncio handlers support
                while self._is_running:
                    # Drain queue
                    try:
                        while True:
                            log_msg, level = self._msg_queue.get(timeout=0.1)
                            if self._buffer_size >= len(self._buffer):
                                self._buffer.extend([""] * len(self._buffer))
                            self._buffer[self._buffer_size] = log_msg
                            self._buffer_size += 1
                    except Empty:
                        pass
                    # No async handlers possible in this mode; skip flush
                return

            # Normal async loop: run periodic ingestion
            async def runner():
                try:
                    while self._is_running:
                        await ingest_once()
                        await asyncio.sleep(0.1)
                    # Final flush
                    await ingest_once()
                except Exception:
                    sys.stderr.write(traceback.format_exc())

            loop.run_until_complete(runner())
        finally:
            if loop is not None and not loop.is_closed():
                try:
                    loop.stop()
                    loop.close()
                except Exception:
                    pass

    def _process_log(self, level: LogLevel, msg: str):
        """Submits a log message to the queue if it meets the minimum base level.

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
            self._msg_queue.put((log_msg, level))
        except Exception:
            sys.stderr.write(traceback.format_exc())

    def set_log_level(self, level: LogLevel) -> None:
        """Modify the logger's base log level at runtime.

        Args:
            level (LogLevel): The new base log level.

        """
        self.debug(f"Changing base log level from {self._config.base_level} to {level}")
        self._config.base_level = level
        for handler in self._handlers:
            handler.add_primary_config(self._config)

    def trace(self, msg: str) -> None:
        """Send a trace-level log message.

        Args:
            msg (str): The log message text.

        """
        valid_level = self._config.base_level.value == LogLevel.TRACE.value
        if self._is_running and valid_level:
            self._process_log(LogLevel.TRACE, msg)

    def debug(self, msg: str) -> None:
        """Send a debug-level log message.

        Args:
            msg (str): The log message text.

        """
        valid_level = self._config.base_level.value <= LogLevel.DEBUG.value
        if self._is_running and valid_level:
            self._process_log(LogLevel.DEBUG, msg)

    def info(self, msg: str) -> None:
        """Send an info-level log message.

        Args:
            msg (str): The log message text.

        """
        valid_level = self._config.base_level.value <= LogLevel.INFO.value
        if self._is_running and valid_level:
            self._process_log(LogLevel.INFO, msg)

    def warning(self, msg: str) -> None:
        """Send a warning-level log message.

        Args:
            msg (str): The log message text.

        """
        valid_level = self._config.base_level.value <= LogLevel.WARNING.value
        if self._is_running and valid_level:
            self._process_log(LogLevel.WARNING, msg)

    def error(self, msg: str) -> None:
        """Send an error-level log message.

        Args:
            msg (str): The log message text.

        """
        valid_level = self._config.base_level.value <= LogLevel.ERROR.value
        if self._is_running and valid_level:
            self._process_log(LogLevel.ERROR, msg)

    async def shutdown(self):
        """Shuts down the logger, ensuring all buffered messages are flushed.

        and handlers are closed.
        """
        # Signal background to stop and wait for thread exit
        self._is_running = False
        with contextlib.suppress(Exception):
            if self._thread.is_alive():
                self._thread.join(timeout=2.0)

        if self._buffer_size > 0:
            # Flush remaining using the current event loop
            try:
                await self._flush_buffer()
            except Exception:
                sys.stderr.write(traceback.format_exc())
        self._buffer.clear()

        # Close handlers (best-effort)
        for h in self._handlers:
            with contextlib.suppress(Exception):
                await h.aclose()

    def is_running(self) -> bool:
        """Check if the master logger is running."""
        return self._is_running

    def get_name(self) -> str:
        """Get the name of the master logger."""
        return self._name

    def get_config(self) -> LoggerConfig:
        """Get the configuration of the master logger."""
        return self._config
