import sys
import traceback
import asyncio
from typing import Optional
from mm_toolbox.time import time_iso8601, time_s

from .config import LogLevel, LoggerConfig
from .handlers import LogHandler

class Logger:
    """
    A simple asynchronous logger that buffers messages and pushes them to
    configured handlers at an appropriate time or based on severity.
    """

    def __init__(
        self,
        config: LoggerConfig = LoggerConfig(),
        handlers: Optional[list[LogHandler]] = None
    ):
        """
        Initializes a Logger with specified configuration and handlers.

        Args:
            config (LoggerConfig): Configuration settings for the logger (base level, stdout, buffer size, etc.).
            handlers (list[LogHandler], optional): A list of handler objects that inherit from LogHandler. 
                Defaults to an empty list if not provided.

        Raises:
            TypeError: If one of the provided handlers does not inherit from LogHandler.
        """
        if handlers is None:
            handlers = []

        self._config = config
        self._handlers = handlers

        for handler in self._handlers:
            handler_base_class = handler.__class__.__base__
            if not handler_base_class == LogHandler:
                raise TypeError(f"Invalid handler base class; expected LogHandler but got {handler_base_class}")

        self._buffer_size = 0
        self._buffer: list[str] = [None] * self._config.buffer_capacity
        self._buffer_start_time = time_s()

        self._msg_queue = asyncio.Queue()
        self._ev_loop = asyncio.get_event_loop()
        self._shutdown_flag = False

        # Start the log ingestor task.
        self._ev_loop.create_task(self._log_ingestor())

    async def _flush_buffer(self):
        """
        Flushes the log message buffer to all handlers.
        """
        for handler in self._handlers:
            await handler.push(self._buffer)
        self._buffer_size = 0
        self._buffer_start_time = time_s()

    async def _log_ingestor(self):
        """
        Asynchronous loop that ingests log messages from the queue and flushes
        them based on severity or buffer fullness.

        Raises:
            Exception: If an error occurs during the log writing process.
        """
        while not self._shutdown_flag or not self._msg_queue.empty():
            try:
                # (log_msg: str, level: LogLevel)
                log_msg, level = await self._msg_queue.get()

                self._buffer[self._buffer_size] = log_msg
                self._buffer_size += 1

                if self._config.stout:
                    print(log_msg)

                # Immediate buffer flush for ERROR or higher.
                if level.value >= LogLevel.ERROR.value:
                    await self._flush_buffer()
                else:
                    # For lower severity, flush if buffer is full or timed out.
                    is_buffer_full = self._buffer_size >= self._config.buffer_capacity
                    is_buffer_expired = (time_s() - self._buffer_start_time) >= self._config.buffer_timeout_s

                    if is_buffer_full or is_buffer_expired:
                        await self._flush_buffer()

                self._msg_queue.task_done()

            except Exception:
                traceback.print_exc(file=sys.stderr)

    def _submit_log(self, level: LogLevel, msg: str):
        """
        Submits a log message to the queue if it meets the minimum base level.

        Args:
            level (LogLevel): The severity level of the message.
            msg (str): The actual log message.
        """
        try:
            if level.value >= self._config.base_level.value:
                log_msg = f"{time_iso8601()} - {level.name} - {msg}"
                self._msg_queue.put_nowait((log_msg, level))
        except Exception:
            traceback.print_exc(file=sys.stderr)

    def set_log_level(self, level: LogLevel):
        """
        Sets the logger's base log level at runtime.

        Args:
            level (LogLevel): The new base log level.
        """
        self._config.base_level = level

    def trace(self, msg: str):
        """
        Logs a message at TRACE level.

        Args:
            msg (str): The text to log at TRACE level.
        """
        self._submit_log(LogLevel.TRACE, msg)

    def debug(self, msg: str):
        """
        Logs a message at DEBUG level.

        Args:
            msg (str): The text to log at DEBUG level.
        """
        self._submit_log(LogLevel.DEBUG, msg)

    def info(self, msg: str):
        """
        Logs a message at INFO level.

        Args:
            msg (str): The text to log at INFO level.
        """
        self._submit_log(LogLevel.INFO, msg)

    def warning(self, msg: str):
        """
        Logs a message at WARNING level.

        Args:
            msg (str): The text to log at WARNING level.
        """
        self._submit_log(LogLevel.WARNING, msg)

    def error(self, msg: str):
        """
        Logs a message at ERROR level.

        Args:
            msg (str): The text to log at ERROR level.
        """
        self._submit_log(LogLevel.ERROR, msg)

    def critical(self, msg: str):
        """
        Logs a message at CRITICAL level.

        Args:
            msg (str): The text to log at CRITICAL level.
        """
        self._submit_log(LogLevel.CRITICAL, msg)

    async def stop(self):
        """
        Shuts down the logger, ensuring all buffered messages are flushed
        and handlers are closed.
        """
        self._shutdown_flag = True
        await self._msg_queue.join()

        if self._buffer:
            await self._flush_buffer()

        for handler in self._handlers:
            await handler.close()
