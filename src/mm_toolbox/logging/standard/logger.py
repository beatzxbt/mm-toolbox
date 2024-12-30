import sys
import traceback
import asyncio
from dataclasses import dataclass
from typing import List, Optional
from mm_toolbox.time import time_iso8601, time_s

from .handlers import (
    LogConfig,
    LogHandler,
    FileLogConfig,
    FileLogHandler,
    DiscordLogConfig,
    DiscordLogHandler,
    TelegramLogConfig,
    TelegramLogHandler,
)

# Updated LOG_LEVEL_MAP with TRACE level
LOG_LEVEL_MAP = {
    5: "TRACE",
    10: "DEBUG",
    20: "INFO",
    30: "WARNING",
    40: "ERROR",
    50: "CRITICAL",
}

LOG_LEVELS = {5, 10, 20, 30, 40, 50}


@dataclass
class LoggerConfig(LogConfig):
    """
    Core configuration for the Logger.

    Attributes
    ----------
    base_level : str
        The minimum log level to record. Default is "WARNING".

    stout : bool
        Whether to print logs to stdout. Default is True.

    max_buffer_size : int
        Maximum number of log messages to buffer. Default is 10.

    max_buffer_age : int
        Maximum age (in seconds) before flushing the buffer. Default is 10.
    """

    base_level: str = "WARNING"
    stout: bool = True
    max_buffer_size: int = 10
    max_buffer_age: int = 10

    def validate(self) -> None:
        """
        Validates the LoggerConfig settings.

        Raises
        ------
        ValueError
            If any configuration parameter is invalid.
        """
        if self.base_level not in LOG_LEVEL_MAP.values():
            raise ValueError(f"Invalid base log level name: {self.base_level}")
        if self.max_buffer_size < 1:
            raise ValueError("Max buffer size must be positive.")
        if self.max_buffer_age < 1:
            raise ValueError("Max buffer age must be positive.")


class Logger:
    """
    A customizable logger that supports asynchronous logging and multiple handlers.

    Parameters
    ----------
    logger_config : LoggerConfig, optional
        Configuration for the Logger. If not provided, defaults will be used.

    file_config : FileLogConfig, optional
        Configuration for file logging.

    discord_config : DiscordLogConfig, optional
        Configuration for Discord logging.

    telegram_config : TelegramLogConfig, optional
        Configuration for Telegram logging.
    """

    def __init__(
        self,
        logger_config: Optional[LoggerConfig] = None,
        file_config: Optional[FileLogConfig] = None,
        discord_config: Optional[DiscordLogConfig] = None,
        telegram_config: Optional[TelegramLogConfig] = None,
    ) -> None:
        self.logger_config = logger_config if logger_config else LoggerConfig()
        self.logger_config.validate()

        self.log_message_buffer: List[str] = []
        self.current_buffer_size = 0
        self.last_flush_time = time_s()

        self._base_level = self._get_log_level(self.logger_config.base_level)
        self._stout = self.logger_config.stout
        self._max_buffer_size = self.logger_config.max_buffer_size
        self._max_buffer_age = self.logger_config.max_buffer_age

        self._log_handlers: List[LogHandler] = []

        if file_config:
            self._log_handlers.append(FileLogHandler(file_config))

        if discord_config:
            self._log_handlers.append(DiscordLogHandler(discord_config))

        if telegram_config:
            self._log_handlers.append(TelegramLogHandler(telegram_config))

        self._queue = asyncio.Queue()
        self._ev_loop = asyncio.get_event_loop()
        self._shutdown_flag = False

        # Start the log ingestor task
        self._ev_loop.create_task(self._log_ingestor())

    def _get_log_level(self, level_name: str) -> int:
        """
        Converts a log level name to its corresponding integer value.

        Parameters
        ----------
        level_name : str
            The name of the log level (e.g., "TRACE", "DEBUG", "INFO").

        Returns
        -------
        int
            The integer value of the log level.

        Raises
        ------
        ValueError
            If the level name is invalid.
        """
        for level, name in LOG_LEVEL_MAP.items():
            if name == level_name:
                return level
        raise ValueError(f"Invalid log level name: {level_name}")

    async def _flush_buffer(self) -> None:
        """
        Flushes the log message buffer to all handlers.
        """
        for handler in self._log_handlers:
            await handler.flush(self.log_message_buffer)
        self.log_message_buffer.clear()
        self.current_buffer_size = 0
        self.last_flush_time = time_s()

    async def _log_ingestor(self) -> None:
        """
        Asynchronous loop that processes log messages from the queue.

        This method runs in a separate task and ingests log messages from the queue,
        flushing the buffer to handlers based on buffer size or age.

        Raises
        ------
        Exception
            If there is an error during the log writing process.
        """
        while not self._shutdown_flag or not self._queue.empty():
            try:
                log_entry, level = await self._queue.get()

                self.log_message_buffer.append(log_entry)
                self.current_buffer_size += 1

                if self._stout:
                    print(log_entry)

                # Immediate flush for ERROR or CRITICAL levels
                if level >= 40:
                    await self._flush_buffer()
                else:
                    # Buffer messages below ERROR level
                    is_buffer_full = self.current_buffer_size >= self._max_buffer_size
                    is_buffer_old = (
                        time_s() - self.last_flush_time >= self._max_buffer_age
                    )

                    # Flush buffer if it's full or aged enough
                    if is_buffer_full or is_buffer_old:
                        await self._flush_buffer()

                self._queue.task_done()

            except Exception:
                traceback.print_exc(file=sys.stderr)

    def _submit_log(self, level: int, message: str) -> None:
        """
        Submits a log message to the queue if it meets the base level requirement.

        Parameters
        ----------
        level : int
            The log level of the message.

        message : str
            The log message.
        """
        try:
            if level >= self._base_level:
                log_entry = f"{time_iso8601()} - {LOG_LEVEL_MAP[level]} - {message}"
                self._ev_loop.call_soon_threadsafe(
                    self._queue.put_nowait, (log_entry, level)
                )
        except Exception as e:
            # Log the exception and continue
            print(f"Failed to submit log: {e}")

    def set_log_level(self, level_name: str) -> None:
        """
        Sets the base log level at runtime.

        Parameters
        ----------
        level_name : str
            The name of the new base log level.
        """
        self._base_level = self._get_log_level(level_name)

    def trace(self, message: str) -> None:
        """
        Logs a message with level TRACE.

        Parameters
        ----------
        message : str
            The message to log.
        """
        self._submit_log(5, message)

    def debug(self, message: str) -> None:
        """
        Logs a message with level DEBUG.

        Parameters
        ----------
        message : str
            The message to log.
        """
        self._submit_log(10, message)

    def info(self, message: str) -> None:
        """
        Logs a message with level INFO.

        Parameters
        ----------
        message : str
            The message to log.
        """
        self._submit_log(20, message)

    def warning(self, message: str) -> None:
        """
        Logs a message with level WARNING.

        Parameters
        ----------
        message : str
            The message to log.
        """
        self._submit_log(30, message)

    def error(self, message: str) -> None:
        """
        Logs a message with level ERROR.

        Parameters
        ----------
        message : str
            The message to log.
        """
        self._submit_log(40, message)

    def critical(self, message: str) -> None:
        """
        Logs a message with level CRITICAL.

        Parameters
        ----------
        message : str
            The message to log.
        """
        self._submit_log(50, message)

    async def shutdown(self) -> None:
        """
        Shuts down the logger, flushing all buffers and closing handlers.

        This method should be called when the logger is no longer needed to ensure
        that all log messages are properly flushed and resources are cleaned up.
        """
        self._shutdown_flag = True  # Stops the ingestor task
        await self._queue.join()
        if self.log_message_buffer:
            await self._flush_buffer()
        for handler in self._log_handlers:
            await handler.close()
