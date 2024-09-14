import asyncio
from dataclasses import dataclass
from typing import List
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

LOG_LEVEL_MAP = {10: "DEBUG", 20: "INFO", 30: "WARNING", 40: "ERROR", 50: "CRITICAL"}

LOG_LEVELS = {10, 20, 30, 40, 50}


@dataclass
class LoggerConfig(LogConfig):
    """
    Core configuration for the Logger.

    Parameters
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
        if self.base_level not in LOG_LEVEL_MAP.values():
            raise ValueError(f"Invalid base log level name: {self.base_level}")
        if self.max_buffer_size < 1:
            raise ValueError("Max buffer size must be positive.")
        if self.max_buffer_age < 1:
            raise ValueError("Max buffer age must be positive.")


class Logger:
    def __init__(
        self,
        logger_config: LoggerConfig = None,
        file_config: FileLogConfig = None,
        discord_config: DiscordLogConfig = None,
        telegram_config: TelegramLogConfig = None,
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
            file_config.validate()
            self._log_handlers.append(FileLogHandler(file_config))

        if discord_config:
            discord_config.validate()
            self._log_handlers.append(DiscordLogHandler(discord_config))

        if telegram_config:
            telegram_config.validate()
            self._log_handlers.append(TelegramLogHandler(telegram_config))

        self._queue = asyncio.Queue()
        self._ev_loop = asyncio.get_event_loop()
        self._shutdown_flag = False

        self._ev_loop.create_task(self._log_ingestor())

    def _get_log_level(self, level_name: str) -> int:
        """
        Converts a log level name to its corresponding integer value.

        Parameters
        ----------
        level_name : str
            The name of the log level (e.g., "DEBUG", "INFO").

        Returns
        -------
        int
            The integer value of the log level.
        """
        for level, name in LOG_LEVEL_MAP.items():
            if name == level_name:
                return level

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

                # Buffer messages below ERROR level
                else:
                    is_buffer_full = self.current_buffer_size >= self._max_buffer_size
                    is_buffer_old = (
                        time_s() - self.last_flush_time >= self._max_buffer_age
                    )

                    # Flush buffer if it's full or aged enough
                    if is_buffer_full or is_buffer_old:
                        await self._flush_buffer()

                self._queue.task_done()

            except Exception as e:
                raise Exception(f"Log writer loop: {e}")

        return

    def _submit_log(self, level: int, message: str) -> None:
        try:
            if level >= self._base_level:
                log_entry = f"{time_iso8601()} - {LOG_LEVEL_MAP[level]} - {message}"
                self._ev_loop.call_soon_threadsafe(
                    self._queue.put_nowait, (log_entry, level)
                )

        except Exception as e:
            raise Exception(f"Failed to submit log: {e}")

    def debug(self, message: str) -> None:
        self._submit_log(10, message)

    def info(self, message: str) -> None:
        self._submit_log(20, message)

    def warning(self, message: str) -> None:
        self._submit_log(30, message)

    def error(self, message: str) -> None:
        self._submit_log(40, message)

    def critical(self, message: str) -> None:
        self._submit_log(50, message)

    async def shutdown(self) -> None:
        """
        Shuts down the logger, flushing all buffers and closing handlers.
        """
        self._shutdown_flag = True  # Kills the ingestor task.
        await self._queue.join()
        if self.log_message_buffer:
            await self._flush_buffer()
        for handler in self._log_handlers:
            await handler.close()
