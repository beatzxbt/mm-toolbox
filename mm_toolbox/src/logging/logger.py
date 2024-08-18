import os
import aiofiles
import asyncio

from mm_toolbox.src.logging.discord import DiscordClient
from mm_toolbox.src.logging.telegram import TelegramClient
from mm_toolbox.src.time import time_iso8601

class Logger:
    def __init__(
        self,
        debug_mode: bool = False,
    ) -> None:
        self.debug_mode = debug_mode

        self.discord_client = None
        self.telegram_client = None

        self.send_to_discord = bool(os.getenv("DISCORD_WEBHOOK"))
        if self.send_to_discord:
            self.discord_client = DiscordClient()
            self.discord_client.start(
                webhook=os.getenv("DISCORD_WEBHOOK")
            )

        self.send_to_telegram = bool(
            os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_CHAT_ID")
        )
        if self.send_to_telegram:
            self.telegram_client = TelegramClient()
            self.telegram_client.start(
                bot_token=os.getenv("TELEGRAM_BOT_TOKEN"), 
                chat_id=os.getenv("TELEGRAM_CHAT_ID")
            )

        self.tasks = []
        self.msgs = []

    async def _write_logs_to_file_(self) -> None:
        """
        Asynchronously write log messages to a file.

        Returns
        -------
        None
        """
        try:
            async with aiofiles.open("logs.txt", "a") as file:
                await file.writelines(f"{line}\n" for line in self.msgs)
        except Exception as e:
            await self.error(f"Error writing logs to file: {e}")
        finally:
            self.msgs.clear()

    async def _message_(self, level: str, topic: str, msg: str, flush_buffer: bool=False) -> None:
        """
        Log a message with a specified logging level.

        Parameters
        ----------
        level : str
            The logging level of the message.

        topic : str
            The topic of the message to log.

        msg : str
            The message to log.

        Returns
        -------
        None
        """
        formatted_msg = f"{time_iso8601()} | {level} | {topic} | {msg}"

        if self.send_to_discord:
            task = asyncio.create_task(self.discord_client.send(formatted_msg, flush_buffer))
            self.tasks.append(task)

        if self.send_to_telegram:
            task = asyncio.create_task(self.telegram_client.send(formatted_msg, flush_buffer))
            self.tasks.append(task)

        print(formatted_msg)

        self.msgs.append(formatted_msg)

        if len(self.msgs) >= 1000 or flush_buffer:
            await self._write_logs_to_file_()

    async def success(self, topic: str, msg: str) -> None:
        await self._message_("SUCCESS", topic.upper(), msg)

    async def info(self, topic: str, msg: str) -> None:
        await self._message_("INFO", topic.upper(), msg)

    async def debug(self, topic: str, msg: str) -> None:
        if self.debug_mode:
            await self._message_("DEBUG", topic.upper(), msg)

    async def warning(self, topic: str, msg: str) -> None:
        await self._message_("WARNING", topic.upper(), msg)

    async def error(self, topic: str, msg: str) -> None:
        await self._message_("ERROR", topic.upper(), msg)

    async def critical(self, topic: str, msg: str) -> None:
        await self._message_("CRITICAL", topic.upper(), msg, flush_buffer=True)

    async def shutdown(self) -> None:
        """
        Shutdown the logger by ensuring all clients are closed and all tasks are complete.

        Returns
        -------
        None
        """
        if self.discord_client:
            await self.discord_client.shutdown()

        if self.telegram_client:
            await self.telegram_client.shutdown()

        if self.tasks:
            await asyncio.gather(*self.tasks)

        if self.msgs:
            await self._write_logs_to_file_()