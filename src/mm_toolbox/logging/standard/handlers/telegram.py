"""Telegram bot log handler for standard logging."""

import asyncio

from mm_toolbox.logging.standard.handlers.base import BaseLogHandler


class TelegramLogHandler(BaseLogHandler):
    """A log handler that sends messages to a Telegram chat via bot API."""

    def __init__(self, bot_token: str, chat_id: str) -> None:
        """Initialize the TelegramLogHandler with bot token and chat ID.

        Args:
            bot_token (str): The Telegram bot token for authorization.
            chat_id (str): The ID of the Telegram chat to receive messages.

        """
        super().__init__()
        self.chat_id = chat_id
        self.url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        self.headers = {"Content-Type": "application/json"}
        self.payload = {
            "chat_id": self.chat_id,
            "text": "",
            "disable_web_page_preview": True,
        }

    async def push(self, buffer):
        try:
            tasks = [self._post(log_msg) for log_msg in buffer]
        except Exception as e:
            self._handle_exception(e, "push")
            return
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, Exception):
                self._handle_exception(res, "push")

    async def _post(self, log_msg: str) -> None:
        """Send a single log message to Telegram.

        Args:
            log_msg (str): Formatted log message content.

        """
        payload = dict(self.payload)
        payload["text"] = log_msg
        resp = await self.http_session.post(
            url=self.url,
            headers=self.headers,
            data=self.json_encode(payload),
        )
        await resp.read()
