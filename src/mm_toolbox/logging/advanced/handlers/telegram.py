import asyncio
from typing import Coroutine
from .base import LogHandler

class TelegramLogHandler(LogHandler):
    """
    A log handler that sends messages to a Telegram chat via bot API.
    """

    def __init__(self, bot_token: str, chat_id: str) -> None:
        """
        Initialize the TelegramLogHandler with bot token and chat ID.

        Args:
            bot_token (str): The Telegram bot token for authorization.
            chat_id (str): The ID of the Telegram chat to receive messages.
        """
        self.chat_id = chat_id
        self.url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        self.headers = {"Content-Type": "application/json"}
        self.partial_payload = {
            "chat_id": self.chat_id,
            "text": "",
            "disable_web_page_preview": True,
        }

    def push(self, buffer):
        try:
            for log in buffer.data:
                self.partial_payload["text"] = f"{log.time} - {log.level} - {log.msg}"
                self.ev_loop.create_task(
                    self.http_session.post(
                        url=self.url,
                        headers=self.headers,
                        data=self.json_encoder.encode(self.partial_payload),
                    )
                )

        except Exception as e:
            print(f"Failed to send message to Telegram; {e}")
