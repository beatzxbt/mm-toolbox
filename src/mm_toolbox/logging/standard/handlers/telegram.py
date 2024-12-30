import asyncio
import aiosonic
import msgspec
from typing import List, Coroutine
from dataclasses import dataclass
from .base import LogConfig, LogHandler


@dataclass
class TelegramLogConfig(LogConfig):
    bot_token: str = ""
    chat_id: str = ""

    def validate(self) -> None:
        """
        Validates the Telegram configuration.
        """
        if not self.bot_token:
            raise ValueError("Missing bot token.")

        if not self.chat_id.isnumeric():
            raise ValueError(f"Invalid chat ID: {self.chat_id}")


class TelegramLogHandler(LogHandler):
    json_encoder = msgspec.json.Encoder()

    def __init__(self, config: TelegramLogConfig) -> None:
        config.validate()
        self.chat_id = config.chat_id

        self.url = f"https://api.telegram.org/bot{config.bot_token}/sendMessage"
        self.headers = {"Content-Type": "application/json"}

        self.client = aiosonic.HTTPClient()

    async def flush(self, buffer: List[str]) -> None:
        try:
            tasks: List[Coroutine] = []
            for log in buffer:
                payload = {
                    "chat_id": self.chat_id,
                    "text": log,
                    "disable_web_page_preview": True,
                }
                tasks.append(
                    self.client.post(
                        url=self.url,
                        headers=self.headers,
                        json=self.json_encoder.encode(payload),
                    )
                )

            await asyncio.gather(*tasks)

        except Exception as e:
            print(f"Failed to send message to Telegram: {e}")

    async def close(self) -> None:
        await self.client.connector.cleanup()
        del self.client
