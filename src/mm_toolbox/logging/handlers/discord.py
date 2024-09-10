import asyncio
import aiosonic
import msgspec
from typing import List, Coroutine
from dataclasses import dataclass
from .base import LogConfig, LogHandler


@dataclass
class DiscordLogConfig(LogConfig):
    webhook: str = ""

    def validate(self) -> None:
        if (
            not self.webhook
            or self.webhook.find("https://discord.com/api/webhooks/") == -1
        ):
            raise ValueError("Missing or invalid webhook url.")


class DiscordLogHandler(LogHandler):
    json_encoder = msgspec.json.Encoder()

    def __init__(self, config: DiscordLogConfig) -> None:
        self.url = config.webhook
        self.headers = {"Content-Type": "application/json"}

        self.client = aiosonic.HTTPClient()

    async def flush(self, buffer) -> None:
        try:
            tasks: List[Coroutine] = []

            for log in buffer:
                tasks.append(
                    self.client.post(
                        url=self.url,
                        headers=self.headers,
                        json=self.json_encoder.encode({"content": log}),
                    )
                )

            await asyncio.gather(*tasks)

        except Exception as e:
            print(f"Failed to send message to Discord: {e}")

    async def close(self) -> None:
        await self.client.connector.cleanup()
        del self.client
