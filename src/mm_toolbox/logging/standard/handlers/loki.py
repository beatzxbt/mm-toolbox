import aiosonic
import msgspec
from dataclasses import dataclass
from typing import List
from mm_toolbox.time import time_iso8601

from .base import LogConfig, LogHandler


@dataclass
class LokiLogConfig(LogConfig):
    url: str = "http://localhost:3100/loki/api/v1/push"
    labels: dict = None
    auth_token: str = None

    def validate(self) -> None:
        if not self.url:
            raise ValueError("Loki URL must be specified.")
        if not isinstance(self.labels, dict):
            raise ValueError("Labels must be a dictionary.")


class LokiLogHandler(LogHandler):
    encoder = msgspec.json.Encoder()

    def __init__(self, config: LokiLogConfig) -> None:
        config.validate()

        self.url = config.url
        self.labels = config.labels
        self.auth_token = config.auth_token
        self.session = aiosonic.HTTPClient()

        self._headers = {"Content-Type": "application/json", "Authorization": ""}

    async def flush(self, buffer: List[str]) -> None:
        """
        Send the buffer of logs to Loki.
        """
        log_stream = {
            "stream": self.labels,
            "values": [[time_iso8601(), log] for log in buffer],
        }

        log_json = self.encoder.encode({"streams": [log_stream]})

        if self.auth_token:
            self._headers["Authorization"] = f"Bearer {self.auth_token}"

        try:
            async with self.session.post(
                self.url, data=log_json, headers=self._headers
            ) as resp:
                if resp.status != 204:
                    print(
                        f"Failed to send logs to Loki: {resp.status}, {await resp.text()}."
                    )

        except Exception as e:
            print(f"Error sending logs to Loki: {e}")

    async def close(self) -> None:
        await self.client.connector.cleanup()
        del self.client
