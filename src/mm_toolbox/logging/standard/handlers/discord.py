"""Discord webhook log handler for standard logging."""

import asyncio

from mm_toolbox.logging.standard.handlers.base import BaseLogHandler


class DiscordLogHandler(BaseLogHandler):
    """A log handler that sends messages to a Discord webhook."""

    def __init__(self, webhook: str):
        """Initializes the DiscordLogHandler.

        Args:
            webhook (str): The Discord webhook URL.

        Raises:
            ValueError: If webhook is invalid.

        """
        super().__init__()

        if not webhook.startswith("https://discord.com/api/webhooks/"):
            raise ValueError(
                f"Invalid webhook format; expected "
                f"'https://discord.com/api/webhooks/*' but got {webhook}"
            )

        self.url = webhook
        self.headers = {"Content-Type": "application/json"}

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
        """Send a single log message to Discord.

        Args:
            log_msg (str): Formatted log message content.

        """
        resp = await self.http_session.post(
            url=self.url,
            headers=self.headers,
            data=self.json_encode({"content": log_msg}),
        )
        await resp.read()
