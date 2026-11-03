"""Discord webhook handler for the advanced logger.

Pushes batches of ``PyLog`` entries to a Discord channel via HTTP POST,
with rate limiting and message chunking to stay within platform limits.
"""

from mm_toolbox.logging._remote import (
    DISCORD_CHUNK_CHARS,
    chunk_messages,
    make_rate_limiter,
)
from mm_toolbox.logging.advanced.handlers.base import BaseLogHandler
from mm_toolbox.logging.advanced.pylog import PyLog


class DiscordLogHandler(BaseLogHandler):
    """A log handler that sends messages to a Discord webhook."""

    def __init__(self, webhook: str) -> None:
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
        self._limiter = make_rate_limiter(5, 2)

    async def _post(self, content: str) -> None:
        """Send a single log message to Discord.

        Args:
            content (str): Message content to send.
        """
        await self._post_json_with_retries(
            self.url,
            {"content": content},
            limiter=self._limiter,
            service="Discord webhook",
        )

    @staticmethod
    def _chunk(text: str, limit: int) -> list[str]:
        return chunk_messages((text,), limit)

    def push(self, logs: list[PyLog]) -> None:
        """Send a batch of log messages to Discord.

        Args:
            logs (list[PyLog]): Batch of log entries.
        """
        try:
            chunks = chunk_messages(
                (self.format_log(log) for log in logs), DISCORD_CHUNK_CHARS
            )
            for index, chunk in enumerate(chunks):
                if self._submit_coro(self._post(chunk)) is None:
                    self._handle_exception(
                        OverflowError(
                            f"delivery capacity reached; dropped {len(chunks) - index} Discord chunks"
                        ),
                        "push",
                    )
                    break

        except Exception as e:
            self._handle_exception(e, "push")
