"""Discord webhook handler for the standard logger."""

from mm_toolbox.logging._remote import (
    DISCORD_CHUNK_CHARS,
    chunk_messages,
    make_rate_limiter,
)
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
        self._limiter = make_rate_limiter(5, 2)

    def push(self, buffer: list[str]) -> None:
        """Queue bounded message chunks for background delivery."""
        chunks = self._chunk_messages(buffer, max_chars=DISCORD_CHUNK_CHARS)
        for index, chunk in enumerate(chunks):
            if not self._enqueue_delivery(chunk, self._deliver_chunk):
                remaining = len(chunks) - index - 1
                if remaining:
                    self._handle_exception(
                        OverflowError(
                            f"delivery stopped; dropped {remaining} additional Discord chunks"
                        ),
                        "push",
                    )
                break

    @staticmethod
    def _chunk_messages(messages: list[str], max_chars: int) -> list[str]:
        return chunk_messages(messages, max_chars)

    async def _deliver_chunk(self, chunk: object) -> None:
        """Send one chunk with bounded timeout and transient retries."""
        session = await self._ensure_session()
        await self._post_with_retries(
            session,
            self.url,
            headers=self.headers,
            payload={"content": str(chunk)},
            service="Discord webhook",
        )
