"""Telegram bot handler for the standard logger."""

from mm_toolbox.logging._remote import (
    TELEGRAM_CHUNK_CHARS,
    chunk_messages,
    make_rate_limiter,
)
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
        self._limiter = make_rate_limiter(1, 1)

    def push(self, buffer: list[str]) -> None:
        """Queue bounded message chunks for background delivery."""
        chunks = self._chunk_messages(buffer, max_chars=TELEGRAM_CHUNK_CHARS)
        for index, chunk in enumerate(chunks):
            if not self._enqueue_delivery(chunk, self._deliver_chunk):
                remaining = len(chunks) - index - 1
                if remaining:
                    self._handle_exception(
                        OverflowError(
                            f"delivery stopped; dropped {remaining} additional Telegram chunks"
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
            payload={
                "chat_id": self.chat_id,
                "text": str(chunk),
                "disable_web_page_preview": True,
            },
            service="Telegram API",
        )
