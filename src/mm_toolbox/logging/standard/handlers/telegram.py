"""Telegram bot handler for the standard logger."""

import aiohttp

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

    def push(self, buffer: list[str]) -> None:
        """Queue bounded message chunks for background delivery."""
        chunks = self._chunk_messages(buffer, max_chars=4096)
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

    def _chunk_messages(self, messages: list[str], max_chars: int) -> list[str]:
        """Group messages into chunks that stay under max_chars.

        Args:
            messages (list[str]): Individual log messages.
            max_chars (int): Maximum characters per chunk.

        Returns:
            list[str]: List of chunk strings.

        """
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for msg in messages:
            # Split an oversized individual message so the service never
            # receives a payload above its documented limit.
            pieces = [msg[i : i + max_chars] for i in range(0, len(msg), max_chars)]
            if not pieces:
                pieces = [""]
            for piece in pieces:
                piece_len = len(piece)
                if current and current_len + 1 + piece_len > max_chars:
                    chunks.append("\n".join(current))
                    current = [piece]
                    current_len = piece_len
                else:
                    current_len += (1 if current else 0) + piece_len
                    current.append(piece)

        if current:
            chunks.append("\n".join(current))

        return chunks

    async def _deliver_chunk(self, chunk: object) -> None:
        """Send one chunk with bounded timeout and transient retries."""
        timeout = aiohttp.ClientTimeout(total=3.0)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            await self._post_with_retries(
                session,
                self.url,
                headers=self.headers,
                payload={"chat_id": self.chat_id, "text": str(chunk)},
                service="Telegram API",
            )
