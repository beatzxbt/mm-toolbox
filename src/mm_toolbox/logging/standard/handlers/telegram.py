"""Telegram bot handler for the standard logger.

Pushes buffered log messages to a Telegram chat via the Bot API,
chunking large batches to respect Telegram message size limits.
"""

import asyncio

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
        """Batch messages into chunks and send to Telegram."""
        chunks = self._chunk_messages(buffer, max_chars=4096)
        asyncio.run(self._push_chunks(chunks))

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
            msg_len = len(msg)
            if current and current_len + 1 + msg_len > max_chars:
                chunks.append("\n".join(current))
                current = [msg]
                current_len = msg_len
            else:
                if current:
                    current_len += 1 + msg_len
                else:
                    current_len = msg_len
                current.append(msg)

        if current:
            chunks.append("\n".join(current))

        return chunks

    async def _push_chunks(self, chunks: list[str]) -> None:
        """Send chunks to Telegram Bot API."""
        async with aiohttp.ClientSession() as session:
            for chunk in chunks:
                payload = {
                    "chat_id": self.chat_id,
                    "text": chunk,
                }
                try:
                    async with session.post(
                        self.url,
                        headers=self.headers,
                        json=payload,
                    ) as response:
                        if response.status >= 400:
                            raise RuntimeError(
                                f"Telegram API returned {response.status}"
                            )
                except Exception as exc:
                    self._handle_exception(exc, "Telegram push")
