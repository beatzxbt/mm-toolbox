"""Discord webhook handler for the standard logger.

Pushes buffered log messages to a Discord channel via HTTP POST,
chunking large batches to stay within Discord message limits.
"""

import asyncio

import aiohttp

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

    def push(self, buffer: list[str]) -> None:
        """Batch messages into chunks and send to Discord."""
        chunks = self._chunk_messages(buffer, max_chars=2000)
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
        """Send chunks to Discord webhook."""
        async with aiohttp.ClientSession() as session:
            for chunk in chunks:
                payload = {"content": chunk}
                try:
                    async with session.post(
                        self.url,
                        headers=self.headers,
                        json=payload,
                    ) as response:
                        if response.status >= 400:
                            raise RuntimeError(
                                f"Discord webhook returned {response.status}"
                            )
                except Exception as exc:
                    self._handle_exception(exc, "Discord push")
