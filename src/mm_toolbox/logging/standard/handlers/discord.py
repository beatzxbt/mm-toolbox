"""Discord webhook handler for the standard logger."""

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
        """Queue bounded message chunks for background delivery."""
        chunks = self._chunk_messages(buffer, max_chars=2000)
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
                payload={"content": str(chunk)},
                service="Discord webhook",
            )
