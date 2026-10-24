"""Telegram bot handler for the advanced logger.

Pushes batches of ``PyLog`` entries to a Telegram chat via the Bot API,
with rate limiting and per-message chunking for size compliance.
"""

from mm_toolbox.logging.advanced.handlers.base import BaseLogHandler, _RateLimiter
from mm_toolbox.logging.advanced.pylog import PyLog


class TelegramLogHandler(BaseLogHandler):
    """A log handler that sends messages to a Telegram chat via bot API."""

    def __init__(self, bot_token: str, chat_id: str):
        """Initialize the TelegramLogHandler with bot token and chat ID.

        Args:
            bot_token (str): The Telegram bot token for authorization.
            chat_id (str): The ID of the Telegram chat to receive messages.

        """
        super().__init__()
        self.chat_id = chat_id
        self.url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        self.headers = {"Content-Type": "application/json"}
        self.partial_payload = {
            "chat_id": self.chat_id,
            "text": "",
            "disable_web_page_preview": True,
        }
        # Telegram: stronger limits and anti-flood; keep conservative
        self._limiter = _RateLimiter(rate_per_sec=1.0, burst=20)

    async def _post(self, text: str) -> None:
        """Send a single log message to Telegram.

        Args:
            text (str): Message content to send.
        """
        payload = dict(self.partial_payload)
        payload["text"] = text
        await self._post_json_with_retries(
            self.url,
            payload,
            limiter=self._limiter,
            service="Telegram API",
        )

    @staticmethod
    def _chunk(text: str, limit: int) -> list[str]:
        """Split a message into chunks under a character limit.

        Args:
            text (str): Full message text.
            limit (int): Maximum characters per chunk.

        Returns:
            list[str]: Chunked message parts.
        """
        if len(text) <= limit:
            return [text]
        parts: list[str] = []
        start = 0
        while start < len(text):
            end = min(len(text), start + limit)
            parts.append(text[start:end])
            start = end
        return parts

    def push(self, logs: list[PyLog]) -> None:
        """Send a batch of log messages to Telegram.

        Args:
            logs (list[PyLog]): Batch of log entries.
        """
        try:
            if not logs:
                return
            # Telegram message limit ~4096 chars
            chunks = []
            for log in logs:
                chunks.extend(self._chunk(self.format_log(log), 3500))
            for index, chunk in enumerate(chunks):
                if self._submit_coro(self._post(chunk)) is None:
                    self._handle_exception(
                        OverflowError(
                            f"delivery capacity reached; dropped {len(chunks) - index} Telegram chunks"
                        ),
                        "push",
                    )
                    break

        except Exception as e:
            self._handle_exception(e, "push")
