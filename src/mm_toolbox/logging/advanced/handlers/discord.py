"""Discord webhook log handler for advanced logging."""

from mm_toolbox.logging.advanced.handlers.base import BaseLogHandler, _RateLimiter
from mm_toolbox.logging.advanced.pylog import PyLog


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
        # Discord: ~5 req/2s per webhook. Use conservative defaults.
        self._limiter = _RateLimiter(rate_per_sec=2.5, burst=5)

    async def _post(self, content: str) -> None:
        """Send a single log message to Discord.

        Args:
            content (str): Message content to send.
        """
        await self._ensure_session()
        await self._limiter.acquire(1)
        resp = await self._http_session.post(  # type: ignore[union-attr]
            self.url,
            headers=self.headers,
            data=self.encode_json({"content": content}),
        )
        await resp.read()

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
        """Send a batch of log messages to Discord.

        Args:
            logs (list[PyLog]): Batch of log entries.
        """
        try:
            formatted_logs = "\n".join([self.format_log(log) for log in logs])
            # Discord limit is 2000 chars; keep headroom for safety
            for chunk in self._chunk(formatted_logs, 1800):
                fut = self._run_coro(self._post(chunk))
                self._track_future(fut)

        except Exception as e:
            self._handle_exception(e, "push")
