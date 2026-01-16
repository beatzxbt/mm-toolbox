"""Telegram bot log handler for advanced logging."""

from mm_toolbox.logging.advanced.handlers.base import BaseLogHandler, _RateLimiter


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
        await self._ensure_session()
        await self._limiter.acquire(1.0)
        payload = dict(self.partial_payload)
        payload["text"] = text
        resp = await self._http_session.post(  # type: ignore[union-attr]
            self.url,
            headers=self.headers,
            data=self.encode_json(payload),
        )
        await resp.read()

    @staticmethod
    def _chunk(text: str, limit: int) -> list[str]:
        if len(text) <= limit:
            return [text]
        parts: list[str] = []
        start = 0
        while start < len(text):
            end = min(len(text), start + limit)
            parts.append(text[start:end])
            start = end
        return parts

    def push(self, logs):
        try:
            # Telegram message limit ~4096 chars
            for log in logs:
                for chunk in self._chunk(self.format_log(log), 3500):
                    fut = self._run_coro(self._post(chunk))
                    self._track_future(fut)

        except Exception as e:
            self._handle_exception(e, "push")
