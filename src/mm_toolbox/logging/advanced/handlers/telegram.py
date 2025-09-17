"""Telegram bot log handler for advanced logging."""

from mm_toolbox.logging.advanced.handlers.base import BaseLogHandler


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

    def push(self, logs):
        try:
            for log in logs:
                self.partial_payload.update(
                    {
                        "text": self.format_log(log)
                    }
                )
                self.ev_loop.create_task(
                    self.http_session.post(
                        url=self.url,
                        headers=self.headers,
                        data=self.encode_json(self.partial_payload),
                    )
                )

        except Exception as e:
            print(f"Failed to send message to Telegram; {e}")
