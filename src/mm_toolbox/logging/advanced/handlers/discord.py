"""Discord webhook log handler for advanced logging."""

from mm_toolbox.logging.advanced.handlers.base import BaseLogHandler


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
                f"Invalid webhook format; expected 'https://discord.com/api/webhooks/*' but got {webhook}"
            )

        self.url = webhook
        self.headers = {"Content-Type": "application/json"}

    def push(self, name, logs):
        try:
            formatted_logs = "\n".join(
                [
                    self.format_log(name=name, time_ns=log[0], level=log[1], msg=log[2])
                    for log in logs
                ]
            )
            self.ev_loop.create_task(
                self.http_session.post(
                    url=self.url,
                    headers=self.headers,
                    data=self.encode_json({"content": formatted_logs}),
                )
            )

        except Exception as e:
            print(f"Failed to send message to Discord; {str(e)}")
