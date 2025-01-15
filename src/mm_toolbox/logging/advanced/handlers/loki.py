import msgspec
from mm_toolbox.time import time_iso8601
from .base import LogHandler

class LokiLogHandler(LogHandler):
    """
    A log handler that sends messages to a Loki endpoint over HTTP.
    """

    json_encoder = msgspec.json.Encoder()

    def __init__(
        self, 
        url: str,
        labels: dict,
        auth_token: str = ""
    ):
        """
        Initialize the LokiLogHandler.

        Args:
            url (str): The full Loki push endpoint URL.
            labels (dict): Labels to associate with all log entries in this stream.
            auth_token (str, optional): Bearer token for authentication if required. Defaults to "".
        """
        self.url = url
        self.labels = labels
        self.auth_token = auth_token
        
        self.headers = {"Content-Type": "application/json", "Authorization": ""}
        if self.auth_token:
            self.headers["Authorization"] = f"Bearer {self.auth_token}"

    def push(self, buffer):
        log_stream = {
            "stream": self.labels,
            "values": [[log.time, log.level, log.msg] for log in buffer.data],
        }

        try:
            self.ev_loop.create_task(
                coro=self.http_session.post(
                    url=self.url, 
                    headers=self.headers, 
                    data=self.json_encoder.encode({"streams": [log_stream]})
                )
            )

        except Exception as e:
            print(f"Error sending logs to Loki; {e}")
