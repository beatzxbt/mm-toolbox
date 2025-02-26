import msgspec
from mm_toolbox.time import time_ns
from .base import LogHandler

class LokiLogHandler(LogHandler):
    """
    A log handler that sends messages to a Loki endpoint over HTTP.
    """

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
        super().__init__()
        self.url = url
        self.labels = labels
        self.auth_token = auth_token
        
        self.headers = {"Content-Type": "application/json"}
        if self.auth_token:
            self.headers["Authorization"] = f"Bearer {self.auth_token}"

    async def push(self, buffer: list[str]) -> None:
        # Loki expects timestamps in nanoseconds
        log_stream = {
            "stream": self.labels,
            "values": [[str(time_ns()), log] for log in buffer],
        }

        try:
            async with self.http_session.post(
                url=self.url, 
                headers=self.headers, 
                json={"streams": [log_stream]}
            ) as resp:
                if resp.status != 204:
                    # We can't log this in fear of it loop erroring. 
                    # Just print the failure and move on... 
                    print(f"Failed to send logs to Loki: {resp.status}, {await resp.text()}.")

        except Exception as e:
            print(f"Error sending logs to Loki; {e}")
