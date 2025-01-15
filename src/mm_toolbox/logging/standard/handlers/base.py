import asyncio
import msgspec
import aiohttp
from abc import ABC, abstractmethod

class LogHandler(ABC):
    """
    Abstract base class for log handlers, defining how log messages 
    should be pushed to their respective destinations.
    """
    def __init__(self):
        self.ev_loop = asyncio.get_event_loop()
        self.json_encoder = msgspec.json.Encoder()
        self.http_session = aiohttp.ClientSession()

    @abstractmethod
    async def push(self, buffer: list[str]) -> None:
        """
        Flushes the given buffer of log entries in some way.

        Args:
            buffer (list[str]): The list of log messages to push.
        """
        pass