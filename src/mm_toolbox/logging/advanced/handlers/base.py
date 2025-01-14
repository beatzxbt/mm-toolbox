import asyncio
import msgspec
import aiohttp
from abc import ABC, abstractmethod
from typing import Union 

from ..structs import LogMessageBatch, DataMessageBatch

type MessageBatch = Union[LogMessageBatch, DataMessageBatch]

class LogHandler(ABC):
    """
    Abstract base class for log handlers.
    All handlers must implement `push(buffer)`.
    Validation should be done in __init__ to catch config errors early.
    """
    def __init__(self):
        self.json_encoder = msgspec.json.Encoder()
        self.http_session = aiohttp.ClientSession()
        self.ev_loop = asyncio.get_event_loop()

    @abstractmethod
    def push(self, buffer: MessageBatch):
        """
        Push a batch of log messages to the external system.

        Parameters
        ----------
        buffer : MessageBatch
            A pointer to the first LogMessage in a contiguous array.
            Implementations should convert this to a Python structure as needed.
        """
        pass