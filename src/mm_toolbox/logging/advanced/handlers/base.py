import asyncio
import msgspec
import aiohttp
from abc import ABC, abstractmethod

from mm_toolbox.time import time_iso8601
from mm_toolbox.logging.advanced.config import LoggerConfig
from mm_toolbox.logging.advanced.structs import LogLevel

class BaseLogHandler(ABC):
    """
    Abstract base class for log handlers.

    All handlers must implement `.push(buffer)`, called from the
    main logger whenever the buffer fills up.

    Validation for any params/args should be done in '__init__' 
    to catch config errors early.
    """
    def __init__(self):
        self._encode_json = None
        self._http_session = None
        self._ev_loop = None
        self._primary_config = None

    @property
    def encode_json(self):
        """Lazily initialize the JSON encoder."""
        if self._encode_json is None:
            self._encode_json = msgspec.json.Encoder().encode
        return self._encode_json

    @property
    def http_session(self):
        """Lazily initialize the HTTP session."""
        if self._http_session is None:
            self._http_session = aiohttp.ClientSession()
        return self._http_session

    @property
    def ev_loop(self):
        """Lazily initialize the event loop."""
        if self._ev_loop is None:
            try:
                self._ev_loop = asyncio.get_event_loop()
            except RuntimeError:
                # If there's no event loop in the current context, create one.
                self._ev_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._ev_loop)
        return self._ev_loop
    
    @property
    def primary_config(self):
        """Get the primary config."""
        return self._primary_config
    
    def add_primary_config(self, config: LoggerConfig):
        """Add the primary config to the handler."""
        self._primary_config = config

    def format_log(self, name: bytes, time_ns: int, level: LogLevel, msg: bytes) -> bytes:
        """Format a log message to a string."""
        if self._primary_config:
            return self.primary_config.str_format % {
                "asctime": time_iso8601(float(time_ns)),
                "levelname": level.value,
                "name": name.decode(),
                "message": msg.decode()
            }.encode()
        else:
            raise RuntimeError(f"No primary config found for handler {self.__class__.__name__}")
    
    @abstractmethod
    def push(self, name: bytes, logs: list[tuple[int, LogLevel, bytes]]):
        """Push a batch of log messages to the external system.
        
        Args:
            name: The name of the log batch.
            logs: A batch of log messages in the format: [(time_ns: int, level: LogLevel, msg: bytes)]
        """
        pass
    
    def __del__(self):
        """Clean up resources when the handler is garbage collected."""
        if self._http_session is not None and not self._http_session.closed:
            try:
                if self._ev_loop is None or self._ev_loop.is_closed():
                    loop = asyncio.new_event_loop()
                    loop.run_until_complete(self._http_session.close())
                    loop.close()
                else:
                    if asyncio.get_event_loop_policy().get_event_loop().is_running():
                        asyncio.create_task(self._http_session.close())
                    else:
                        self._ev_loop.run_until_complete(self._http_session.close())
            except Exception:
                # Suppress exceptions during cleanup
                pass