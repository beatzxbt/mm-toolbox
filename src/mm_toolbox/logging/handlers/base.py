from abc import ABC, abstractmethod
from typing import List

class LogConfig(ABC):
    """
    Abstract base class for log configurations.

    Provides an interface for setting parameters such as buffer sizes, 
    flush intervals, output file paths, etc.
    """

    @abstractmethod
    def validate(self) -> None:
        """
        Validate the configuration. This method should raise an exception if 
        the configuration is invalid.
        """
        pass


class LogHandler(ABC):
    """
    Abstract base class for log handlers.

    Provides an interface for handling log entries, which must be implemented
    by subclasses to define specific logging behavior, such as writing to a file
    or sending logs to external services.
    """
    
    @abstractmethod
    async def flush(self, buffer: List[str]) -> None:
        """Flush any buffered log entries."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close any open files or connections for the handler."""
        pass
