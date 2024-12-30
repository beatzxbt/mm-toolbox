import zmq
import msgspec
from typing import Union, Literal
from dataclasses import dataclass
from mm_toolbox import time_iso8601

from .base import LogConfig, LogHandler


@dataclass
class ZMQLogConfig(LogConfig):
    transport: Union[Literal["IPC"], Literal["TCP"]]
    host: str = "localhost"  # Default host for TCP transport
    port: int = None  # Port is required for TCP transport
    path: str = None  # Path is required for IPC transport

    def validate(self) -> None:
        if self.transport == "TCP":
            if self.port is None:
                raise ValueError("Port must be specified for TCP transport.")
            if not (0 <= self.port <= 65535):
                raise ValueError("Port must be between 0 and 65535.")
        elif self.transport == "IPC":
            if not self.path:
                raise ValueError("Path must be specified for IPC transport.")
        else:
            raise ValueError(f"Unsupported transport type: {self.transport}")

        if self.buffer_size <= 0:
            raise ValueError("Buffer size must be greater than 0")
        if self.flush_interval <= 0:
            raise ValueError("Flush interval must be greater than 0")


class ZMQLogHandler(LogHandler):
    encoder = msgspec.json.Encoder()

    def __init__(self, config: ZMQLogConfig) -> None:
        config.validate()
        self.transport = config.transport
        self.host = config.host
        self.port = config.port
        self.path = config.path
        self.buffer_size = config.buffer_size
        self.flush_interval = config.flush_interval

        # Initialize ZeroMQ context and socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)

        if self.transport == "TCP":
            address = f"tcp://{self.host}:{self.port}"
            self.socket.connect(address)
        elif self.transport == "IPC":
            address = f"ipc://{self.path}"
            self.socket.connect(address)
        else:
            raise ValueError(f"Unsupported transport type: {self.transport}")

    async def flush(self, buffer) -> None:
        combined_logs = {
            "time": time_iso8601(),
            "logs": buffer,  # Assuming buffer is a list of log entries
        }
        message = self.encoder.encode(combined_logs)
        self.socket.send(message)

    async def close(self) -> None:
        self.socket.close()
        self.context.term()
