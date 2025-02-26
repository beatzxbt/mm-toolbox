import zmq
from mm_toolbox.time import time_iso8601
from .base import LogHandler
from ...utils.zmq import ZmqConnection

class ZMQLogHandler(LogHandler):
    """
    A log handler that publishes messages to a ZeroMQ socket.

    Can use either 'ipc' or 'tcp' transport for distributing log messages.
    """

    def __init__(self, transport: str, path: str) -> None:
        """
        Initialize the ZMQLogHandler, binding a PUB socket at the specified path.

        Args:
            transport (str): Either "ipc" or "tcp".
            path (str): The endpoint path (e.g. "ipc:///some/path.ipc" or "tcp://127.0.0.1:5556").

        Raises:
            ValueError: If the transport or path is invalid.
        """
        super().__init__()

        self.transport = transport.lower()
        
        if self.transport == "ipc":
            if not path.startswith("ipc://"):
                raise ValueError(f"Invalid IPC path '{path}'. Must start with 'ipc://'.")
            self.path = path

        elif self.transport == "tcp":
            if not path.startswith("tcp://"):
                raise ValueError(f"Invalid TCP path '{path}'. Must start with 'tcp://'.")
            
            # Additional validation for host:port
            tcp_part = path[len("tcp://"):]
            slash_split = tcp_part.split("/", 1)
            host_port = slash_split[0]
            if ":" not in host_port:
                raise ValueError(f"TCP path '{path}' has no port specified.")
            
            host, port_str = host_port.split(":", 1)
            try:
                port = int(port_str)
            except ValueError:
                raise ValueError(f"Invalid port in TCP path '{path}'. Must be an integer.")
            
            if not (1 <= port <= 65535):
                raise ValueError(f"Port must be between 1 and 65535, got {port}.")

            self.path = path
        else:
            raise ValueError(f"Invalid transport; expected ['ipc', 'tcp'] but got '{self.transport}'")

        self.connection = ZmqConnection(
            socket_type=zmq.PUB,
            path=self.path,
            bind=True
        )
        self.connection.start()

    async def push(self, buffer) -> None:
        """
        Publish each message in the buffer via the ZeroMQ connection.

        Args:
            buffer (list[str]): The messages to publish.
        """
        for log_msg in buffer:
            self.connection.send(data=log_msg.encode())
