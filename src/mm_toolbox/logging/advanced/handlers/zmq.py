import zmq

from mm_toolbox.logging.advanced.handlers.base import BaseLogHandler
from mm_toolbox.logging.utils.zmq import ZmqConnection

class ZMQLogHandler(BaseLogHandler):
    """
    A log handler that publishes messages to a ZeroMQ socket.
    """

    def __init__(self, path: str, do_format: bool = True) -> None:
        """
        Initialize the ZMQLogHandler, binding a PUB socket at the specified path.

        Args:
            path (str): The endpoint path (e.g. "ipc:///some/path.ipc", "tcp://127.0.0.1:5556", 
                        or "inproc://logger").
            do_format (bool, optional): If True, format the log messages before sending. 
                                        If False, send the raw log objects. Defaults to True.
        """
        super().__init__()
        self.path = path

        self.do_format = do_format

        # Any formatting issues with the path will be thrown within the ZmqConnection constructor 
        # by ZMQ, so we don't need to handle it beforehand. 
        self.connection = ZmqConnection(
            socket_type=zmq.PUB,
            path=self.path,
            bind=True
        )
        self.connection.start()

    def push(self, name, logs) -> None:
        """
        Publish each message in the buffer via the ZeroMQ connection.

        Args:
            buffer: A batch of log messages.
        """
        try:
            for log in logs:
                if self.do_format:
                    encoded_log = self.format_log(
                        name=name, 
                        time_ns=log[0], 
                        level=log[1], 
                        msg=log[2]
                    )
                else:
                    encoded_log = self.encode_json(log)

                self.connection.send(encoded_log)
                
        except Exception as e:
            print(f"Failed to publish logs via ZMQ; {e}")