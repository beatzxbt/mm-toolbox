import asyncio
import time
import zmq
import zmq.asyncio

class ZmqConnection:
    """
    An abstract ZeroMQ connection capable of creating and managing
    various socket types (PUB, SUB, REQ, REP, DEALER, ROUTER, etc.).

    This class can bind or connect to a specified endpoint, optionally
    applying a subscription filter (for SUB/XSUB sockets) or running a
    warm-up interval (for PUB/XPUB sockets).
    """

    def __init__(
        self,
        socket_type=zmq.SUB,
        path: str = "tcp://127.0.0.1:5555",
        subscribe_filter: str = "",
        warmup_timeout: float = 0.0,
        bind: bool = False,
        snd_hwm: int = 100_000,
        rcv_hwm: int = 100_000,
    ):
        """
        Initialize a ZMQ connection with customizable parameters.

        Args:
            socket_type (int, optional): A valid ZMQ socket type, e.g. zmq.PUB, zmq.SUB, zmq.REQ, etc.
                Defaults to zmq.SUB.
            path (str, optional): The endpoint to bind or connect to, e.g. "tcp://127.0.0.1:5555".
                Defaults to "tcp://127.0.0.1:5555".
            subscribe_filter (str, optional): The subscription filter for SUB/XSUB sockets.
                Ignored otherwise. Defaults to "" (empty string).
            warmup_timeout (float, optional): If > 0 and the socket is PUB/XPUB, waits
                this many seconds after start() to give subscribers time to connect.
                Defaults to 0.0.
            bind (bool, optional): If True, call bind() on the socket; otherwise call connect().
                Defaults to False.
            snd_hwm (int, optional): High-water mark for sends. Defaults to 100,000.
            rcv_hwm (int, optional): High-water mark for receives. Defaults to 100,000.
        """
        self.socket_type = socket_type
        self.path = path
        self.subscribe_filter = subscribe_filter
        self.warmup_timeout = warmup_timeout
        self.bind = bind
        self.snd_hwm = snd_hwm
        self.rcv_hwm = rcv_hwm

        self._context = zmq.asyncio.Context()
        self._socket = None
        self._is_started = False
        self._listen_task = None
        self._running = False

    async def start(self):
        """
        Create the socket and either bind or connect. If the socket is PUB/XPUB
        and warmup_timeout > 0, pause briefly to let subscribers connect.

        Raises:
            RuntimeError: If the socket is already started.
        """
        if self._is_started:
            return

        self._socket = self._context.socket(self.socket_type)
        self._socket.setsockopt(zmq.SNDHWM, self.snd_hwm)
        self._socket.setsockopt(zmq.RCVHWM, self.rcv_hwm)

        if self.bind:
            self._socket.bind(self.path)
        else:
            self._socket.connect(self.path)

        # For SUB/XSUB, apply the subscription filter
        if self.socket_type in (zmq.SUB, zmq.XSUB):
            self._socket.setsockopt_string(zmq.SUBSCRIBE, self.subscribe_filter)

        self._is_started = True

        # If PUB/XPUB and warmup_timeout is set, give subscribers time to connect
        if self.socket_type in (zmq.PUB, zmq.XPUB) and self.warmup_timeout > 0:
            await self._pub_warmup()

    async def _pub_warmup(self):
        """
        Briefly wait to allow subscribers to connect before sending messages.
        """
        start_ts = time.time()
        while time.time() - start_ts < self.warmup_timeout:
            await asyncio.sleep(0.1)

    async def stop(self):
        """
        Close the socket and terminate the context, stopping any I/O activity.
        """
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        self._context.term()
        self._is_started = False
        self._running = False

    async def send(self, data: bytes):
        """
        Send data on this socket.

        This method is valid for socket types that support outbound messages,
        such as PUB, PUSH, DEALER, REQ, etc.

        Args:
            data (bytes): The data to send.

        Raises:
            RuntimeError: If the socket is not started.
        """
        if not self._is_started:
            raise RuntimeError("Socket is not started. Call .start() first.")
        await self._socket.send(data, copy=False)

    async def recv(self) -> bytes:
        """
        Receive data on this socket.

        This method is valid for socket types that support inbound messages,
        such as SUB, PULL, DEALER, REP, etc.

        Returns:
            bytes: The data received from the socket.

        Raises:
            RuntimeError: If the socket is not started.
        """
        if not self._is_started:
            raise RuntimeError("Socket is not started. Call .start() first.")
        msg = await self._socket.recv()
        return msg

    def listen(self, on_message: callable):
        """
        Start a background task that repeatedly calls recv() and
        invokes on_message(msg) for each incoming message.

        Typically used for SUB sockets or other receiving sockets that want
        to handle messages asynchronously.

        Args:
            on_message (callable): A callback function that takes one argument: the received message.

        Raises:
            RuntimeError: If the socket is not started.
        """
        if not self._is_started:
            raise RuntimeError("Socket not started. Call await .start() first.")
        if self._listen_task:
            # Already listening
            return

        self._running = True
        loop = asyncio.get_event_loop()
        self._listen_task = loop.create_task(self._listen_loop(on_message))

    async def _listen_loop(self, on_message: callable):
        """
        Internal coroutine that continuously receives messages and invokes on_message(msg).
        """
        while self._running:
            try:
                msg = await self._socket.recv()
                on_message(msg)
            except asyncio.CancelledError:
                break

    def cancel_listening(self):
        """
        Cancel the background listening task, stopping any further automatic reception.
        """
        if self._listen_task:
            self._listen_task.cancel()
            self._listen_task = None
        self._running = False
