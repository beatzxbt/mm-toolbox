import asyncio
import inspect
import time
import zmq
import zmq.asyncio
import threading

from typing import Callable

# Rebind it so its easier to understand and access as an Enum
from zmq.constants import SocketType as ZmqSocketType


class ZmqConnection:
    """
    A synchronous ZeroMQ connection capable of creating and managing
    various socket types (PUB, SUB, REQ, REP, DEALER, ROUTER, etc.).

    This class can either bind or connect to a specified endpoint.
    An optional subscription filter can be applied to SUB/XSUB sockets,
    and an optional warm-up can be applied to PUB/XPUB sockets before sending.

    This version uses the standard synchronous zmq API with blocking calls.
    """

    def __init__(
        self,
        socket_type=ZmqSocketType.SUB,
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
            socket_type (int, optional): A valid ZMQ socket type from ZmqSocketType,
                e.g. ZmqSocketType.PUB, ZmqSocketType.SUB, ZmqSocketType.REQ, etc.
                Defaults to ZmqSocketType.SUB.
            path (str, optional): The endpoint to bind or connect to, e.g. "tcp://127.0.0.1:5555".
                Defaults to "tcp://127.0.0.1:5555".
            subscribe_filter (str, optional): The subscription filter for SUB/XSUB sockets.
                Ignored otherwise. Defaults to "" (empty string).
            warmup_timeout (float, optional): If > 0 and the socket is PUB/XPUB, we wait this many
                seconds after `start()` to give subscribers time to connect. Defaults to 0.0.
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

        self._context = zmq.Context()
        self._socket = None
        self._listener_thread = None
        self._is_started = False
        self._is_listening = False

    def _ensure_started(self):
        """
        Ensure that the socket has started.
        """
        if not self._is_started:
            raise RuntimeError("Socket has not started; call '.start()' first")

    def start(self):
        """
        Create the socket and either bind or connect. If the socket is PUB/XPUB
        and warmup_timeout > 0, sleep briefly to let subscribers connect.

        Raises:
            RuntimeError: If the socket is already started.
            RuntimeError: If a ZMQ error occurs, with details about the failure.
        """
        if self._is_started:
            return

        try:
            self._socket = self._context.socket(self.socket_type)
            self._socket.setsockopt(zmq.SNDHWM, self.snd_hwm)
            self._socket.setsockopt(zmq.RCVHWM, self.rcv_hwm)

            if self.bind:
                self._socket.bind(self.path)
            else:
                self._socket.connect(self.path)

            # For SUB/XSUB, apply the subscription filter
            if self.socket_type in (ZmqSocketType.SUB, ZmqSocketType.XSUB):
                self._socket.setsockopt_string(zmq.SUBSCRIBE, self.subscribe_filter)

            self._is_started = True

            # If PUB/XPUB and warmup_timeout is set, give subscribers time to connect
            if (
                self.socket_type in (ZmqSocketType.PUB, ZmqSocketType.XPUB)
                and self.warmup_timeout > 0
            ):
                # These args were provided to maintain 1:1 compatibility with the
                # Async version, however it's not advisable to block the main loop
                # for that long. This change may be reversed in the future, but for now
                # it stays disabled.
                pass
        except zmq.error.ZMQError as e:
            # Clean up resources if socket was created
            if self._socket is not None:
                self._socket.close()
                self._socket = None

            action = "binding to" if self.bind else "connecting to"
            raise RuntimeError(
                f"ZMQ error when {action} {self.path} with socket type {self.socket_type}: {str(e)}"
            ) from e

    def send(self, data: bytes):
        """
        Send data on this socket.

        This method is valid for socket types that support outbound messages,
        such as PUB, PUSH, DEALER, REQ, etc.

        Args:
            data (bytes): The data to send.

        Raises:
            RuntimeError: If the socket is not started.
        """
        self._ensure_started()
        self._socket.send(data, copy=True)

    def recv(self) -> bytes:
        """
        Receive data on this socket (blocking).

        This method is valid for socket types that support inbound messages,
        such as SUB, PULL, DEALER, REP, etc.

        Returns:
            bytes: The data received from the socket.

        Raises:
            RuntimeError: If the socket is not started.
        """
        self._ensure_started()
        msg = self._socket.recv()
        return msg

    def listen(self, on_message: Callable[[bytes], None]):
        """
        Start a background thread that repeatedly calls recv() in a loop
        and invokes on_message(msg) for each incoming message.

        Typically used for SUB sockets or other receiving sockets that want
        to handle messages asynchronously.

        Args:
            on_message (callable[[bytes], None]): A callback function that takes one argument: the received message.

        Raises:
            RuntimeError: If the socket is not started.
            ValueError: If the callback function does not take exactly one bytes argument.
        """
        self._ensure_started()

        if self._listener_thread:
            return

        # Validate the callback function's signature
        sig = inspect.signature(on_message)
        if len(sig.parameters) != 1:
            raise ValueError("Callback function must take exactly one argument.")
        if inspect.iscoroutinefunction(on_message):
            raise ValueError(
                "Invalid function signature; expected a regular function but got an async function."
            )

        # NOTE: Having issues with Cython functions due to their "type name" format
        # compared to Python's "name: type" format. For now, this type check will
        # remain disabled until there is a way to make it Cython compatible.

        # first_param = list(sig.parameters.values())[0]
        # if first_param.annotation is not bytes:
        #     raise ValueError("Callback function must take a bytes argument.")

        self._is_listening = True
        self._listener_thread = threading.Thread(
            target=self._listen_loop, args=(on_message,), daemon=True
        )
        self._listener_thread.start()

    def _listen_loop(self, on_message: Callable[[bytes], None]):
        """
        Blocking loop that continuously receives messages and invokes on_message(msg).
        Runs in a dedicated thread.
        """
        while self._is_started and self._is_listening:
            try:
                msg = self._socket.recv()
                on_message(msg)
            except zmq.error.ZMQError:
                break

    def cancel_listening(self):
        """
        Stop the background listening thread.
        """
        self._is_listening = False
        if self._listener_thread and self._listener_thread.is_alive():
            # We can try closing the socket or do something else to break recv()
            # For now, we rely on .stop() or .close() to break the recv() loop
            # or raise a ZMQError.
            self._listener_thread.join(timeout=1.0)
        self._listener_thread = None

    def stop(self):
        """
        Close the socket and terminate the context.
        """
        self.cancel_listening()
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        self._context.term()
        self._is_started = False


class AsyncZmqConnection:
    """
    An abstract ZeroMQ connection capable of creating and managing
    various socket types (PUB, SUB, REQ, REP, DEALER, ROUTER, etc.).

    This class can bind or connect to a specified endpoint, optionally
    applying a subscription filter (for SUB/XSUB sockets) or running a
    warm-up interval (for PUB/XPUB sockets).
    """

    def __init__(
        self,
        socket_type=ZmqSocketType.SUB,
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
            socket_type (int, optional): A valid ZMQ socket type from zmq.constants.ZmqSocketType,
                e.g. ZmqSocketType.PUB, ZmqSocketType.SUB, ZmqSocketType.REQ, etc.
                Defaults to ZmqSocketType.SUB.
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

    def _ensure_started(self):
        """
        Ensure that the socket has started.
        """
        if not self._is_started:
            raise RuntimeError("Socket has not started; call '.start()' first")

    async def start(self):
        """
        Create the socket and either bind or connect. If the socket is PUB/XPUB
        and warmup_timeout > 0, pause briefly to let subscribers connect.

        Raises:
            RuntimeError: If the socket is already started.
            RuntimeError: If a ZMQ error occurs, with details about the failure.
        """
        if self._is_started:
            return

        try:
            self._socket = self._context.socket(self.socket_type)
            self._socket.setsockopt(zmq.SNDHWM, self.snd_hwm)
            self._socket.setsockopt(zmq.RCVHWM, self.rcv_hwm)

            if self.bind:
                self._socket.bind(self.path)
            else:
                self._socket.connect(self.path)

            # For SUB/XSUB, apply the subscription filter
            if self.socket_type in (ZmqSocketType.SUB, ZmqSocketType.XSUB):
                self._socket.setsockopt_string(zmq.SUBSCRIBE, self.subscribe_filter)

            self._is_started = True

            # If PUB/XPUB and warmup_timeout is set, give subscribers time to connect
            if (
                self.socket_type in (ZmqSocketType.PUB, ZmqSocketType.XPUB)
                and self.warmup_timeout > 0
            ):
                start_ts = time.time()
                while time.time() - start_ts < self.warmup_timeout:
                    await asyncio.sleep(0.1)
        except zmq.error.ZMQError as e:
            # Clean up resources if socket was created
            if self._socket is not None:
                self._socket.close()
                self._socket = None

            action = "binding to" if self.bind else "connecting to"
            raise RuntimeError(
                f"ZMQ error when {action} {self.path} with socket type {self.socket_type}: {str(e)}"
            ) from e

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
        self._ensure_started()
        await self._socket.send(data, copy=True)

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
        self._ensure_started()
        msg = await self._socket.recv()
        return msg

    def listen(self, on_message: Callable[[bytes], None]):
        """
        Start a background task that repeatedly calls recv() and
        invokes on_message(msg) for each incoming message.

        Typically used for SUB sockets or other receiving sockets that want
        to handle messages asynchronously.

        Args:
            on_message (callable[[bytes], None]): A callback function that takes one argument: the received message.

        Raises:
            RuntimeError: If the socket is not started.
            ValueError: If the callback function does not take exactly one bytes argument.
        """
        self._ensure_started()
        if isinstance(self._listen_task, asyncio.Task):
            # Already listening
            return

        # Validate the callback function's signature and that its an async function
        sig = inspect.signature(on_message)
        if len(sig.parameters) != 1:
            raise ValueError(
                f"Invalid function signature; expected exactly one argument but got {len(sig.parameters)}"
            )
        if not inspect.iscoroutinefunction(on_message):
            raise ValueError(
                "Invalid function signature; expected an async function but got a regular function."
            )

        # NOTE: Having issues with Cython functions due to their "type name" format
        # compared to Python's "name: type" format. For now, this type check will
        # remain disabled until there is a way to make it Cython compatible.
        # first_param = list(sig.parameters.values())[0]
        # if first_param.annotation is not bytes:
        #     raise ValueError("Callback function must take a bytes argument.")

        self._listen_task = asyncio.create_task(self._listen_loop(on_message))

    async def _listen_loop(self, on_message: Callable[[bytes], None]):
        """
        Internal coroutine that continuously receives messages and invokes on_message(msg).
        """
        while self._is_started:
            try:
                msg = await self._socket.recv()
                await on_message(msg)
            except asyncio.CancelledError:
                return
            except Exception:
                # We expect any exceptions caused in the on_message()
                # to be handled/logged properly by the user.
                pass

    def cancel_listening(self):
        """
        Cancel the background listening task, stopping any further automatic reception.
        """
        if self._listen_task and not self._listen_task.done():
            self._listen_task.cancel()
            self._listen_task = None

    def stop(self):
        """
        Close the socket and terminate the context, stopping any I/O activity.
        """
        self.cancel_listening()
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        self._context.term()
        self._is_started = False
