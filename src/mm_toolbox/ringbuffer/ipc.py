"""Inter-process communication using ZMQ for ring buffer data."""

from collections.abc import AsyncIterator, Iterable

import zmq
import zmq.asyncio as azmq
from msgspec import Struct
from zmq.constants import SocketType as ZmqSocketType

type Item = bytes | memoryview | bytearray


class IPCRingBufferConfig(Struct):
    """Configuration for IPC ringbuffer."""

    path: str
    backlog: int
    num_producers: int
    num_consumers: int

    def __post_init__(self) -> None:
        """Validate IPC configuration parameters."""
        if not self.path.startswith("ipc://"):
            raise ValueError(f"Invalid path; expected 'ipc://' but got {self.path}")
        if self.backlog <= 0:
            raise ValueError(f"Invalid backlog; expected >0 but got {self.backlog}")
        if self.num_producers <= 0 or self.num_consumers <= 0:
            raise ValueError("num_producers and num_consumers must be > 0")
        if self.num_producers > 1 and self.num_consumers > 1:
            raise ValueError("MPMC is not supported; set one side to 1 or use a proxy")

    @classmethod
    def default(cls) -> "IPCRingBufferConfig":
        """Create default IPC configuration."""
        return cls(
            path="ipc:///tmp/ringbuffer",
            backlog=2**16,
            num_producers=1,
            num_consumers=1,
        )

    def should_producer_bind(self) -> bool:
        """Determine if producer should bind to socket based on topology."""
        if self.num_producers == 1 and self.num_consumers > 1:
            return True  # SPMC
        if self.num_consumers == 1 and self.num_producers > 1:
            return False  # MPSC
        return True  # SPSC


class IPCRingBufferProducer:
    """Producer built for sending bytes across processes. Wraps ZMQ.
    Used in conjunction with IPCRingBufferConsumer.
    """

    def __init__(self, config: IPCRingBufferConfig) -> None:
        """Initialize IPC producer with configuration."""
        self._path = config.path
        self._backlog = config.backlog
        self._context = zmq.Context()
        self._socket = self._context.socket(ZmqSocketType.PUSH)
        if config.should_producer_bind():
            self._socket.bind(self._path)
        else:
            self._socket.connect(self._path)
        self._socket.setsockopt(zmq.SNDHWM, self._backlog)
        self._is_started = True

    def insert(self, item: Item, copy: bool = True) -> None:
        """Insert an item into the ring buffer."""
        self.__enforce_producer_started()
        self._socket.send(item, copy=copy)

    def insert_batch(self, items: Iterable[Item], copy: bool = True) -> None:
        """Insert a batch of items into the ring buffer."""
        self.__enforce_producer_started()
        for item in items:
            self.insert(item, copy=copy)

    def insert_packed(self, batch: Iterable[Item], copy: bool = True) -> None:
        """Insert a batch of items into the ring buffer, packed into a single message."""
        self.__enforce_producer_started()
        parts: list[memoryview] = []
        total_size = 0
        for item in batch:
            mv = memoryview(item)
            parts.append(mv)
            total_size += 4 + len(mv)
        if total_size == 0:
            return
        buf = bytearray(total_size)
        offset = 0
        for mv in parts:
            length = len(mv)
            buf[offset : offset + 4] = length.to_bytes(4, "little")
            offset += 4
            buf[offset : offset + length] = mv
            offset += length
        self._socket.send(buf, copy=copy)

    def stop(self) -> None:
        """Stop the producer and close the socket."""
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        self._context.term()
        self._is_started = False

    def __enforce_producer_started(self) -> None:
        """Enforce that the producer is started."""
        if not self._is_started:
            raise RuntimeError("Producer not started")


class IPCRingBufferConsumer:
    """Consumer built for receiving bytes across processes. Wraps ZMQ.
    Used in conjunction with IPCRingBufferProducer.
    """

    def __init__(self, config: IPCRingBufferConfig) -> None:
        """Initialize IPC consumer with configuration."""
        self._path = config.path
        self._backlog = config.backlog
        self._context = zmq.Context()
        self._socket = self._context.socket(ZmqSocketType.PULL)
        self._socket.setsockopt(zmq.RCVHWM, self._backlog)
        self._asocket = azmq.Socket.shadow(self._socket)
        if config.should_producer_bind():
            self._socket.connect(self._path)
        else:
            self._socket.bind(self._path)
        self._is_started = True

    def consume(self) -> Item:
        """Consume an item from the ring buffer, blocking until one is available."""
        if not self._is_started:
            raise RuntimeError("Consumer not started")
        return self._socket.recv()

    def consume_all(self) -> list[Item]:
        """Drain the ring buffer, non-blocking."""
        self.__enforce_consumer_started()

        items: list[Item] = []
        while True:
            try:
                msg = self._socket.recv(flags=zmq.DONTWAIT)
                items.append(msg)
            except zmq.Again:
                break
        return items

    def consume_packed(self) -> list[Item]:
        """Consume a packed batch of items from the ring buffer, blocking until one is available."""
        self.__enforce_consumer_started()
        items: list[Item] = []
        buf = self._socket.recv()
        buf_len = len(buf)
        buf_mv = memoryview(buf)
        offset = 0
        while offset + 4 <= buf_len:
            length = int.from_bytes(buf_mv[offset : offset + 4], "little")
            offset += 4
            if offset + length > buf_len:
                raise ValueError("Corrupted packed message: length exceeds buffer")
            item = buf_mv[offset : offset + length]
            items.append(item)
            offset += length
        return items

    async def aconsume(self) -> Item:
        """Mirror of consume, but non-blocking."""
        self.__enforce_consumer_started()
        return await self._asocket.recv()

    async def aconsume_packed(self) -> list[Item]:
        """Mirror of consume_packed, but non-blocking."""
        self.__enforce_consumer_started()
        buf = await self._asocket.recv()
        buf_len = len(buf)
        buf_mv = memoryview(buf)
        offset = 0
        items: list[Item] = []
        while offset + 4 <= buf_len:
            length = int.from_bytes(buf_mv[offset : offset + 4], "little")
            offset += 4
            if offset + length > buf_len:
                raise ValueError("Corrupted packed message: length exceeds buffer")
            item = buf_mv[offset : offset + length]
            items.append(item)
            offset += length
        return items

    def stop(self) -> None:
        """Stop the consumer and close the socket."""
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        if self._asocket is not None:
            self._asocket.close()
            self._asocket = None
        self._context.term()
        self._is_started = False

    def __aiter__(self) -> AsyncIterator[Item]:
        """Async iterator for the consumer."""
        return self

    async def __anext__(self) -> Item:
        """Async next for the consumer."""
        return await self.aconsume()

    def __enforce_consumer_started(self) -> None:
        """Enforce that the consumer is started."""
        if not self._is_started:
            raise RuntimeError("Consumer not started")
