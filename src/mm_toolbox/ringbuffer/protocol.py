from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Protocol, TypeVar

T = TypeVar("T")


class RingBufferProtocol(Protocol[T]):
    """Core protocol shared by all ringbuffer implementations."""

    def insert(self, item: T) -> bool:
        """Insert a single item into the ringbuffer.

        Returns:
            True if the insertion succeeded.
        """
        ...

    def insert_batch(self, items: list[T]) -> bool:
        """Insert multiple items into the ringbuffer.

        Returns:
            True if the batch insertion succeeded.
        """
        ...

    def consume(self) -> T:
        """Remove and return the oldest item from the ringbuffer."""
        ...

    def consume_all(self) -> list[T]:
        """Remove and return all currently available items."""
        ...

    def consume_iterable(self) -> Iterator[T]:
        """Yield items from the ringbuffer in FIFO order."""
        ...

    def unwrapped(self) -> list[T]:
        """Return all logical contents without consuming."""
        ...

    def contains(self, item: T) -> bool:
        """Check if item is present in the buffer."""
        ...

    def is_empty(self) -> bool:
        """Check if the buffer is empty."""
        ...

    def is_full(self) -> bool:
        """Check if the buffer is full."""
        ...

    def clear(self) -> None:
        """Clear the buffer."""
        ...

    def peekleft(self) -> T:
        """Return the oldest item without removing it."""
        ...

    def peekright(self) -> T:
        """Return the newest item without removing it."""
        ...

    @property
    def latest_insert_time_ns(self) -> int:
        """Return the timestamp (ns) of the latest successful insert."""
        ...

    @property
    def latest_consume_time_ns(self) -> int:
        """Return the timestamp (ns) of the latest successful consume."""
        ...

    def __len__(self) -> int:
        """Return the number of items currently in the ringbuffer."""
        ...


class SupportsAsyncConsume(Protocol[T]):
    """Mixin protocol for ringbuffers that support async consumption."""

    async def aconsume(self) -> T:
        """Asynchronously remove and return the oldest item."""
        ...

    async def aconsume_iterable(self) -> AsyncIterator[T]:
        """Asynchronously yield items from the ringbuffer in FIFO order."""
        ...


class AsyncRingBufferProtocol(RingBufferProtocol[T], SupportsAsyncConsume[T], Protocol):
    """Protocol for ringbuffers that support both sync and async consumption."""


class RingBufferProducerProtocol(Protocol[T]):
    """Protocol for the producer side of a split ringbuffer design."""

    def insert(self, item: T) -> bool:
        """Insert a single item into the ringbuffer.

        Returns:
            True if the insertion succeeded.
        """
        ...

    def insert_batch(self, items: list[T]) -> bool:
        """Insert multiple items into the ringbuffer.

        Returns:
            True if the batch insertion succeeded.
        """
        ...

    def is_empty(self) -> bool:
        """Check if the buffer is empty."""
        ...

    def is_full(self) -> bool:
        """Check if the buffer is full."""
        ...

    def __len__(self) -> int:
        """Return the number of items currently in the ringbuffer."""
        ...


class RingBufferConsumerProtocol(Protocol[T]):
    """Protocol for the consumer side of a split ringbuffer design."""

    def consume(self) -> T:
        """Remove and return the oldest item from the ringbuffer."""
        ...

    def consume_iterable(self) -> Iterator[T]:
        """Yield items from the ringbuffer in FIFO order."""
        ...

    def unwrapped(self) -> list[T]:
        """Return all logical contents without consuming."""
        ...

    def contains(self, item: T) -> bool:
        """Check if item is present in the buffer."""
        ...

    def is_empty(self) -> bool:
        """Check if the buffer is empty."""
        ...

    def is_full(self) -> bool:
        """Check if the buffer is full."""
        ...

    def clear(self) -> None:
        """Clear the buffer."""
        ...

    def peekleft(self) -> T:
        """Return the oldest item without removing it."""
        ...

    def peekright(self) -> T:
        """Return the newest item without removing it."""
        ...

    def __len__(self) -> int:
        """Return the number of items currently in the ringbuffer."""
        ...


class AsyncRingBufferConsumerProtocol(
    RingBufferConsumerProtocol[T], SupportsAsyncConsume[T], Protocol
):
    """Protocol for split consumers that also support async methods."""

    ...
