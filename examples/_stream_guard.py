"""Fail-fast checks for the examples' bounded market-data transports.

These examples do not implement recovery or guarantee loss-free throughput.
Restart the whole example after a loss; a new snapshot alone cannot repair
missing trades or the resulting candles and moving averages.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any, Protocol


class _LossAwareClient(Protocol):
    """The examples require observability beyond the public client protocol."""

    def get_dropped_messages(self) -> int: ...

    def get_connection_count(self) -> int: ...

    def __aiter__(self) -> AsyncIterator[bytes]: ...


class FeedIntegrityError(RuntimeError):
    """The example can no longer establish complete market-data continuity."""


def assert_no_receive_loss(client: _LossAwareClient, name: str) -> None:
    """Reject every receive overwrite, including one during initial startup."""
    dropped = client.get_dropped_messages()
    if dropped:
        raise FeedIntegrityError(
            f"{name}: receive buffer overwrote {dropped} payloads; restart the example"
        )


def assert_feed_connected(
    client: _LossAwareClient, name: str, connections: int = 1
) -> None:
    """Reject loss and disconnected continuity; counters start at zero."""
    assert_no_receive_loss(client, name)
    connected = client.get_connection_count()
    if connected != connections:
        raise FeedIntegrityError(
            f"{name}: expected {connections} live connections, got {connected}; "
            "continuity is unproven, restart the example"
        )


async def feed_messages(
    client: _LossAwareClient, name: str, connections: int = 1
) -> AsyncIterator[bytes]:
    """Check validity before the first and every subsequent state update."""
    assert_feed_connected(client, name, connections)
    async for message in client:
        assert_feed_connected(client, name, connections)
        yield message
    raise FeedIntegrityError(f"{name}: receive stream ended unexpectedly")


async def monitor_feeds(
    feeds: Sequence[tuple[_LossAwareClient, str, int]],
) -> None:
    """Detect idle disconnects too, without waiting for another payload."""
    while True:
        for client, name, connections in feeds:
            assert_feed_connected(client, name, connections)
        await asyncio.sleep(0.05)


async def run_feeds(
    feeds: Sequence[tuple[_LossAwareClient, str, Callable[[bytes], None]]],
) -> None:
    """Consume fully opened feeds and propagate the first failure to the owner."""

    async def consume(client, name, process):
        async for message in feed_messages(client, name):
            process(message)

    tasks = [
        asyncio.create_task(consume(client, name, process))
        for client, name, process in feeds
    ]
    tasks.append(
        asyncio.create_task(
            monitor_feeds([(client, name, 1) for client, name, _ in feeds])
        )
    )
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for client, name, _ in feeds:
            assert_no_receive_loss(client, name)


def assert_no_transport_loss(transport: Any, name: str) -> None:
    """A fresh example-owned mapping must never have overwritten a record."""
    dropped = transport.dropped_count
    if dropped:
        raise FeedIntegrityError(
            f"{name}: SHM transport overwrote {dropped} records; restart the example"
        )


def forward_message(producer: Any, payload: bytes, name: str) -> None:
    """Retain overwrite-oldest transport semantics, but stop this application."""
    assert_no_transport_loss(producer, name)
    if not producer.insert(payload):
        raise FeedIntegrityError(f"{name}: SHM transport rejected a market update")
    assert_no_transport_loss(producer, name)


def consume_batch(consumer: Any, name: str) -> list[bytes]:
    """Detect overwrite during the native read before applying the batch."""
    assert_no_transport_loss(consumer, name)
    messages = consumer.consume_all()
    assert_no_transport_loss(consumer, name)
    return messages
