"""Fail-fast checks for the examples' bounded market-data transports.

These examples do not implement recovery or guarantee loss-free throughput.
Restart the whole example after a loss; a new snapshot alone cannot repair
missing trades or the resulting candles and moving averages.
"""

from __future__ import annotations

import asyncio
import json
from collections import deque
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol


class _LossAwareClient(Protocol):
    """The examples require observability beyond the public client protocol."""

    def send_data(self, msg: bytes, only_fastest: bool = True) -> None: ...

    def get_dropped_messages(self) -> int: ...

    def get_connection_count(self) -> int: ...

    def __aiter__(self) -> AsyncIterator[bytes]: ...


class FeedIntegrityError(RuntimeError):
    """The example can no longer establish complete market-data continuity."""


@dataclass(frozen=True, slots=True)
class FeedSubscription:
    """One Binance SUBSCRIBE request for a logical feed."""

    streams: tuple[str, ...]
    request_id: int

    def __post_init__(self) -> None:
        if not self.streams or any(not stream for stream in self.streams):
            raise ValueError("A feed subscription needs at least one stream name")
        if (
            isinstance(self.request_id, bool)
            or not isinstance(self.request_id, int)
            or self.request_id < 0
        ):
            raise ValueError("A Binance request id must be an unsigned integer")


_STARTUP_STAGE_MAX_MESSAGES = 4096
_STARTUP_STAGE_MAX_BYTES = 16 * 1024 * 1024
_CONTROL_PREFIXES = (
    b'"result"',
    b'"id"',
    b'"code"',
    b'"msg"',
    b'"method"',
    b'"params"',
    b'"error"',
)


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
    *,
    connection_counts: Sequence[int] | None = None,
    subscriptions: Sequence[FeedSubscription] | None = None,
    subscription_timeout_s: float = 5.0,
    on_ready: Callable[[], None] | None = None,
    ready_event: asyncio.Event | None = None,
) -> None:
    """Subscribe and consume fully opened feeds as one all-or-nothing group.

    Subscription ACKs are checked once per logical client. A ``WsPool`` sends
    each request to every socket, while its shared receive buffer deduplicates
    the identical responses; this does not assert one ACK per socket.
    """

    if not feeds:
        raise ValueError("At least one feed is required")
    if connection_counts is None:
        connection_counts = [1] * len(feeds)
    if len(connection_counts) != len(feeds) or any(
        count <= 0 for count in connection_counts
    ):
        raise ValueError("Each feed must have a positive expected connection count")
    if subscriptions is not None:
        if len(subscriptions) != len(feeds):
            raise ValueError("Each feed must have exactly one subscription")
        if subscription_timeout_s <= 0:
            raise ValueError("subscription_timeout_s must be > 0")
        request_ids = [subscription.request_id for subscription in subscriptions]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("Feed subscription request ids must be unique")

    for (client, name, _), connection_count in zip(
        feeds, connection_counts, strict=True
    ):
        assert_feed_connected(client, name, connection_count)

    def subscription_payload(subscription: FeedSubscription) -> bytes:
        return json.dumps(
            {
                "method": "SUBSCRIBE",
                "params": list(subscription.streams),
                "id": subscription.request_id,
            },
            separators=(",", ":"),
        ).encode("utf-8")

    def is_control_candidate(message: bytes) -> bool:
        start = 0
        while start < len(message) and message[start] in b" \t\r\n":
            start += 1
        if start >= len(message) or message[start] != ord("{"):
            return False
        start += 1
        while start < len(message) and message[start] in b" \t\r\n":
            start += 1
        return message.startswith(_CONTROL_PREFIXES, start)

    def inspect_control_message(message: bytes, request_id: int, name: str) -> bool:
        try:
            payload = json.loads(message)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise FeedIntegrityError(f"{name}: malformed Binance control message")
        if not isinstance(payload, dict):
            raise FeedIntegrityError(f"{name}: malformed Binance control message")

        if "code" in payload or "msg" in payload:
            if "code" not in payload or "msg" not in payload:
                raise FeedIntegrityError(f"{name}: malformed Binance error response")
            raise FeedIntegrityError(
                f"{name}: Binance rejected subscription "
                f"({payload.get('code')}): {payload.get('msg')}"
            )

        if "id" in payload or "result" in payload:
            if set(payload) != {"result", "id"}:
                raise FeedIntegrityError(
                    f"{name}: malformed or unsuccessful Binance subscription ACK"
                )
            response_id = payload["id"]
            if isinstance(response_id, bool) or not isinstance(response_id, int):
                raise FeedIntegrityError(f"{name}: invalid Binance subscription ACK id")
            if response_id != request_id:
                raise FeedIntegrityError(
                    f"{name}: unexpected Binance response id {response_id}; "
                    f"expected {request_id}"
                )
            if payload["result"] is not None:
                raise FeedIntegrityError(
                    f"{name}: malformed or unsuccessful Binance subscription ACK"
                )
            return True

        raise FeedIntegrityError(f"{name}: unexpected Binance control message")

    def stage_startup_message(
        staged: deque[bytes], staged_bytes: int, message: bytes, name: str
    ) -> int:
        # Keep the temporary stage within the receive-buffer defaults. Never
        # evict silently: an unexpected pre-ACK burst that exceeds either
        # budget fails the complete startup group.
        if (
            len(message) > _STARTUP_STAGE_MAX_BYTES
            or len(staged) >= _STARTUP_STAGE_MAX_MESSAGES
        ):
            raise FeedIntegrityError(
                f"{name}: unexpected pre-ACK data exceeded startup staging limit"
            )
        staged_bytes += len(message)
        if staged_bytes > _STARTUP_STAGE_MAX_BYTES:
            raise FeedIntegrityError(
                f"{name}: unexpected pre-ACK data exceeded startup staging limit"
            )
        staged.append(message)
        return staged_bytes

    async def wait_for_ack(
        client: _LossAwareClient,
        name: str,
        connection_count: int,
        subscription: FeedSubscription,
        started: asyncio.Event,
    ) -> deque[bytes]:
        staged: deque[bytes] = deque()
        staged_bytes = 0
        started.set()
        async for message in feed_messages(client, name, connection_count):
            if is_control_candidate(message) and inspect_control_message(
                message, subscription.request_id, name
            ):
                return staged
            staged_bytes = stage_startup_message(staged, staged_bytes, message, name)
        raise FeedIntegrityError(
            f"{name}: receive stream ended before subscription ACK"
        )

    async def consume(client, name, connection_count, process, request_id):
        async for message in feed_messages(client, name, connection_count):
            # Pools broadcast one request to all sockets. Identical ACKs are
            # normally deduplicated, but a delayed duplicate can arrive after
            # the logical barrier; suppress valid ACKs and fail every other
            # control response instead of handing it to a market decoder.
            if request_id is not None and is_control_candidate(message):
                inspect_control_message(message, request_id, name)
                continue
            process(message)

    monitor = asyncio.create_task(
        monitor_feeds(
            [
                (client, name, count)
                for (client, name, _), count in zip(
                    feeds, connection_counts, strict=True
                )
            ]
        )
    )
    tasks: list[asyncio.Task] = [monitor]
    failure: BaseException | None = None
    try:
        staged_messages: list[deque[bytes]] = [deque() for _ in feeds]
        if subscriptions is not None:
            started = [asyncio.Event() for _ in feeds]
            ack_tasks = [
                asyncio.create_task(
                    wait_for_ack(client, name, connection_count, subscription, event)
                )
                for (client, name, _), connection_count, subscription, event in zip(
                    feeds, connection_counts, subscriptions, started, strict=True
                )
            ]
            tasks.extend(ack_tasks)
            await asyncio.gather(*(event.wait() for event in started))
            for task in ack_tasks:
                if task.done() and not task.cancelled():
                    task.result()
            for (client, _, _), subscription in zip(feeds, subscriptions, strict=True):
                client.send_data(subscription_payload(subscription), only_fastest=False)
            deadline = asyncio.get_running_loop().time() + subscription_timeout_s
            ack_results: dict[asyncio.Task, deque[bytes]] = {}
            for task in ack_tasks:
                if task.done():
                    ack_results[task] = task.result()
            pending_acks = [task for task in ack_tasks if task not in ack_results]
            while pending_acks:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                done, _ = await asyncio.wait(
                    [*pending_acks, monitor],
                    timeout=remaining,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if monitor in done:
                    monitor.result()
                for task in done:
                    if task in pending_acks:
                        ack_results[task] = task.result()
                pending_acks = [
                    task for task in pending_acks if task not in ack_results
                ]
                if not done:
                    break
            if pending_acks:
                waiting = [
                    name
                    for (_, name, _), task in zip(feeds, ack_tasks, strict=True)
                    if task in pending_acks
                ]
                raise FeedIntegrityError(
                    f"Timed out after {subscription_timeout_s:g}s waiting for "
                    f"Binance subscription ACK from {', '.join(waiting)}"
                )
            staged_messages = [ack_results[task] for task in ack_tasks]
            tasks = [monitor]

        for (client, name, _), connection_count in zip(
            feeds, connection_counts, strict=True
        ):
            assert_feed_connected(client, name, connection_count)
        if on_ready is not None:
            on_ready()
        for (client, name, process), messages, connection_count in zip(
            feeds, staged_messages, connection_counts, strict=True
        ):
            for message in messages:
                assert_feed_connected(client, name, connection_count)
                process(message)
        if ready_event is not None:
            ready_event.set()

        consumer_tasks = [
            asyncio.create_task(
                consume(
                    client,
                    name,
                    connection_count,
                    process,
                    subscription.request_id if subscriptions is not None else None,
                )
            )
            for (client, name, process), connection_count, subscription in zip(
                feeds,
                connection_counts,
                subscriptions if subscriptions is not None else [None] * len(feeds),
                strict=True,
            )
        ]
        tasks.extend(consumer_tasks)
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
        raise RuntimeError("A feed task stopped unexpectedly")
    except BaseException as exc:
        failure = exc
        raise
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if failure is None:
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
