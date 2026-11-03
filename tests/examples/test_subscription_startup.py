"""The example feeds publish data only after every subscription is ready."""

from __future__ import annotations

import asyncio
import json

import pytest
from websockets.asyncio.server import serve

from examples._stream_guard import FeedIntegrityError, FeedSubscription, run_feeds
from mm_toolbox.websocket import WsConnectionConfig, WsPool, WsPoolConfig, WsSingle


@pytest.mark.asyncio
async def test_delayed_handshake_and_pre_ack_burst_do_not_publish_early():
    burst_sent = asyncio.Event()
    release_slow_handshake = asyncio.Event()
    fast_ack_sent = asyncio.Event()
    slow_subscribe_seen = asyncio.Event()
    release_slow_ack = asyncio.Event()
    processed_burst = asyncio.Event()
    processed: list[int | str] = []
    burst_count = 256

    async def delay_slow_handshake(connection, request):
        if request.path == "/slow":
            await burst_sent.wait()
            await release_slow_handshake.wait()
        return None

    async def handler(websocket):
        path = websocket.request.path
        if path == "/fast":
            for sequence in range(burst_count):
                await websocket.send(json.dumps({"event": sequence}))
                await asyncio.sleep(0)
            burst_sent.set()

        request = json.loads(await websocket.recv())
        assert request["method"] == "SUBSCRIBE"
        if path == "/slow":
            slow_subscribe_seen.set()
            await release_slow_ack.wait()
            ack = f'{{ "result" : null, "id" : {request["id"]} }}'
        else:
            ack = json.dumps(
                {"result": None, "id": request["id"]}, separators=(",", ":")
            )
        await websocket.send(ack)
        if path == "/fast":
            fast_ack_sent.set()
        await websocket.wait_closed()

    async with serve(
        handler,
        "127.0.0.1",
        0,
        process_request=delay_slow_handshake,
    ) as server:
        port = server.sockets[0].getsockname()[1]
        fast = WsSingle(
            WsConnectionConfig.default(
                f"ws://127.0.0.1:{port}/fast", auto_reconnect=False
            )
        )
        slow = WsSingle(
            WsConnectionConfig.default(
                f"ws://127.0.0.1:{port}/slow", auto_reconnect=False
            )
        )

        def process(message: bytes) -> None:
            processed.append(json.loads(message)["event"])
            if len(processed) == burst_count + 1:
                processed_burst.set()

        async def start_group() -> None:
            async with fast, slow:
                assert fast.get_connection_count() == slow.get_connection_count() == 1
                await run_feeds(
                    [(fast, "fast", process), (slow, "slow", process)],
                    subscriptions=[
                        FeedSubscription(("stream@fast",), 11),
                        FeedSubscription(("stream@slow",), 12),
                    ],
                    on_ready=lambda: processed.append("ready"),
                    subscription_timeout_s=2.0,
                )

        task = asyncio.create_task(start_group())
        try:
            await asyncio.wait_for(burst_sent.wait(), timeout=2.0)
            # The second HTTP upgrade is intentionally held while the first
            # already-open idle socket receives more than the old 128-frame
            # ring could retain. No application reader has been started yet.
            await asyncio.sleep(0.05)
            assert processed == []
            assert fast.get_dropped_messages() == 0

            release_slow_handshake.set()
            await asyncio.wait_for(fast_ack_sent.wait(), timeout=2.0)
            await asyncio.wait_for(slow_subscribe_seen.wait(), timeout=2.0)
            assert processed == []

            release_slow_ack.set()
            await asyncio.wait_for(processed_burst.wait(), timeout=2.0)
            assert processed == ["ready", *range(burst_count)]
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            release_slow_handshake.set()
            release_slow_ack.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            fast.close()
            slow.close()

    assert processed[0] == "ready"
    assert processed[1:] == list(range(burst_count))
    assert fast.get_dropped_messages() == 0


class _NackFeed:
    def __init__(self, response: dict | None):
        self.response = response
        self.connected = 1
        self.dropped = 0
        self.queue: asyncio.Queue[bytes] = asyncio.Queue()
        self.cancelled = asyncio.Event()
        self.broadcast_modes: list[bool] = []

    def get_dropped_messages(self):
        return self.dropped

    def get_connection_count(self):
        return self.connected

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return await self.queue.get()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise

    def send_data(self, message: bytes, only_fastest: bool = True):
        self.broadcast_modes.append(only_fastest)
        request = json.loads(message)
        if self.response is not None:
            self.queue.put_nowait(
                json.dumps(
                    {**self.response, "id": request["id"]}, separators=(",", ":")
                ).encode()
            )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response,match",
    [
        ({"code": 2, "msg": "invalid subscription"}, "rejected subscription"),
        ({"result": [], "extra": True}, "malformed or unsuccessful"),
    ],
)
async def test_subscription_nack_fails_group_and_cancels_sibling_readers(
    response, match
):
    rejecting, waiting = _NackFeed(response), _NackFeed(None)
    processed: list[bytes] = []
    ready = asyncio.Event()

    with pytest.raises(FeedIntegrityError, match=match):
        await asyncio.wait_for(
            run_feeds(
                [
                    (rejecting, "rejecting", processed.append),
                    (waiting, "waiting", processed.append),
                ],
                subscriptions=[
                    FeedSubscription(("stream@reject",), 1),
                    FeedSubscription(("stream@wait",), 2),
                ],
                ready_event=ready,
                subscription_timeout_s=1.0,
            ),
            timeout=1.0,
        )

    assert processed == []
    assert not ready.is_set()
    assert waiting.cancelled.is_set()
    assert rejecting.broadcast_modes == [False]
    assert waiting.broadcast_modes == [False]


@pytest.mark.asyncio
@pytest.mark.parametrize("ack_id", [True, 1.0])
async def test_subscription_ack_requires_exact_integer_request_id(ack_id):
    wrong_id, waiting = _NackFeed({"result": None}), _NackFeed(None)
    processed: list[bytes] = []

    # bool and float compare equal to integer 1 in Python; neither can satisfy it.
    wrong_id.send_data = lambda message, only_fastest=True: wrong_id.queue.put_nowait(
        json.dumps({"result": None, "id": ack_id}).encode()
    )
    with pytest.raises(FeedIntegrityError, match="invalid Binance subscription ACK id"):
        await run_feeds(
            [
                (wrong_id, "wrong-id", processed.append),
                (waiting, "waiting", processed.append),
            ],
            subscriptions=[
                FeedSubscription(("stream@wrong",), 1),
                FeedSubscription(("stream@wait",), 2),
            ],
        )
    assert processed == []
    assert waiting.cancelled.is_set()


@pytest.mark.asyncio
async def test_missing_subscription_ack_times_out_and_cancels_group():
    first, second = _NackFeed(None), _NackFeed(None)
    processed: list[bytes] = []

    with pytest.raises(FeedIntegrityError, match="Timed out.*first, second"):
        await run_feeds(
            [(first, "first", processed.append), (second, "second", processed.append)],
            subscriptions=[
                FeedSubscription(("stream@first",), 1),
                FeedSubscription(("stream@second",), 2),
            ],
            subscription_timeout_s=0.01,
        )

    assert processed == []
    assert first.cancelled.is_set()
    assert second.cancelled.is_set()


@pytest.mark.asyncio
async def test_pool_late_duplicate_ack_is_suppressed_after_hash_history_rolls():
    # Exercise the same eviction boundary with a small history. Scanning the
    # default 16k history for every message is quadratic under Cython tracing.
    hash_capacity = 256
    market_count = hash_capacity + 1
    connection_index = 0
    release_slow_ack = asyncio.Event()
    market_burst_done = asyncio.Event()
    late_market_message = asyncio.Event()
    delivered: list[int | str] = []

    async def handler(websocket):
        nonlocal connection_index
        index = connection_index
        connection_index += 1
        request = json.loads(await websocket.recv())
        assert request["method"] == "SUBSCRIBE"
        ack = f'{{ "result" : null, "id" : {request["id"]} }}'
        if index == 0:
            await websocket.send(ack)
            for sequence in range(market_count):
                await websocket.send(json.dumps({"event": sequence}))
                if sequence % 256 == 0:
                    await asyncio.sleep(0)
        else:
            await release_slow_ack.wait()
            # Identical bytes would be deduplicated before the history rolls.
            await websocket.send(ack)
            await websocket.send(json.dumps({"event": "after-late-ack"}))
        await websocket.wait_closed()

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        pool = await WsPool.new(
            WsConnectionConfig.default(f"ws://127.0.0.1:{port}", auto_reconnect=False),
            pool_config=WsPoolConfig(
                num_connections=2,
                evict_interval_s=60,
                hash_capacity=hash_capacity,
            ),
        )

        def process(message: bytes) -> None:
            event = json.loads(message)["event"]
            delivered.append(event)
            if event == market_count - 1:
                market_burst_done.set()
            elif event == "after-late-ack":
                late_market_message.set()

        async with pool:
            task = asyncio.create_task(
                run_feeds(
                    [(pool, "pool-feed", process)],
                    connection_counts=[2],
                    subscriptions=[FeedSubscription(("stream@pool",), 21)],
                    on_ready=lambda: delivered.append("ready"),
                )
            )
            late_waiter = asyncio.create_task(late_market_message.wait())
            try:
                await asyncio.wait_for(market_burst_done.wait(), timeout=10.0)
                assert delivered[:2] == ["ready", 0]
                assert (
                    sum(isinstance(event, int) for event in delivered) == market_count
                )
                release_slow_ack.set()
                done, _ = await asyncio.wait(
                    [task, late_waiter],
                    timeout=5.0,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if task in done:
                    task.result()
                    pytest.fail("Feed runner stopped before the post-ACK market frame")
                assert late_waiter in done
                assert delivered[-1] == "after-late-ack"
                assert (
                    sum(isinstance(event, int) for event in delivered) == market_count
                )
            finally:
                release_slow_ack.set()
                late_waiter.cancel()
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, late_waiter, return_exceptions=True)


@pytest.mark.asyncio
async def test_late_subscription_nack_fails_and_never_reaches_market_processor():
    received: list[int] = []

    class LateNackFeed(_NackFeed):
        def send_data(self, message: bytes, only_fastest: bool = True):
            request_id = json.loads(message)["id"]
            self.queue.put_nowait(
                json.dumps({"result": None, "id": request_id}).encode()
            )
            self.queue.put_nowait(json.dumps({"event": 1}).encode())
            self.queue.put_nowait(
                json.dumps(
                    {"code": 2, "msg": "late rejection", "id": request_id}
                ).encode()
            )

    feed = LateNackFeed(None)

    def process(message: bytes) -> None:
        received.append(json.loads(message)["event"])

    with pytest.raises(
        FeedIntegrityError, match="rejected subscription.*late rejection"
    ):
        await asyncio.wait_for(
            run_feeds(
                [(feed, "pool-feed", process)],
                subscriptions=[FeedSubscription(("stream@pool",), 22)],
            ),
            timeout=1.0,
        )
    assert received == [1]


@pytest.mark.asyncio
async def test_unexpected_pre_ack_burst_fails_without_silent_eviction():
    class BurstFeed(_NackFeed):
        def send_data(self, message: bytes, only_fastest: bool = True):
            request = json.loads(message)
            for sequence in range(4097):
                self.queue.put_nowait(json.dumps({"event": sequence}).encode())
            self.queue.put_nowait(
                json.dumps({"result": None, "id": request["id"]}).encode()
            )

    burst, sibling = BurstFeed(None), _NackFeed(None)
    processed: list[bytes] = []

    with pytest.raises(FeedIntegrityError, match="startup staging limit"):
        await run_feeds(
            [
                (burst, "burst", processed.append),
                (sibling, "sibling", processed.append),
            ],
            subscriptions=[
                FeedSubscription(("stream@burst",), 1),
                FeedSubscription(("stream@sibling",), 2),
            ],
        )
    assert processed == []
    assert sibling.cancelled.is_set()
