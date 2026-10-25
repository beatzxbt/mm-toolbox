"""Examples stop before applying data whose continuity is no longer known."""

import asyncio
from types import SimpleNamespace

import msgspec
import pytest

from examples import (
    binance_stream,
    candle_types_demo,
    moving_averages_demo,
    shm_mpsc_pipeline,
)
from examples._stream_guard import (
    FeedIntegrityError,
    consume_batch,
    forward_message,
    run_feeds,
)
from mm_toolbox.ringbuffer.shm import (
    ShmMpscConsumer,
    ShmMpscProducer,
    ShmSpscConsumer,
    ShmSpscProducer,
)


class FakeFeed:
    def __init__(self):
        self.dropped = 0
        self.connected = 1
        self.queue = asyncio.Queue()
        self.waiting = asyncio.Event()
        self.cancelled = asyncio.Event()

    def get_dropped_messages(self):
        return self.dropped

    def get_connection_count(self):
        return self.connected

    def __aiter__(self):
        return self

    async def __anext__(self):
        self.waiting.set()
        try:
            return await self.queue.get()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


@pytest.mark.asyncio
async def test_first_startup_overwrite_prevents_any_market_update():
    feed = FakeFeed()
    feed.dropped = 1
    feed.queue.put_nowait(b"latest")
    received = []
    with pytest.raises(FeedIntegrityError, match="overwrote 1"):
        await run_feeds([(feed, "trades", received.append)])
    assert received == []


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["overwrite", "disconnect", "decode"])
async def test_running_feed_failure_cancels_sibling_readers(failure):
    first, second = FakeFeed(), FakeFeed()
    received = []

    def process(message):
        if failure == "decode":
            raise ValueError("injected decode failure")
        received.append(message)

    task = asyncio.create_task(
        run_feeds([(first, "first", process), (second, "second", received.append)])
    )
    await asyncio.wait_for(
        asyncio.gather(first.waiting.wait(), second.waiting.wait()), 1
    )
    if failure == "overwrite":
        first.dropped = 1
        first.queue.put_nowait(b"incomplete")
    elif failure == "disconnect":
        first.connected = 0
    else:
        first.queue.put_nowait(b"invalid")
    with pytest.raises((FeedIntegrityError, ValueError)):
        await asyncio.wait_for(task, 1)
    assert second.cancelled.is_set()
    assert received == []


@pytest.mark.parametrize(
    "producer_type,consumer_type",
    [(ShmSpscProducer, ShmSpscConsumer), (ShmMpscProducer, ShmMpscConsumer)],
)
def test_shm_overwrite_is_accepted_by_transport_and_rejected_by_example(
    tmp_path, producer_type, consumer_type
):
    options = {"num_rings": 2} if producer_type is ShmMpscProducer else {}
    producer = producer_type(str(tmp_path / "market.shm"), 256, **options)
    consumer = consumer_type(str(tmp_path / "market.shm"))
    try:
        with pytest.raises(FeedIntegrityError, match="SHM transport overwrote"):
            for _ in range(100):
                forward_message(producer, b"x" * 32, "market")
        assert producer.dropped_count > 0
        assert producer.failed_inserts == 0
        with pytest.raises(FeedIntegrityError):
            consume_batch(consumer, "market")
    finally:
        consumer.close()
        producer.close()


def test_overwrite_during_native_read_rejects_batch():
    class Consumer:
        dropped_count = 0

        def consume_all(self):
            self.dropped_count = 1
            return [b"incomplete"]

    with pytest.raises(FeedIntegrityError, match="overwrote 1"):
        consume_batch(Consumer(), "market")


def test_rejected_insert_is_not_treated_as_delivered():
    producer = SimpleNamespace(dropped_count=0, insert=lambda payload: False)
    with pytest.raises(FeedIntegrityError, match="rejected"):
        forward_message(producer, b"update", "market")


@pytest.mark.parametrize(
    "owner,method",
    [
        (binance_stream.BinanceStreamProcessor, "_process_bbo_message"),
        (binance_stream.BinanceStreamProcessor, "_process_trade_message"),
        (binance_stream.BinanceStreamProcessor, "_process_depth_message"),
        (shm_mpsc_pipeline.WorkerProcess, "_process_trade_message"),
        (shm_mpsc_pipeline.WorkerProcess, "_process_bbo_message"),
    ],
)
def test_worker_decode_errors_escape_to_process_owner(owner, method):
    processor = owner.__new__(owner)
    with pytest.raises(msgspec.DecodeError):
        getattr(processor, method)(b"invalid json")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "module,consumer",
    [(candle_types_demo, "_consume_trades"), (moving_averages_demo, "_consume_bbo")],
)
async def test_standalone_demos_reject_startup_loss_before_update(module, consumer):
    feed = FakeFeed()
    feed.connected = 2
    feed.dropped = 1
    with pytest.raises(FeedIntegrityError):
        if module is candle_types_demo:
            await getattr(module, consumer)(feed, None, asyncio.Event())
        else:
            await getattr(module, consumer)(None, feed, asyncio.Event())


@pytest.mark.asyncio
async def test_candle_delivery_failure_stops_demo_without_success_summary(monkeypatch):
    class Pool(FakeFeed):
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            self.closed = True

    class Aggregator:
        def __init__(self):
            self._consumer_tasks = []
            self.summarized = False

        def start_consumers(self):
            async def fail():
                raise RuntimeError("injected candle delivery failure")

            self._consumer_tasks.append(asyncio.create_task(fail()))

        async def stop_consumers(self):
            await asyncio.gather(*self._consumer_tasks, return_exceptions=True)

        def print_summary(self):
            self.summarized = True

    pool, aggregator = Pool(), Aggregator()
    pool.connected = 2

    async def open_pool(**kwargs):
        return pool

    loop = asyncio.get_running_loop()
    monkeypatch.setattr(loop, "add_signal_handler", lambda *args: None)
    monkeypatch.setattr(loop, "remove_signal_handler", lambda *args: None)
    monkeypatch.setattr(candle_types_demo, "WsPool", SimpleNamespace(new=open_pool))
    monkeypatch.setattr(candle_types_demo, "CandleAggregator", lambda: aggregator)
    with pytest.raises(RuntimeError, match="candle delivery failure"):
        await asyncio.wait_for(candle_types_demo._run(), 1)
    assert pool.closed
    assert not aggregator.summarized


def test_mpsc_snapshot_failure_never_publishes_ready(monkeypatch):
    closed, ready = [], []

    class Master:
        def __init__(self, *args):
            pass

        def _load_snapshots(self):
            raise ValueError("injected REST failure")

        def shutdown(self):
            closed.append(True)

    monkeypatch.setattr(shm_mpsc_pipeline, "MasterProcess", Master)
    monkeypatch.setattr(
        shm_mpsc_pipeline.StartupEvent, "set", lambda self: ready.append(True)
    )
    with pytest.raises(ValueError, match="REST failure"):
        shm_mpsc_pipeline.master_entry([], {}, "logger", "data", "ready")
    assert ready == []
    assert closed == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize("startup_loss", [False, True])
async def test_binance_full_startup_checks_loss_before_forwarding_ready_or_market_data(
    monkeypatch, startup_loss
):
    opened, closed, sent = [], [], []

    class Stream(FakeFeed):
        def __init__(self, config):
            super().__init__()
            self.name = config.wss_url
            assert not config.auto_reconnect

        async def __aenter__(self):
            opened.append(self)
            if startup_loss and "@depth" in self.name:
                self.dropped = 1
            return self

        async def __aexit__(self, *args):
            closed.append(self)

    async def stop_after_startup(feeds):
        assert len(opened) == 3
        assert [msgspec.json.decode(message)["msg_type"] for message in sent] == [
            "streams_ready"
        ]
        raise ValueError("injected stop after full startup")

    processor = binance_stream.BinanceStreamProcessor.__new__(
        binance_stream.BinanceStreamProcessor
    )
    processor.symbol = "BTCUSDT"
    processor._logger = SimpleNamespace(info=lambda message: None)
    processor.data_producer = SimpleNamespace(
        dropped_count=0, insert=lambda payload: sent.append(payload) or True
    )
    processor.encoder = msgspec.json.Encoder()
    monkeypatch.setattr(binance_stream, "WsSingle", Stream)
    monkeypatch.setattr(binance_stream, "run_feeds", stop_after_startup)
    with pytest.raises((FeedIntegrityError, ValueError)):
        await processor._run_streams()
    assert len(opened) == len(closed) == 3
    assert len(sent) == (0 if startup_loss else 1)


@pytest.mark.parametrize("exitcode", [0, 1])
def test_parent_rejects_early_child_exit_and_joins_every_child(
    monkeypatch, tmp_path, exitcode
):
    children = []

    class Process:
        pid = None

        def __init__(self, target, **kwargs):
            self.name = target.__name__
            self.exitcode = None
            self.joined = False
            self.alive = False
            children.append(self)

        def start(self):
            if (
                len(
                    [
                        child
                        for child in children
                        if child.alive or child.exitcode is not None
                    ]
                )
                == 0
            ):
                self.exitcode = exitcode
            else:
                self.alive = True

        def is_alive(self):
            return self.alive

        def join(self, timeout):
            self.joined = True
            self.alive = False
            if self.exitcode is None:
                self.exitcode = 0

        def terminate(self):
            self.alive = False
            self.exitcode = -15

    monkeypatch.setattr(binance_stream.multiprocessing, "Process", Process)
    with pytest.raises(RuntimeError, match="child process"):
        binance_stream._run_processes(None, str(tmp_path))
    assert len(children) == 2
    assert all(child.joined and not child.is_alive() for child in children)
