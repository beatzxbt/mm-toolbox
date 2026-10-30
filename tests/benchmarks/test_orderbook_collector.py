"""Failure handling and lifecycle checks for the orderbook data collector."""

import asyncio
import json
from io import StringIO
import sys
from types import SimpleNamespace

import aiohttp
import pytest

from benchmarks.orderbook import collect_data


_CLOSE = object()
_ERROR = object()


class FakeWebSocket:
    def __init__(self):
        self.messages = asyncio.Queue()
        self.first_message = asyncio.Event()
        self.reading = asyncio.Event()
        self.read_count = 0
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        self.reading.set()
        message = await self.messages.get()
        if message is _CLOSE:
            raise StopAsyncIteration
        if message is _ERROR:
            return SimpleNamespace(type=aiohttp.WSMsgType.ERROR)
        if isinstance(message, Exception):
            raise message
        self.read_count += 1
        self.first_message.set()
        return SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data=json.dumps(message))

    def exception(self):
        return ConnectionError("injected WebSocket error")


class FakeConnection:
    def __init__(self, session, kind):
        self.session = session
        self.kind = kind

    async def __aenter__(self):
        if self.kind == self.session.hang_connect:
            await asyncio.Event().wait()
        if self.kind == self.session.fail_connect:
            # Give the other stream a chance to fill the old collector's limit.
            await asyncio.sleep(0.01)
            raise ConnectionError("injected connection failure")
        self.session.connected.add(self.kind)
        self.session.one_connected.set()
        if len(self.session.connected) == 2:
            self.session.both_connected.set()
        return self.session.sockets[self.kind]

    async def __aexit__(self, *_):
        self.session.sockets[self.kind].closed = True


class FakeSession:
    def __init__(self, fail_connect=None, hang_connect=None):
        self.fail_connect = fail_connect
        self.hang_connect = hang_connect
        self.sockets = {"bbo": FakeWebSocket(), "depth": FakeWebSocket()}
        self.connected = set()
        self.one_connected = asyncio.Event()
        self.both_connected = asyncio.Event()
        self.closed = False
        self.urls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        self.closed = True

    def ws_connect(self, url):
        self.urls.append(url)
        return FakeConnection(self, "depth" if "@depth" in url else "bbo")


def make_collector(tmp_path, monkeypatch, session, *, max_messages=5):
    async def fake_snapshot(collector, _session):
        if collector.config.max_messages > 1:
            assert _session.sockets["depth"].first_message.is_set()
        return {"lastUpdateId": 100, "bids": [], "asks": []}

    monkeypatch.setattr(collect_data.aiohttp, "ClientSession", lambda: session)
    monkeypatch.setattr(
        collect_data.BinanceDataCollector, "fetch_snapshot", fake_snapshot
    )
    path = tmp_path / "capture.jsonl"
    config = collect_data.CollectorConfig(str(path), max_messages, "BTCUSDT", 5)
    return collect_data.BinanceDataCollector(config), path


def read_types(path):
    return [json.loads(line)["type"] for line in path.read_text().splitlines()]


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", ["bbo", "depth"])
async def test_startup_failure_does_not_publish_a_partial_stream(
    tmp_path, monkeypatch, failed
):
    session = FakeSession(fail_connect=failed)
    working = "depth" if failed == "bbo" else "bbo"
    for index in range(4):
        session.sockets[working].messages.put_nowait({"u": index, "sequence": index})
    collector, path = make_collector(tmp_path, monkeypatch, session)
    path.write_text("previous capture\n")

    with pytest.raises(RuntimeError, match="injected connection failure"):
        await asyncio.wait_for(collector.run(), 2)

    assert path.read_text() == "previous capture\n"
    assert not list(tmp_path.glob("*.partial"))
    assert session.sockets[working].closed
    assert session.closed
    assert collector.output_file is None


@pytest.mark.asyncio
async def test_stalled_startup_times_out_and_closes_the_other_feed(
    tmp_path, monkeypatch
):
    session = FakeSession(hang_connect="depth")
    collector, path = make_collector(tmp_path, monkeypatch, session)
    monkeypatch.setattr(collect_data, "_STREAM_START_TIMEOUT_S", 0.02)

    with pytest.raises(TimeoutError, match="both WebSocket feeds to start"):
        await asyncio.wait_for(collector.run(), 2)

    assert not path.exists()
    assert session.sockets["bbo"].closed
    assert session.closed
    assert collector.output_file is None


@pytest.mark.asyncio
@pytest.mark.parametrize("message", [_CLOSE, _ERROR])
async def test_stream_close_or_error_fails_collection(tmp_path, monkeypatch, message):
    session = FakeSession()
    session.sockets["depth"].messages.put_nowait(message)
    collector, path = make_collector(tmp_path, monkeypatch, session)

    with pytest.raises(RuntimeError, match="Depth stream failed"):
        await asyncio.wait_for(collector.run(), 2)

    assert not path.exists()
    assert all(socket.closed for socket in session.sockets.values())
    assert session.closed
    assert collector.output_file is None


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["read", "error-frame", "close"])
async def test_stream_failure_is_not_hidden_by_bridge_message_limit(
    tmp_path, monkeypatch, failure
):
    session = FakeSession()
    collector, path = make_collector(tmp_path, monkeypatch, session, max_messages=2)
    path.write_text("previous capture\n")

    class FailureAtLimitSocket(FakeWebSocket):
        async def __anext__(self):
            if self.read_count:
                # The bridge really reaches the target before this already
                # pending read reports failure, exercising the completion race.
                await collector._stop_event.wait()
                if failure == "read":
                    raise ConnectionError("injected read failure at message limit")
                if failure == "close":
                    raise StopAsyncIteration
                return SimpleNamespace(type=aiohttp.WSMsgType.ERROR)
            return await super().__anext__()

    session.sockets["depth"] = FailureAtLimitSocket()
    session.sockets["depth"].messages.put_nowait({"U": 99, "u": 101, "pu": 98})

    with pytest.raises(RuntimeError, match="Depth stream failed"):
        await asyncio.wait_for(collector.run(), 2)

    assert collector.message_count == 2
    assert path.read_text() == "previous capture\n"
    assert not list(tmp_path.glob("*.partial"))
    assert all(socket.closed for socket in session.sockets.values())
    assert session.closed
    assert collector.output_file is None


@pytest.mark.asyncio
async def test_successful_collection_reads_both_feeds_and_closes_them(
    tmp_path, monkeypatch
):
    session = FakeSession()
    session.sockets["bbo"].messages.put_nowait({"u": 100, "bid": "old"})
    session.sockets["depth"].messages.put_nowait({"U": 99, "u": 101, "pu": 98})
    collector, path = make_collector(tmp_path, monkeypatch, session)
    path.write_text("previous capture\n")

    async def finish_after_bridge():
        while collector.message_count < 2:
            await asyncio.sleep(0)
        session.sockets["bbo"].messages.put_nowait({"u": 102, "bid": "new"})
        while session.sockets["bbo"].read_count < 2:
            await asyncio.sleep(0)
        session.sockets["depth"].messages.put_nowait({"U": 102, "u": 103, "pu": 101})
        session.sockets["depth"].messages.put_nowait({"U": 104, "u": 105, "pu": 103})

    feeder = asyncio.create_task(finish_after_bridge())
    try:
        await asyncio.wait_for(collector.run(), 2)
        await asyncio.wait_for(feeder, 2)
    finally:
        feeder.cancel()
        await asyncio.gather(feeder, return_exceptions=True)

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["type"] for record in records] == [
        "snapshot",
        "delta",
        "bbo",
        "delta",
        "delta",
    ]
    assert [record["data"].get("u", 100) for record in records] == [
        100,
        101,
        102,
        103,
        105,
    ]
    assert records[1]["recv_ts_ns"] < records[0]["recv_ts_ns"]
    assert not list(tmp_path.glob("*.partial"))
    assert session.urls == [
        "wss://fstream.binance.com/public/ws/btcusdt@bookTicker",
        "wss://fstream.binance.com/public/ws/btcusdt@depth@100ms",
    ]
    assert all(socket.closed for socket in session.sockets.values())
    assert session.closed
    assert collector.output_file is None


@pytest.mark.asyncio
async def test_depth_updates_are_buffered_while_snapshot_is_in_flight(
    tmp_path, monkeypatch
):
    session = FakeSession()
    session.sockets["depth"].messages.put_nowait({"U": 90, "u": 99, "pu": 89})
    collector, path = make_collector(tmp_path, monkeypatch, session, max_messages=3)
    snapshot_started, allow_snapshot = asyncio.Event(), asyncio.Event()

    async def delayed_snapshot(_collector, _session):
        snapshot_started.set()
        await allow_snapshot.wait()
        return {"lastUpdateId": 100, "bids": [], "asks": []}

    monkeypatch.setattr(
        collect_data.BinanceDataCollector, "fetch_snapshot", delayed_snapshot
    )
    task = asyncio.create_task(collector.run())
    try:
        await asyncio.wait_for(snapshot_started.wait(), 2)
        session.sockets["depth"].messages.put_nowait({"U": 99, "u": 101, "pu": 98})
        session.sockets["depth"].messages.put_nowait({"U": 102, "u": 103, "pu": 101})

        async def wait_for_buffered_depth():
            while session.sockets["depth"].read_count < 3:
                await asyncio.sleep(0)

        await asyncio.wait_for(wait_for_buffered_depth(), 2)
        allow_snapshot.set()
        await asyncio.wait_for(task, 2)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["type"] for record in records] == ["snapshot", "delta", "delta"]
    assert records[1]["data"]["U"] <= 100 <= records[1]["data"]["u"]
    assert records[2]["data"]["pu"] == records[1]["data"]["u"]
    assert records[1]["recv_ts_ns"] < records[0]["recv_ts_ns"]


@pytest.mark.asyncio
async def test_two_message_capture_finishes_at_the_bridge(tmp_path, monkeypatch):
    session = FakeSession()
    session.sockets["depth"].messages.put_nowait({"U": 99, "u": 101, "pu": 98})
    collector, path = make_collector(tmp_path, monkeypatch, session, max_messages=2)

    await asyncio.wait_for(collector.run(), 2)

    assert read_types(path) == ["snapshot", "delta"]
    assert all(socket.closed for socket in session.sockets.values())


@pytest.mark.asyncio
async def test_first_depth_can_start_snapshot_when_bbo_has_filled_buffer(
    tmp_path, monkeypatch
):
    session = FakeSession()
    session.sockets["bbo"].messages.put_nowait({"u": 100, "bid": "1"})
    session.sockets["depth"].messages.put_nowait({"U": 99, "u": 101, "pu": 98})
    collector, path = make_collector(tmp_path, monkeypatch, session, max_messages=2)
    buffer_type = collect_data._BufferedEvents
    monkeypatch.setattr(
        collect_data,
        "_BufferedEvents",
        lambda: buffer_type(max_events=1, max_bytes=1024),
    )

    await asyncio.wait_for(collector.run(), 2)

    assert read_types(path) == ["snapshot", "delta"]
    assert all(socket.closed for socket in session.sockets.values())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("events", "error"),
    [
        ([{"U": 101, "u": 102, "pu": 99}], "cannot bridge"),
        (
            [
                {"U": 99, "u": 101, "pu": 98},
                {"U": 102, "u": 103, "pu": 99},
            ],
            "Depth sequence gap",
        ),
        ([{"U": 99, "u": 101}], "invalid sequence IDs"),
        ([{"U": 99, "u": 100, "pu": 101}], "invalid sequence IDs"),
        ([{"U": True, "u": 101, "pu": 98}], "invalid sequence IDs"),
    ],
)
async def test_invalid_depth_continuity_does_not_publish_capture(
    tmp_path, monkeypatch, events, error
):
    session = FakeSession()
    for event in events:
        session.sockets["depth"].messages.put_nowait(event)
    collector, path = make_collector(tmp_path, monkeypatch, session)
    path.write_text("previous capture\n")

    with pytest.raises((RuntimeError, ValueError), match=error):
        await asyncio.wait_for(collector.run(), 2)

    assert path.read_text() == "previous capture\n"
    assert not list(tmp_path.glob("*.partial"))
    assert all(socket.closed for socket in session.sockets.values())
    assert session.closed


def test_interrupted_cli_returns_nonzero_and_reports_discard(monkeypatch, capsys):
    def interrupted(coroutine):
        coroutine.close()
        raise KeyboardInterrupt

    monkeypatch.setattr(sys, "argv", ["collect_data.py"])
    monkeypatch.setattr(collect_data.asyncio, "run", interrupted)

    with pytest.raises(SystemExit) as error:
        collect_data.main()

    assert error.value.code == 130
    assert "Interrupted. Discarded" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_stale_depth_times_out_without_publishing(tmp_path, monkeypatch):
    session = FakeSession()
    session.sockets["depth"].messages.put_nowait({"U": 90, "u": 99, "pu": 89})
    collector, path = make_collector(tmp_path, monkeypatch, session)
    monkeypatch.setattr(collect_data, "_BRIDGE_TIMEOUT_S", 0.02)

    with pytest.raises(TimeoutError, match="depth snapshot bridge"):
        await asyncio.wait_for(collector.run(), 2)

    assert not path.exists()
    assert not list(tmp_path.glob("*.partial"))
    assert all(socket.closed for socket in session.sockets.values())


@pytest.mark.asyncio
async def test_missing_first_depth_event_times_out(tmp_path, monkeypatch):
    session = FakeSession()
    collector, path = make_collector(tmp_path, monkeypatch, session)
    monkeypatch.setattr(collect_data, "_FIRST_DEPTH_TIMEOUT_S", 0.02)

    with pytest.raises(TimeoutError, match="first depth event"):
        await asyncio.wait_for(collector.run(), 2)

    assert not path.exists()
    assert not list(tmp_path.glob("*.partial"))
    assert all(socket.closed for socket in session.sockets.values())


@pytest.mark.asyncio
async def test_snapshot_timeout_preserves_previous_capture(tmp_path, monkeypatch):
    session = FakeSession()
    session.sockets["depth"].messages.put_nowait({"U": 99, "u": 101, "pu": 98})
    collector, path = make_collector(tmp_path, monkeypatch, session)
    path.write_text("previous capture\n")

    async def stalled_snapshot(_collector, _session):
        await asyncio.Event().wait()

    monkeypatch.setattr(
        collect_data.BinanceDataCollector, "fetch_snapshot", stalled_snapshot
    )
    monkeypatch.setattr(collect_data, "_SNAPSHOT_TIMEOUT_S", 0.02)
    with pytest.raises(TimeoutError, match="depth snapshot"):
        await asyncio.wait_for(collector.run(), 2)

    assert path.read_text() == "previous capture\n"
    assert not list(tmp_path.glob("*.partial"))
    assert all(socket.closed for socket in session.sockets.values())


@pytest.mark.asyncio
@pytest.mark.parametrize("max_messages", [1, 5])
async def test_invalid_snapshot_id_preserves_previous_capture(
    tmp_path, monkeypatch, max_messages
):
    session = FakeSession()
    session.sockets["depth"].messages.put_nowait({"U": 99, "u": 101, "pu": 98})
    collector, path = make_collector(
        tmp_path, monkeypatch, session, max_messages=max_messages
    )
    path.write_text("previous capture\n")

    async def invalid_snapshot(_collector, _session):
        return {"lastUpdateId": True, "bids": [], "asks": []}

    monkeypatch.setattr(
        collect_data.BinanceDataCollector, "fetch_snapshot", invalid_snapshot
    )
    with pytest.raises(ValueError, match="invalid lastUpdateId"):
        await asyncio.wait_for(collector.run(), 2)

    assert path.read_text() == "previous capture\n"
    assert not list(tmp_path.glob("*.partial"))
    assert all(
        socket.closed == (max_messages > 1) for socket in session.sockets.values()
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("max_events", "max_bytes", "first"), [(1, 8, "a"), (2, 4, "abcd")]
)
async def test_buffer_applies_backpressure_without_discarding_events(
    max_events, max_bytes, first
):
    with pytest.raises(ValueError, match="buffer limits"):
        collect_data._BufferedEvents(max_events=0, max_bytes=4)
    events = collect_data._BufferedEvents(max_events=max_events, max_bytes=max_bytes)
    await events.put("delta", first, 1)
    blocked = asyncio.create_task(events.put("bbo", "x", 2))
    await asyncio.sleep(0)
    assert not blocked.done()
    assert await events.get() == ("delta", first, 1)
    await asyncio.wait_for(blocked, 2)
    assert await events.get() == ("bbo", "x", 2)
    with pytest.raises(ValueError, match="buffer limit"):
        await events.put("delta", "x" * (max_bytes + 1), 3)


@pytest.mark.asyncio
async def test_snapshot_only_capture_with_one_message_limit(tmp_path, monkeypatch):
    session = FakeSession()
    collector, path = make_collector(tmp_path, monkeypatch, session, max_messages=1)

    await asyncio.wait_for(collector.run(), 2)

    assert read_types(path) == ["snapshot"]
    assert not session.connected
    assert session.closed


@pytest.mark.asyncio
async def test_cancellation_closes_connected_streams(tmp_path, monkeypatch):
    session = FakeSession()
    collector, _ = make_collector(tmp_path, monkeypatch, session)
    task = asyncio.create_task(collector.run())
    await asyncio.wait_for(
        asyncio.gather(*(socket.reading.wait() for socket in session.sockets.values())),
        2,
    )
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 2)

    assert all(socket.closed for socket in session.sockets.values())
    assert session.closed
    assert collector.output_file is None


@pytest.mark.asyncio
async def test_cancellation_during_startup_closes_ready_stream(tmp_path, monkeypatch):
    session = FakeSession(hang_connect="depth")
    collector, _ = make_collector(tmp_path, monkeypatch, session)
    task = asyncio.create_task(collector.run())
    await asyncio.wait_for(session.one_connected.wait(), 2)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 2)

    assert session.sockets["bbo"].closed
    assert session.closed
    assert collector.output_file is None


@pytest.mark.asyncio
async def test_cancellation_after_bridge_preserves_previous_capture(
    tmp_path, monkeypatch
):
    session = FakeSession()
    session.sockets["depth"].messages.put_nowait({"U": 99, "u": 101, "pu": 98})
    collector, path = make_collector(tmp_path, monkeypatch, session)
    path.write_text("previous capture\n")
    task = asyncio.create_task(collector.run())

    async def wait_for_bridge():
        while collector.message_count < 2:
            await asyncio.sleep(0)

    await asyncio.wait_for(wait_for_bridge(), 2)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 2)

    assert path.read_text() == "previous capture\n"
    assert not list(tmp_path.glob("*.partial"))
    assert all(socket.closed for socket in session.sockets.values())


async def publish_events(events, *, max_messages, snapshot=None):
    """Exercise the real publication owner with deterministic receive order."""
    snapshot = snapshot or {"lastUpdateId": 100, "bids": [], "asks": []}
    collector = collect_data.BinanceDataCollector(
        collect_data.CollectorConfig("unused.jsonl", max_messages, "BTCUSDT", 5)
    )
    collector.output_file = StringIO()
    collector._write_message("snapshot", snapshot, 999)
    queue = collect_data._BufferedEvents()
    for timestamp, (kind, data) in enumerate(events, 1):
        await queue.put(kind, json.dumps(data), timestamp)
    await asyncio.wait_for(
        collector._process_events(queue, snapshot["lastUpdateId"]), 2
    )
    return [json.loads(line) for line in collector.output_file.getvalue().splitlines()]


@pytest.mark.asyncio
async def test_late_bbo_cannot_revert_newer_applied_depth():
    records = await publish_events(
        [
            ("delta", {"U": 99, "u": 110, "pu": 98}),
            ("bbo", {"u": 105, "b": "99", "B": "1", "a": "101", "A": "1"}),
            ("delta", {"U": 111, "u": 115, "pu": 110}),
        ],
        max_messages=3,
    )
    assert [record["type"] for record in records] == ["snapshot", "delta", "delta"]
    assert [record["data"].get("u", 100) for record in records] == [100, 110, 115]


@pytest.mark.asyncio
async def test_future_bbo_waits_without_discarding_intermediate_depth():
    records = await publish_events(
        [
            ("delta", {"U": 99, "u": 110, "pu": 98}),
            ("bbo", {"u": 120}),
            ("bbo", {"u": 118}),
            ("delta", {"U": 111, "u": 115, "pu": 110}),
            ("delta", {"U": 116, "u": 125, "pu": 115}),
        ],
        max_messages=6,
    )
    assert [record["data"].get("u", 100) for record in records] == [
        100,
        110,
        115,
        118,
        120,
        125,
    ]
    assert [record["recv_ts_ns"] for record in records] == [999, 1, 4, 3, 2, 5]
    assert [
        record["data"]["pu"] for record in records if record["type"] == "delta"
    ] == [98, 110, 115]


@pytest.mark.asyncio
async def test_duplicate_ids_are_ignored_and_depth_wins_equal_bbo_ties():
    records = await publish_events(
        [
            ("bbo", {"u": 100}),
            ("delta", {"U": 99, "u": 100, "pu": 98}),
            ("bbo", {"u": 115, "source": "first"}),
            ("bbo", {"u": 115, "source": "duplicate"}),
            ("bbo", {"u": 110}),
            ("delta", {"U": 101, "u": 110, "pu": 100}),
            ("delta", {"U": 101, "u": 110, "pu": 100, "b": [["99", "9"]]}),
            ("delta", {"U": 101, "u": 105, "pu": 100}),
            ("bbo", {"u": 112, "source": "first"}),
            ("bbo", {"u": 112, "source": "duplicate"}),
            ("delta", {"U": 111, "u": 115, "pu": 110}),
            ("bbo", {"u": 112, "source": "late"}),
            ("delta", {"U": 116, "u": 120, "pu": 115}),
        ],
        max_messages=6,
    )
    assert [record["data"].get("u", 100) for record in records] == [
        100,
        100,
        110,
        112,
        115,
        120,
    ]
    assert [record["data"] for record in records if record["type"] == "bbo"] == [
        {"u": 112, "source": "first"}
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("update_id", [None, True, -1, 1.5, "120"])
async def test_malformed_bbo_id_preserves_completed_capture(
    tmp_path, monkeypatch, update_id
):
    session = FakeSession()
    session.sockets["depth"].messages.put_nowait({"U": 99, "u": 110, "pu": 98})
    session.sockets["bbo"].messages.put_nowait({"u": update_id})
    collector, path = make_collector(tmp_path, monkeypatch, session)
    path.write_text("previous capture\n")

    with pytest.raises(ValueError, match="BBO event has an invalid update ID"):
        await asyncio.wait_for(collector.run(), 2)

    assert path.read_text() == "previous capture\n"
    assert not list(tmp_path.glob("*.partial"))
    assert all(socket.closed for socket in session.sockets.values())
    assert session.closed


@pytest.mark.asyncio
async def test_gap_is_validated_before_releasing_pending_bbo(tmp_path):
    collector = collect_data.BinanceDataCollector(
        collect_data.CollectorConfig(str(tmp_path / "unused"), 5, "BTCUSDT", 5)
    )
    collector.output_file = StringIO()
    collector._write_message("snapshot", {"lastUpdateId": 100}, 0)
    events = collect_data._BufferedEvents()
    for kind, data in [
        ("delta", {"U": 99, "u": 110, "pu": 98}),
        ("bbo", {"u": 120}),
        ("delta", {"U": 116, "u": 125, "pu": 115}),
    ]:
        await events.put(kind, json.dumps(data), 1)
    with pytest.raises(RuntimeError, match="expected pu=110, got 115"):
        await collector._process_events(events, 100)
    records = [
        json.loads(line) for line in collector.output_file.getvalue().splitlines()
    ]
    assert [record["type"] for record in records] == ["snapshot", "delta"]


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", ["count", "bytes"])
async def test_pending_bbo_exhaustion_fails_instead_of_blocking_covering_depth(
    tmp_path, monkeypatch, limit
):
    session = FakeSession()
    session.sockets["depth"].messages.put_nowait({"U": 99, "u": 110, "pu": 98})
    collector, path = make_collector(tmp_path, monkeypatch, session, max_messages=4)
    path.write_text("previous capture\n")
    if limit == "count":
        monkeypatch.setattr(collect_data, "_MAX_PENDING_BBO_EVENTS", 1)
    else:
        monkeypatch.setattr(
            collect_data, "_MAX_PENDING_BBO_BYTES", len(json.dumps({"u": 120}))
        )

    async def send_future_bbos():
        while collector.message_count < 2:
            await asyncio.sleep(0)
        session.sockets["bbo"].messages.put_nowait({"u": 120})
        session.sockets["bbo"].messages.put_nowait({"u": 121})
        while session.sockets["bbo"].read_count < 2:
            await asyncio.sleep(0)
        session.sockets["depth"].messages.put_nowait({"U": 111, "u": 125, "pu": 110})

    feeder = asyncio.create_task(send_future_bbos())
    try:
        with pytest.raises(RuntimeError, match="Pending BBO buffer limit exceeded"):
            await asyncio.wait_for(collector.run(), 2)
        await asyncio.wait_for(feeder, 2)
    finally:
        feeder.cancel()
        await asyncio.gather(feeder, return_exceptions=True)
    assert path.read_text() == "previous capture\n"
    assert not list(tmp_path.glob("*.partial"))
    assert all(socket.closed for socket in session.sockets.values())
    assert session.closed
    assert collector.output_file is None


@pytest.mark.asyncio
async def test_duplicate_traffic_cannot_extend_pending_depth_deadline(
    tmp_path, monkeypatch
):
    session = FakeSession()
    bridge = {"U": 99, "u": 110, "pu": 98}
    session.sockets["depth"].messages.put_nowait(bridge)
    session.sockets["bbo"].messages.put_nowait({"u": 120})
    collector, path = make_collector(tmp_path, monkeypatch, session)
    path.write_text("previous capture\n")
    monkeypatch.setattr(collect_data, "_PENDING_DEPTH_TIMEOUT_S", 0.02)

    async def duplicate_traffic():
        while True:
            session.sockets["depth"].messages.put_nowait(bridge)
            session.sockets["bbo"].messages.put_nowait({"u": 120})
            session.sockets["bbo"].messages.put_nowait({"u": 105})
            await asyncio.sleep(0.001)

    feeder = asyncio.create_task(duplicate_traffic())
    try:
        with pytest.raises(TimeoutError, match="depth progress for pending BBOs"):
            await asyncio.wait_for(collector.run(), 2)
    finally:
        feeder.cancel()
        await asyncio.gather(feeder, return_exceptions=True)
    assert collector.message_count == 2
    assert path.read_text() == "previous capture\n"
    assert not list(tmp_path.glob("*.partial"))
    assert all(socket.closed for socket in session.sockets.values())
    assert session.closed
    assert collector.output_file is None


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["disconnect", "cancel"])
async def test_pending_bbo_failure_cleans_tasks_and_preserves_capture(
    tmp_path, monkeypatch, failure
):
    session = FakeSession()
    session.sockets["depth"].messages.put_nowait({"U": 99, "u": 110, "pu": 98})
    collector, path = make_collector(tmp_path, monkeypatch, session)
    path.write_text("previous capture\n")
    existing_tasks = asyncio.all_tasks()
    task = asyncio.create_task(collector.run())
    while collector.message_count < 2:
        await asyncio.sleep(0)
    session.sockets["bbo"].messages.put_nowait({"u": 120})
    while session.sockets["bbo"].read_count < 1:
        await asyncio.sleep(0)
    await asyncio.sleep(0)
    if failure == "disconnect":
        session.sockets["depth"].messages.put_nowait(_CLOSE)
        with pytest.raises(RuntimeError, match="closed before collection completed"):
            await asyncio.wait_for(task, 2)
    else:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)

    assert not [
        task for task in asyncio.all_tasks() - existing_tasks if not task.done()
    ]
    assert collector.message_count == 2
    assert path.read_text() == "previous capture\n"
    assert not list(tmp_path.glob("*.partial"))
    assert all(socket.closed for socket in session.sockets.values())
    assert session.closed
    assert collector.output_file is None


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["standard", "advanced-python", "advanced-cython"])
async def test_offline_publication_replays_full_depth_against_independent_reference(
    target,
):
    """Compact generated feed: source IDs deliberately disagree with arrival order."""
    from tests.orderbook.adapters import make_target
    from tests.orderbook.reference import ReferenceBook

    snapshot = {
        "lastUpdateId": 100,
        "bids": [["99", "3"], ["98", "4"]],
        "asks": [["101", "3"], ["102", "4"]],
    }
    source = [
        (
            "delta",
            {"U": 99, "u": 110, "pu": 98, "b": [["99", "2"]], "a": [["101", "2"]]},
        ),
        ("bbo", {"u": 120, "b": "99", "B": "7", "a": "101", "A": "7"}),
        (
            "delta",
            {"U": 111, "u": 115, "pu": 110, "b": [["98", "5"]], "a": [["102", "5"]]},
        ),
        ("bbo", {"u": 118, "b": "99", "B": "6", "a": "101", "A": "6"}),
        ("bbo", {"u": 118, "b": "99", "B": "999", "a": "101", "A": "999"}),
        ("bbo", {"u": 125, "b": "99", "B": "999", "a": "101", "A": "999"}),
        (
            "delta",
            {"U": 116, "u": 125, "pu": 115, "b": [["99", "8"]], "a": [["101", "8"]]},
        ),
        ("bbo", {"u": 119, "b": "99", "B": "999", "a": "101", "A": "999"}),
        # This future record stays pending at the finite capture boundary.
        ("bbo", {"u": 140, "b": "99", "B": "999", "a": "101", "A": "999"}),
        (
            "delta",
            {"U": 126, "u": 130, "pu": 125, "b": [["98", "0"]], "a": [["102", "0"]]},
        ),
    ]
    expected_depth = {
        110: ({99: 2, 98: 4}, {101: 2, 102: 4}),
        115: ({99: 2, 98: 5}, {101: 2, 102: 5}),
        125: ({99: 8, 98: 5}, {101: 8, 102: 5}),
        130: ({99: 8}, {101: 8}),
    }
    # A plain depth-only dictionary never observes BBOs or production code.
    raw_bids = {int(p): int(q) for p, q in snapshot["bids"]}
    raw_asks = {int(p): int(q) for p, q in snapshot["asks"]}
    for kind, data in source:
        if kind != "delta":
            continue
        for side, field in ((raw_bids, "b"), (raw_asks, "a")):
            for price, quantity in data[field]:
                if int(quantity):
                    side[int(price)] = int(quantity)
                else:
                    side.pop(int(price), None)
        assert (raw_bids, raw_asks) == expected_depth[data["u"]]

    records = await publish_events(source, max_messages=7, snapshot=snapshot)
    assert [record["data"].get("u", 100) for record in records] == [
        100,
        110,
        115,
        118,
        120,
        125,
        130,
    ]
    assert [record["recv_ts_ns"] for record in records] == [999, 1, 3, 4, 2, 7, 10]
    assert [
        record["data"]["u"] for record in records if record["type"] == "delta"
    ] == list(expected_depth)
    assert [
        record["data"]["pu"] for record in records if record["type"] == "delta"
    ] == [98, 110, 115, 125]

    book, reference = make_target(target), ReferenceBook()

    def rows(data, field):
        return [(float(price), float(quantity), 1) for price, quantity in data[field]]

    for record in records:
        kind, data = record["type"], record["data"]
        for destination in (book, reference):
            if kind == "snapshot":
                destination.snapshot(rows(data, "asks"), rows(data, "bids"))
            elif kind == "delta":
                destination.deltas(rows(data, "a"), rows(data, "b"))
            else:
                destination.bbo(
                    (float(data["a"]), float(data["A"]), 1),
                    (float(data["b"]), float(data["B"]), 1),
                )
        for side in ("bids", "asks"):
            assert book.rows(side) == reference.rows(side)
        if kind == "delta":
            actual = tuple(
                {int(p): int(q) for p, q, _ in book.rows(side)}
                for side in ("bids", "asks")
            )
            assert actual == expected_depth[data["u"]]


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["standard", "advanced-python", "advanced-cython"])
async def test_recorded_binance_prefix_publication_matches_raw_depth_oracle(target):
    """F7: real captured multi-level BBO movement reconciles at every depth."""
    from collections import Counter
    from decimal import Decimal
    from pathlib import Path

    from tests.orderbook.adapters import make_target
    from tests.orderbook.reference import ReferenceBook
    from tests.orderbook.test_scenarios import _export_units

    fixture = Path(__file__).parents[1] / "orderbook/data/binance_futures_replay.jsonl"
    source = [json.loads(line) for line in fixture.read_text().splitlines()]
    assert Counter(record["type"] for record in source) == {
        "snapshot": 1,
        "delta": 33,
        "bbo": 916,
    }
    assert source[357]["data"]["u"] == 11932603897551
    assert (source[356]["data"]["b"], source[357]["data"]["b"]) == (
        "80240.00",
        "80239.40",
    )
    assert source[-1]["type"] == "delta"
    assert source[-1]["data"]["u"] == 11932603928847
    tick, lot = Decimal("0.01"), Decimal("0.001")

    def units(value, quantum):
        return int(Decimal(value) / quantum)

    snapshot = source[0]["data"]
    raw_bids = {
        units(p, tick): units(q, lot) for p, q in snapshot["bids"] if Decimal(q)
    }
    raw_asks = {
        units(p, tick): units(q, lot) for p, q in snapshot["asks"] if Decimal(q)
    }
    depth_only = {}
    last_depth = snapshot["lastUpdateId"]
    for record in source[1:]:
        if record["type"] != "delta":
            continue
        data = record["data"]
        if not depth_only:
            assert data["U"] <= last_depth <= data["u"]
        else:
            assert data["pu"] == last_depth
        for side, field in ((raw_bids, "b"), (raw_asks, "a")):
            for price, quantity in data[field]:
                price, quantity = units(price, tick), units(quantity, lot)
                if quantity:
                    side[price] = quantity
                else:
                    side.pop(price, None)
        depth_only[data["u"]] = (raw_bids.copy(), raw_asks.copy())
        last_depth = data["u"]

    collector = collect_data.BinanceDataCollector(
        collect_data.CollectorConfig("unused.jsonl", 744, "BTCUSDT", 5)
    )
    collector.output_file = StringIO()
    collector._write_message("snapshot", snapshot, source[0]["recv_ts_ns"])
    events = collect_data._BufferedEvents()
    for record in source[1:]:
        await events.put(
            record["type"], json.dumps(record["data"]), record["recv_ts_ns"]
        )
    await asyncio.wait_for(
        collector._process_events(events, snapshot["lastUpdateId"]), 2
    )
    published = [
        json.loads(line) for line in collector.output_file.getvalue().splitlines()
    ]
    assert Counter(record["type"] for record in published) == {
        "snapshot": 1,
        "delta": 33,
        "bbo": 710,
    }
    ids = [record["data"].get("u", snapshot["lastUpdateId"]) for record in published]
    assert all(left < right for left, right in zip(ids, ids[1:]))
    assert published[-1] == source[-1]
    assert [record for record in published if record["type"] == "delta"] == [
        record for record in source if record["type"] == "delta"
    ]
    # Gate reordering must retain the original receive stamp and exact payload.
    original_rows = {json.dumps(record, sort_keys=True) for record in source}
    assert all(
        json.dumps(record, sort_keys=True) in original_rows for record in published
    )

    book = make_target(target, tick=float(tick), lot=float(lot), capacity=4096)
    reference = ReferenceBook(float(tick), float(lot))
    seen_depth = []
    multi_level_jump_checked = False

    def rows(data, field):
        return [(float(p), float(q), 1) for p, q in data[field]]

    for record in published:
        data, kind = record["data"], record["type"]
        if kind == "bbo" and data["u"] == source[357]["data"]["u"]:
            incoming_bid = units(data["b"], tick)
            assert sum(price > incoming_bid for price in reference.bids) >= 2
            multi_level_jump_checked = True
        for destination in (book, reference):
            if kind == "snapshot":
                destination.snapshot(rows(data, "asks"), rows(data, "bids"))
            elif kind == "delta":
                destination.deltas(rows(data, "a"), rows(data, "b"))
            else:
                destination.bbo(
                    (float(data["a"]), float(data["A"]), 1),
                    (float(data["b"]), float(data["B"]), 1),
                )
        actual = {}
        for side in ("bids", "asks"):
            actual[side] = [
                (_export_units(p, tick), _export_units(q, lot), n)
                for p, q, n in book.rows(side)
            ]
            assert actual[side] == reference.integer_rows(side), (
                target,
                kind,
                data.get("u"),
                side,
            )
        if kind == "delta":
            seen_depth.append(data["u"])
            assert (
                tuple({p: q for p, q, _ in actual[side]} for side in ("bids", "asks"))
                == depth_only[data["u"]]
            )
    assert multi_level_jump_checked
    assert seen_depth == list(depth_only)
