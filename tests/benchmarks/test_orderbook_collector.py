"""Failure handling and lifecycle checks for the orderbook data collector."""

import asyncio
import json
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
        session.sockets[working].messages.put_nowait({"sequence": index})
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
async def test_stream_error_is_not_hidden_by_simultaneous_message_limit(
    tmp_path, monkeypatch
):
    session = FakeSession()
    for index in range(4):
        session.sockets["bbo"].messages.put_nowait({"sequence": index})
    session.sockets["depth"].messages.put_nowait(
        ConnectionError("injected read failure")
    )
    collector, _ = make_collector(tmp_path, monkeypatch, session)

    with pytest.raises(RuntimeError, match="injected read failure"):
        await asyncio.wait_for(collector.run(), 2)

    assert all(socket.closed for socket in session.sockets.values())
    assert session.closed


@pytest.mark.asyncio
async def test_successful_collection_reads_both_feeds_and_closes_them(
    tmp_path, monkeypatch
):
    session = FakeSession()
    session.sockets["bbo"].messages.put_nowait({"bid": "1"})
    session.sockets["depth"].messages.put_nowait({"U": 90, "u": 99, "pu": 89})
    session.sockets["depth"].messages.put_nowait({"U": 99, "u": 101, "pu": 98})
    session.sockets["depth"].messages.put_nowait({"U": 102, "u": 103, "pu": 101})
    collector, path = make_collector(tmp_path, monkeypatch, session)
    path.write_text("previous capture\n")

    async def finish_after_bridge():
        await asyncio.gather(
            *(socket.first_message.wait() for socket in session.sockets.values())
        )
        session.sockets["bbo"].messages.put_nowait({"bid": "2"})
        session.sockets["bbo"].messages.put_nowait({"bid": "3"})
        session.sockets["bbo"].messages.put_nowait({"bid": "4"})

    feeder = asyncio.create_task(finish_after_bridge())
    try:
        await asyncio.wait_for(collector.run(), 2)
        await asyncio.wait_for(feeder, 2)
    finally:
        feeder.cancel()
        await asyncio.gather(feeder, return_exceptions=True)

    types = read_types(path)
    assert len(types) == 5
    assert types[:2] == ["snapshot", "delta"]
    assert {"snapshot", "bbo", "delta"} <= set(types)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records[1]["data"]["U"] <= 100 <= records[1]["data"]["u"]
    deltas = [record["data"] for record in records if record["type"] == "delta"]
    assert len(deltas) == 2
    assert deltas[1]["pu"] == deltas[0]["u"]
    assert all(record["data"].get("bid") != "1" for record in records)
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
    session.sockets["bbo"].messages.put_nowait({"bid": "1"})
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
async def test_invalid_snapshot_id_preserves_previous_capture(tmp_path, monkeypatch):
    session = FakeSession()
    session.sockets["depth"].messages.put_nowait({"U": 99, "u": 101, "pu": 98})
    collector, path = make_collector(tmp_path, monkeypatch, session)
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
    assert all(socket.closed for socket in session.sockets.values())


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
