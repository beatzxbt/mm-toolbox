"""Live Binance orderbook validation for the standard orderbook implementation.

Run with:
    uv run pytest tests/orderbook/standard/test_live_binance.py --run-live --live-timeout 30 -v -s
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import aiohttp
import msgspec
import pytest

from mm_toolbox.orderbook.standard import Orderbook, OrderbookLevel


@dataclass(slots=True)
class DepthEvent:
    """Buffered Binance spot diff-depth event."""

    seq: int
    U: int
    u: int
    bids: list[list[str]]
    asks: list[list[str]]


@dataclass(slots=True)
class BookTickerEvent:
    """Buffered Binance spot bookTicker event."""

    seq: int
    u: int
    bid_price: float
    bid_qty: float
    ask_price: float
    ask_qty: float


@dataclass(slots=True)
class ReplayStats:
    """Summary stats for depth replay validation."""

    applied_depth_events: int = 0
    skipped_old_events: int = 0
    skipped_prestart_events: int = 0
    sequence_gap_events: int = 0
    state_mismatches: int = 0
    raised_errors: int = 0


def _ticks(price: float, inv_tick_size: float) -> int:
    return int(price * inv_tick_size)


def _lots(size: float, inv_lot_size: float) -> int:
    return int(size * inv_lot_size)


async def _fetch_exchange_filters(
    session: aiohttp.ClientSession,
    symbol: str,
) -> tuple[float, float]:
    url = "https://api.binance.com/api/v3/exchangeInfo"
    async with session.get(url, params={"symbol": symbol}, timeout=15) as resp:
        resp.raise_for_status()
        payload = await resp.json()

    symbols = payload.get("symbols", [])
    assert len(symbols) == 1, f"Expected one exchangeInfo symbol entry, got {len(symbols)}"

    filters = symbols[0].get("filters", [])
    price_filter = next((f for f in filters if f.get("filterType") == "PRICE_FILTER"), None)
    lot_filter = next((f for f in filters if f.get("filterType") == "LOT_SIZE"), None)
    assert price_filter is not None, "Missing PRICE_FILTER"
    assert lot_filter is not None, "Missing LOT_SIZE"

    tick_size = float(price_filter["tickSize"])
    lot_size = float(lot_filter["stepSize"])
    assert tick_size > 0.0
    assert lot_size > 0.0
    return tick_size, lot_size


async def _fetch_snapshot(
    session: aiohttp.ClientSession,
    symbol: str,
    limit: int,
) -> dict[str, Any]:
    url = "https://api.binance.com/api/v3/depth"
    async with session.get(
        url,
        params={"symbol": symbol, "limit": str(limit)},
        timeout=20,
    ) as resp:
        resp.raise_for_status()
        payload = await resp.json()

    assert "lastUpdateId" in payload
    assert "bids" in payload
    assert "asks" in payload
    return payload


async def _capture_events(
    symbol: str,
    duration_s: float,
    snapshot_limit: int,
) -> tuple[
    dict[str, Any],
    float,
    float,
    list[DepthEvent],
    list[BookTickerEvent],
    list[DepthEvent | BookTickerEvent],
]:
    stream_url = (
        "wss://stream.binance.com:9443/stream?streams="
        f"{symbol.lower()}@depth@100ms/{symbol.lower()}@bookTicker"
    )

    depth_events: list[DepthEvent] = []
    ticker_events: list[BookTickerEvent] = []
    ordered_events: list[DepthEvent | BookTickerEvent] = []

    timeout = aiohttp.ClientTimeout(total=None, connect=10, sock_read=10)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tick_size, lot_size = await _fetch_exchange_filters(session, symbol)

        ws_timeout = aiohttp.ClientWSTimeout(ws_receive=10.0)
        async with session.ws_connect(
            stream_url,
            heartbeat=20.0,
            timeout=ws_timeout,
        ) as ws:
            snapshot = await _fetch_snapshot(session, symbol, snapshot_limit)

            start = time.monotonic()
            seq = 0
            while time.monotonic() - start < duration_s:
                msg = await ws.receive(timeout=10.0)
                if msg.type == aiohttp.WSMsgType.TEXT:
                    payload = msgspec.json.decode(msg.data)
                    stream = payload.get("stream", "")
                    data = payload.get("data", payload)

                    if "U" in data and "u" in data and "b" in data and "a" in data:
                        evt = DepthEvent(
                            seq=seq,
                            U=int(data["U"]),
                            u=int(data["u"]),
                            bids=data.get("b", []),
                            asks=data.get("a", []),
                        )
                        depth_events.append(evt)
                        ordered_events.append(evt)
                        seq += 1
                        continue

                    if (
                        stream.endswith("@bookTicker")
                        or (
                            "u" in data
                            and "b" in data
                            and "B" in data
                            and "a" in data
                            and "A" in data
                            and "U" not in data
                        )
                    ):
                        evt = BookTickerEvent(
                            seq=seq,
                            u=int(data["u"]),
                            bid_price=float(data["b"]),
                            bid_qty=float(data["B"]),
                            ask_price=float(data["a"]),
                            ask_qty=float(data["A"]),
                        )
                        ticker_events.append(evt)
                        ordered_events.append(evt)
                        seq += 1
                        continue

                if msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED):
                    raise RuntimeError("WebSocket closed unexpectedly during capture")
                if msg.type == aiohttp.WSMsgType.ERROR:
                    raise RuntimeError(f"WebSocket error: {ws.exception()}")

    return snapshot, tick_size, lot_size, depth_events, ticker_events, ordered_events


async def _capture_events_with_retries(
    symbol: str,
    duration_s: float,
    snapshot_limit: int,
    max_attempts: int = 3,
) -> tuple[
    dict[str, Any],
    float,
    float,
    list[DepthEvent],
    list[BookTickerEvent],
    list[DepthEvent | BookTickerEvent],
]:
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return await _capture_events(symbol, duration_s, snapshot_limit)
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError, RuntimeError) as exc:
            last_exc = exc
            if attempt + 1 == max_attempts:
                break
            await asyncio.sleep(1.0 + attempt)

    raise AssertionError(f"Unable to capture live Binance events after retries: {last_exc}")


def _snapshot_levels(snapshot: dict[str, Any]) -> tuple[list[OrderbookLevel], list[OrderbookLevel]]:
    bids = [
        OrderbookLevel(price=float(price), size=float(size), norders=0)
        for price, size in snapshot["bids"]
    ]
    asks = [
        OrderbookLevel(price=float(price), size=float(size), norders=0)
        for price, size in snapshot["asks"]
    ]
    return asks, bids


def _snapshot_reference(
    snapshot: dict[str, Any],
    inv_tick_size: float,
    inv_lot_size: float,
) -> tuple[dict[int, int], dict[int, int]]:
    bids: dict[int, int] = {}
    asks: dict[int, int] = {}

    for price, size in snapshot["bids"]:
        t = _ticks(float(price), inv_tick_size)
        lots = _lots(float(size), inv_lot_size)
        if lots > 0:
            bids[t] = lots

    for price, size in snapshot["asks"]:
        t = _ticks(float(price), inv_tick_size)
        lots = _lots(float(size), inv_lot_size)
        if lots > 0:
            asks[t] = lots

    return bids, asks


def _apply_depth_to_reference(
    event: DepthEvent,
    bids: dict[int, int],
    asks: dict[int, int],
    inv_tick_size: float,
    inv_lot_size: float,
) -> None:
    for price, size in event.bids:
        t = _ticks(float(price), inv_tick_size)
        lots = _lots(float(size), inv_lot_size)
        if lots == 0:
            bids.pop(t, None)
        else:
            bids[t] = lots

    for price, size in event.asks:
        t = _ticks(float(price), inv_tick_size)
        lots = _lots(float(size), inv_lot_size)
        if lots == 0:
            asks.pop(t, None)
        else:
            asks[t] = lots


def _apply_depth_to_orderbook(ob: Orderbook, event: DepthEvent) -> None:
    bids = [
        OrderbookLevel(price=float(price), size=float(size), norders=0)
        for price, size in event.bids
    ]
    asks = [
        OrderbookLevel(price=float(price), size=float(size), norders=0)
        for price, size in event.asks
    ]
    ob.consume_deltas(asks=asks, bids=bids)


def _replay_depth_with_reference(
    snapshot: dict[str, Any],
    depth_events: list[DepthEvent],
    tick_size: float,
    lot_size: float,
) -> tuple[ReplayStats, list[str]]:
    inv_tick_size = 1.0 / tick_size
    inv_lot_size = 1.0 / lot_size

    ob = Orderbook(tick_size=tick_size, lot_size=lot_size, size=1)
    asks_levels, bids_levels = _snapshot_levels(snapshot)
    ob.consume_snapshot(asks=asks_levels, bids=bids_levels)

    ref_bids, ref_asks = _snapshot_reference(snapshot, inv_tick_size, inv_lot_size)
    stats = ReplayStats()
    errors: list[str] = []

    def record_error(msg: str) -> None:
        stats.state_mismatches += 1
        if len(errors) < 20:
            errors.append(msg)

    def validate_state(context: str) -> None:
        bid_keys = sorted(ref_bids)
        ask_keys = sorted(ref_asks)

        if ob._sorted_bid_ticks != bid_keys:
            record_error(f"{context}: sorted bid ticks mismatch")
        if ob._sorted_ask_ticks != ask_keys:
            record_error(f"{context}: sorted ask ticks mismatch")
        if set(ob._bids) != set(ref_bids):
            record_error(f"{context}: bid key set mismatch")
        if set(ob._asks) != set(ref_asks):
            record_error(f"{context}: ask key set mismatch")

        for tick, lots in ref_bids.items():
            level = ob._bids.get(tick)
            if level is None or level.lots != lots:
                record_error(f"{context}: bid lots mismatch at tick={tick}")
                break

        for tick, lots in ref_asks.items():
            level = ob._asks.get(tick)
            if level is None or level.lots != lots:
                record_error(f"{context}: ask lots mismatch at tick={tick}")
                break

        both_sides = bool(ref_bids) and bool(ref_asks)
        if ob.is_populated() != both_sides:
            record_error(f"{context}: is_populated mismatch")

        if both_sides:
            expected_bid = bid_keys[-1]
            expected_ask = ask_keys[0]
            try:
                best_bid, best_ask = ob.get_bbo()
            except Exception as exc:
                record_error(f"{context}: get_bbo raised unexpectedly: {type(exc).__name__}")
                return
            if best_bid.ticks != expected_bid or best_ask.ticks != expected_ask:
                record_error(f"{context}: bbo mismatch")
        else:
            try:
                ob.get_bbo()
                record_error(f"{context}: get_bbo should fail when side is empty")
            except ValueError:
                pass

    validate_state("snapshot")

    last_update_id = int(snapshot["lastUpdateId"])
    started = False

    for event in depth_events:
        if event.u <= last_update_id:
            stats.skipped_old_events += 1
            continue

        if not started:
            if event.U <= last_update_id + 1 <= event.u:
                started = True
            else:
                stats.skipped_prestart_events += 1
                continue
        else:
            if event.u < last_update_id + 1:
                stats.skipped_old_events += 1
                continue
            if event.U > last_update_id + 1:
                stats.sequence_gap_events += 1
                record_error(
                    f"sequence gap at seq={event.seq}; expected={last_update_id + 1} got=[{event.U},{event.u}]"
                )
                break

        try:
            _apply_depth_to_reference(event, ref_bids, ref_asks, inv_tick_size, inv_lot_size)
            _apply_depth_to_orderbook(ob, event)
            stats.applied_depth_events += 1
            last_update_id = event.u
            validate_state(f"depth seq={event.seq}")
        except Exception as exc:
            stats.raised_errors += 1
            record_error(f"depth apply failed at seq={event.seq}: {type(exc).__name__}: {exc}")
            break

    return stats, errors


def _replay_interleaved_with_book_ticker(
    snapshot: dict[str, Any],
    ordered_events: list[DepthEvent | BookTickerEvent],
    tick_size: float,
    lot_size: float,
) -> tuple[dict[str, int], list[str]]:
    inv_tick_size = 1.0 / tick_size
    inv_lot_size = 1.0 / lot_size
    ob = Orderbook(tick_size=tick_size, lot_size=lot_size, size=1)
    asks_levels, bids_levels = _snapshot_levels(snapshot)
    ob.consume_snapshot(asks=asks_levels, bids=bids_levels)

    stats = {
        "depth_events": 0,
        "book_ticker_events": 0,
        "book_ticker_checked": 0,
        "book_ticker_stale_or_out_of_order": 0,
        "book_ticker_mismatches": 0,
    }
    errors: list[str] = []

    last_ticker_u = -1
    for event in ordered_events:
        if isinstance(event, DepthEvent):
            _apply_depth_to_orderbook(ob, event)
            stats["depth_events"] += 1
            continue

        stats["book_ticker_events"] += 1
        if event.u < last_ticker_u:
            stats["book_ticker_stale_or_out_of_order"] += 1
        last_ticker_u = max(last_ticker_u, event.u)

        ob.consume_bbo(
            ask=OrderbookLevel(price=event.ask_price, size=event.ask_qty, norders=0),
            bid=OrderbookLevel(price=event.bid_price, size=event.bid_qty, norders=0),
        )

        bid_lots = _lots(event.bid_qty, inv_lot_size)
        ask_lots = _lots(event.ask_qty, inv_lot_size)
        if bid_lots <= 0 or ask_lots <= 0:
            continue

        stats["book_ticker_checked"] += 1
        expected_bid_tick = _ticks(event.bid_price, inv_tick_size)
        expected_ask_tick = _ticks(event.ask_price, inv_tick_size)

        try:
            best_bid, best_ask = ob.get_bbo()
        except Exception as exc:
            stats["book_ticker_mismatches"] += 1
            if len(errors) < 20:
                errors.append(f"bookTicker get_bbo failed: {type(exc).__name__}: {exc}")
            continue

        if best_bid.ticks != expected_bid_tick or best_ask.ticks != expected_ask_tick:
            stats["book_ticker_mismatches"] += 1
            if len(errors) < 20:
                errors.append(
                    f"bookTicker mismatch seq={event.seq} ob=({best_bid.ticks},{best_ask.ticks}) evt=({expected_bid_tick},{expected_ask_tick})"
                )

        if ob._sorted_bid_ticks and ob._sorted_bid_ticks[-1] != expected_bid_tick:
            stats["book_ticker_mismatches"] += 1
            if len(errors) < 20:
                errors.append(
                    f"stale better bid remained top={ob._sorted_bid_ticks[-1]} expected={expected_bid_tick}"
                )

        if ob._sorted_ask_ticks and ob._sorted_ask_ticks[0] != expected_ask_tick:
            stats["book_ticker_mismatches"] += 1
            if len(errors) < 20:
                errors.append(
                    f"stale better ask remained top={ob._sorted_ask_ticks[0]} expected={expected_ask_tick}"
                )

    return stats, errors


@pytest.mark.live
@pytest.mark.slow
@pytest.mark.asyncio
async def test_live_binance_btcusdt_depth_and_bookticker_zero_faults(
    request: pytest.FixtureRequest,
) -> None:
    """Validate live Binance BTCUSDT updates with strict zero-fault replay checks."""
    symbol = "BTCUSDT"
    live_timeout = float(request.config.getoption("--live-timeout"))
    capture_duration_s = max(10.0, min(live_timeout, 45.0))
    snapshot_limit = 1000

    (
        snapshot,
        tick_size,
        lot_size,
        depth_events,
        ticker_events,
        ordered_events,
    ) = await _capture_events_with_retries(
        symbol=symbol,
        duration_s=capture_duration_s,
        snapshot_limit=snapshot_limit,
    )

    assert len(depth_events) > 0, "No depth events captured"
    assert len(ticker_events) > 0, "No bookTicker events captured"

    depth_stats, depth_errors = _replay_depth_with_reference(
        snapshot=snapshot,
        depth_events=depth_events,
        tick_size=tick_size,
        lot_size=lot_size,
    )
    assert depth_stats.applied_depth_events > 0, "No depth events were applied"
    assert depth_stats.sequence_gap_events == 0, f"Sequence gaps found: {depth_errors}"
    assert depth_stats.state_mismatches == 0, f"Depth mismatches found: {depth_errors}"
    assert depth_stats.raised_errors == 0, f"Depth replay errors: {depth_errors}"
    assert not depth_errors, f"Depth errors: {depth_errors}"

    interleaved_stats, interleaved_errors = _replay_interleaved_with_book_ticker(
        snapshot=snapshot,
        ordered_events=ordered_events,
        tick_size=tick_size,
        lot_size=lot_size,
    )
    assert interleaved_stats["book_ticker_events"] > 0
    assert interleaved_stats["book_ticker_checked"] > 0
    assert interleaved_stats["book_ticker_stale_or_out_of_order"] == 0
    assert interleaved_stats["book_ticker_mismatches"] == 0, interleaved_errors
    assert not interleaved_errors, interleaved_errors
