"""Live Binance orderbook validation for the standard orderbook implementation.

Layer 3 integration tests: captures live Binance market data and replays it
through the standard orderbook, validating zero-fault state consistency.
Covers depth replay, interleaved depth+bookTicker replay, and sequence gap detection.

Run with:
    uv run pytest tests/orderbook/standard/test_live_binance.py --run-live --live-timeout 30 -v -s
"""

from __future__ import annotations

import asyncio
import time
from decimal import Decimal
from typing import Any

import aiohttp
import msgspec
import pytest

from tests.orderbook.standard.binance_replay import (
    BookTickerEvent,
    DepthEvent,
    _replay_depth_with_reference,
    _replay_interleaved_with_book_ticker,
)


async def _fetch_exchange_filters(
    session: aiohttp.ClientSession,
    symbol: str,
) -> tuple[str, str]:
    """Fetch tick_size and lot_size from Binance exchange info.

    Args:
        session: aiohttp client session.
        symbol: Trading pair symbol (e.g., "BTCUSDT").

    Returns:
        Tuple of (tick_size, lot_size).
    """
    url = "https://api.binance.com/api/v3/exchangeInfo"
    async with session.get(url, params={"symbol": symbol}, timeout=15) as resp:
        resp.raise_for_status()
        payload = await resp.json()

    symbols = payload.get("symbols", [])
    assert len(symbols) == 1, (
        f"Expected one exchangeInfo symbol entry, got {len(symbols)}"
    )

    filters = symbols[0].get("filters", [])
    price_filter = next(
        (f for f in filters if f.get("filterType") == "PRICE_FILTER"), None
    )
    lot_filter = next((f for f in filters if f.get("filterType") == "LOT_SIZE"), None)
    assert price_filter is not None, "Missing PRICE_FILTER"
    assert lot_filter is not None, "Missing LOT_SIZE"

    tick_size = price_filter["tickSize"]
    lot_size = lot_filter["stepSize"]
    assert Decimal(tick_size) > 0
    assert Decimal(lot_size) > 0
    return tick_size, lot_size


async def _fetch_snapshot(
    session: aiohttp.ClientSession,
    symbol: str,
    limit: int,
) -> dict[str, Any]:
    """Fetch orderbook snapshot from Binance.

    Args:
        session: aiohttp client session.
        symbol: Trading pair symbol.
        limit: Number of levels to fetch.

    Returns:
        Snapshot dictionary with lastUpdateId, bids, and asks.
    """
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
    str,
    str,
    list[DepthEvent],
    list[BookTickerEvent],
    list[DepthEvent | BookTickerEvent],
]:
    """Capture live Binance depth and bookTicker events.

    Args:
        symbol: Trading pair symbol.
        duration_s: Capture duration in seconds.
        snapshot_limit: Number of levels for initial snapshot.

    Returns:
        Tuple of (snapshot, tick_size, lot_size, depth_events, ticker_events, ordered_events).
    """
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

                    if stream.endswith("@bookTicker") or (
                        "u" in data
                        and "b" in data
                        and "B" in data
                        and "a" in data
                        and "A" in data
                        and "U" not in data
                    ):
                        evt = BookTickerEvent(
                            seq=seq,
                            u=int(data["u"]),
                            bid_price=data["b"],
                            bid_qty=data["B"],
                            ask_price=data["a"],
                            ask_qty=data["A"],
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
    str,
    str,
    list[DepthEvent],
    list[BookTickerEvent],
    list[DepthEvent | BookTickerEvent],
]:
    """Capture live events with retry logic.

    Args:
        symbol: Trading pair symbol.
        duration_s: Capture duration in seconds.
        snapshot_limit: Number of levels for initial snapshot.
        max_attempts: Maximum retry attempts.

    Returns:
        Same tuple as _capture_events.
    """
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return await _capture_events(symbol, duration_s, snapshot_limit)
        except (
            aiohttp.ClientError,
            asyncio.TimeoutError,
            OSError,
            RuntimeError,
        ) as exc:
            last_exc = exc
            if attempt + 1 == max_attempts:
                break
            await asyncio.sleep(1.0 + attempt)

    raise AssertionError(
        f"Unable to capture live Binance events after retries: {last_exc}"
    )


@pytest.mark.live
@pytest.mark.slow
@pytest.mark.asyncio
async def test_live_binance_btcusdt_depth_and_bookticker_zero_faults(
    request: pytest.FixtureRequest,
) -> None:
    """Validate live Binance BTCUSDT updates with strict zero-fault replay checks.

    Captures live market data, replays depth events against a reference implementation,
    and validates interleaved depth+bookTicker handling with zero tolerance for mismatches.
    """
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
