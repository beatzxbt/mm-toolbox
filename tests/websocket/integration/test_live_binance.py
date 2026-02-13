"""Live Binance Futures integration tests for websocket components.

Run with: pytest tests/websocket/integration/test_live_binance.py --run-live
"""

from __future__ import annotations

import asyncio
from decimal import Decimal, InvalidOperation
from time import time_ns
from typing import Any

import msgspec
import pytest

from mm_toolbox.websocket.connection import ConnectionState, WsConnectionConfig
from mm_toolbox.websocket.pool import WsPool, WsPoolConfig
from mm_toolbox.websocket.single import WsSingle


class BinanceFuturesBookTicker(msgspec.Struct):
    """Strict schema for Binance Futures @bookTicker events."""

    e: str
    u: int
    E: int
    T: int
    s: str
    b: str
    B: str
    a: str
    A: str


class BinanceFuturesTrade(msgspec.Struct):
    """Strict schema for Binance Futures @trade events."""

    e: str
    E: int
    T: int
    s: str
    t: int
    p: str
    q: str
    X: str
    m: bool


class BinanceCombinedStreamEnvelope(msgspec.Struct):
    """Strict schema for Binance combined stream envelope."""

    stream: str
    data: msgspec.Raw


BOOK_TICKER_DECODER = msgspec.json.Decoder(type=BinanceFuturesBookTicker, strict=True)
TRADE_DECODER = msgspec.json.Decoder(type=BinanceFuturesTrade, strict=True)
COMBINED_ENVELOPE_DECODER = msgspec.json.Decoder(
    type=BinanceCombinedStreamEnvelope, strict=True
)


def _noop_message_handler(msg: bytes) -> None:
    """No-op callback required by WsPool constructor.

    Args:
        msg (bytes): Message payload.

    Returns:
        None: This callback does not return a value.
    """
    return None


def _to_decimal(value: str, field: str) -> Decimal:
    """Convert a numeric string to Decimal for strict numeric checks.

    Args:
        value (str): Numeric string.
        field (str): Field name for assertion messages.

    Returns:
        Decimal: Converted value.
    """
    try:
        return Decimal(value)
    except InvalidOperation as exc:
        raise AssertionError(f"Invalid decimal in field '{field}': {value}") from exc


def _now_ms() -> int:
    """Current unix timestamp in milliseconds."""
    return time_ns() // 1_000_000


def _validate_timestamps(
    event_time_ms: int,
    transaction_time_ms: int,
    *,
    freshness_budget_ms: int,
    clock_future_drift_ms: int,
) -> None:
    """Validate event freshness and basic clock sanity.

    Args:
        event_time_ms (int): Event timestamp from exchange.
        transaction_time_ms (int): Transaction timestamp from exchange.
        freshness_budget_ms (int): Max allowed staleness window.
        clock_future_drift_ms (int): Allowed positive drift versus local clock.

    Returns:
        None: This helper does not return a value.
    """
    recv_ms = _now_ms()
    assert event_time_ms > 0
    assert transaction_time_ms > 0
    assert event_time_ms <= recv_ms + clock_future_drift_ms
    assert transaction_time_ms <= recv_ms + clock_future_drift_ms
    assert recv_ms - event_time_ms <= freshness_budget_ms
    assert recv_ms - transaction_time_ms <= freshness_budget_ms


def _validate_book_ticker(
    data: BinanceFuturesBookTicker,
    *,
    expected_symbols: set[str],
    freshness_budget_ms: int,
    clock_future_drift_ms: int,
) -> None:
    """Validate strict semantics for book ticker events."""
    assert data.e == "bookTicker"
    assert data.s in expected_symbols
    assert data.u > 0
    _validate_timestamps(
        data.E,
        data.T,
        freshness_budget_ms=freshness_budget_ms,
        clock_future_drift_ms=clock_future_drift_ms,
    )

    bid = _to_decimal(data.b, "b")
    bid_qty = _to_decimal(data.B, "B")
    ask = _to_decimal(data.a, "a")
    ask_qty = _to_decimal(data.A, "A")
    assert bid > 0
    assert ask > 0
    assert bid <= ask
    assert bid_qty >= 0
    assert ask_qty >= 0


def _validate_trade(
    data: BinanceFuturesTrade,
    *,
    expected_symbols: set[str],
    freshness_budget_ms: int,
    clock_future_drift_ms: int,
) -> None:
    """Validate strict semantics for raw trade events."""
    assert data.e == "trade"
    assert data.s in expected_symbols
    assert data.t > 0
    assert data.X != ""
    _validate_timestamps(
        data.E,
        data.T,
        freshness_budget_ms=freshness_budget_ms,
        clock_future_drift_ms=clock_future_drift_ms,
    )

    price = _to_decimal(data.p, "p")
    qty = _to_decimal(data.q, "q")
    assert price > 0
    assert qty > 0


async def _wait_for_state(
    get_state,
    expected: ConnectionState,
    *,
    timeout_s: float,
) -> None:
    """Wait for a websocket wrapper to reach an expected state.

    Args:
        get_state: Callable returning a ConnectionState.
        expected (ConnectionState): Expected state.
        timeout_s (float): Timeout in seconds.

    Returns:
        None: This helper does not return a value.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while loop.time() < deadline:
        if get_state() == expected:
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"Timed out waiting for state={expected}")


async def _wait_for_pool_connections(pool: WsPool, *, timeout_s: float) -> None:
    """Wait until a pool reports at least one connected websocket."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while loop.time() < deadline:
        if pool.get_connection_count() > 0:
            return
        await asyncio.sleep(0.05)
    raise AssertionError("Timed out waiting for active pool connections")


async def _next_message(stream: WsSingle | WsPool, *, timeout_s: float) -> bytes:
    """Receive one websocket payload from an async iterator wrapper."""
    return await asyncio.wait_for(stream.__anext__(), timeout=timeout_s)


def _build_expected_streams(
    symbols: list[str], stream_kinds: list[str]
) -> tuple[list[str], set[str]]:
    """Create combined-stream URL params and normalized expected stream names.

    Args:
        symbols (list[str]): Lowercase futures symbols.
        stream_kinds (list[str]): Stream kinds (e.g. bookTicker, trade).

    Returns:
        tuple[list[str], set[str]]: URL stream params and lowercase expected set.
    """
    url_streams = [f"{symbol}@{kind}" for symbol in symbols for kind in stream_kinds]
    expected_streams = {stream.lower() for stream in url_streams}
    return url_streams, expected_streams


def _decode_and_validate_combined_payload(
    payload: bytes,
    *,
    expected_streams: set[str],
    expected_symbols: set[str],
    freshness_budget_ms: int,
    clock_future_drift_ms: int,
) -> str:
    """Decode one combined-stream payload and validate by stream type.

    Args:
        payload (bytes): Raw websocket payload.
        expected_streams (set[str]): Required lowercase stream names.
        expected_symbols (set[str]): Required uppercase symbols.
        freshness_budget_ms (int): Max allowed staleness.
        clock_future_drift_ms (int): Allowed positive drift versus local clock.

    Returns:
        str: Lowercase stream name (e.g. btcusdt@trade).
    """
    envelope = COMBINED_ENVELOPE_DECODER.decode(payload)
    stream_name = envelope.stream.lower()
    assert stream_name in expected_streams

    inner_payload = bytes(envelope.data)
    if stream_name.endswith("@bookticker"):
        _validate_book_ticker(
            BOOK_TICKER_DECODER.decode(inner_payload),
            expected_symbols=expected_symbols,
            freshness_budget_ms=freshness_budget_ms,
            clock_future_drift_ms=clock_future_drift_ms,
        )
    elif stream_name.endswith("@trade"):
        _validate_trade(
            TRADE_DECODER.decode(inner_payload),
            expected_symbols=expected_symbols,
            freshness_budget_ms=freshness_budget_ms,
            clock_future_drift_ms=clock_future_drift_ms,
        )
    else:
        raise AssertionError(f"Unexpected stream received: {stream_name}")

    return stream_name


async def _collect_until_coverage(
    stream: WsSingle | WsPool,
    *,
    expected_streams: set[str],
    expected_symbols: set[str],
    timeout_s: float,
    freshness_budget_ms: int,
    clock_future_drift_ms: int,
) -> dict[str, int]:
    """Collect combined stream messages until all expected streams are observed.

    Args:
        stream (WsSingle | WsPool): Message source.
        expected_streams (set[str]): Required lowercase stream names.
        expected_symbols (set[str]): Required uppercase symbols.
        timeout_s (float): Collection timeout.
        freshness_budget_ms (int): Max allowed staleness.
        clock_future_drift_ms (int): Allowed positive drift versus local clock.

    Returns:
        dict[str, int]: Per-stream message counts.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    counts: dict[str, int] = {}

    while loop.time() < deadline and len(counts) < len(expected_streams):
        remaining = deadline - loop.time()
        payload = await _next_message(stream, timeout_s=max(0.1, remaining))
        stream_name = _decode_and_validate_combined_payload(
            payload,
            expected_streams=expected_streams,
            expected_symbols=expected_symbols,
            freshness_budget_ms=freshness_budget_ms,
            clock_future_drift_ms=clock_future_drift_ms,
        )
        counts[stream_name] = counts.get(stream_name, 0) + 1

    missing_streams = expected_streams - set(counts)
    assert not missing_streams, f"Missing streams: {sorted(missing_streams)}"
    return counts


@pytest.mark.asyncio
@pytest.mark.live
async def test_single_btc_bookticker_smoke(
    live_test_config: dict[str, Any], live_timeout_s: float
) -> None:
    """Validate WsSingle on real BTC futures @bookTicker feed."""
    config = WsConnectionConfig.default(
        f"{live_test_config['binance_futures_base']}/btcusdt@bookTicker"
    )
    ws = WsSingle(config)
    timeout_s = max(6.0, min(live_timeout_s, 20.0))

    valid_count = 0
    async with ws:
        await _wait_for_state(ws.get_state, ConnectionState.CONNECTED, timeout_s=10.0)
        deadline = asyncio.get_running_loop().time() + timeout_s
        while asyncio.get_running_loop().time() < deadline and valid_count < 3:
            payload = await _next_message(ws, timeout_s=1.0)
            event = BOOK_TICKER_DECODER.decode(payload)
            _validate_book_ticker(
                event,
                expected_symbols={"BTCUSDT"},
                freshness_budget_ms=live_test_config["freshness_budget_ms"],
                clock_future_drift_ms=live_test_config["clock_future_drift_ms"],
            )
            valid_count += 1

    assert valid_count >= 3


@pytest.mark.asyncio
@pytest.mark.live
async def test_pool_btc_bookticker_smoke(
    live_test_config: dict[str, Any], live_timeout_s: float
) -> None:
    """Validate WsPool on real BTC futures @bookTicker feed."""
    config = WsConnectionConfig.default(
        f"{live_test_config['binance_futures_base']}/btcusdt@bookTicker"
    )
    pool = await WsPool.new(
        config=config,
        on_message=_noop_message_handler,
        pool_config=WsPoolConfig(num_connections=2, evict_interval_s=60),
    )
    timeout_s = max(6.0, min(live_timeout_s, 20.0))

    valid_count = 0
    async with pool:
        await _wait_for_state(
            pool.get_state, ConnectionState.CONNECTED, timeout_s=10.0
        )
        await _wait_for_pool_connections(pool, timeout_s=10.0)
        deadline = asyncio.get_running_loop().time() + timeout_s
        while asyncio.get_running_loop().time() < deadline and valid_count < 3:
            payload = await _next_message(pool, timeout_s=1.0)
            event = BOOK_TICKER_DECODER.decode(payload)
            _validate_book_ticker(
                event,
                expected_symbols={"BTCUSDT"},
                freshness_budget_ms=live_test_config["freshness_budget_ms"],
                clock_future_drift_ms=live_test_config["clock_future_drift_ms"],
            )
            valid_count += 1

    assert valid_count >= 3


@pytest.mark.asyncio
@pytest.mark.live
@pytest.mark.stress
async def test_single_combined_stream_load_realtime(
    live_test_config: dict[str, Any], live_timeout_s: float
) -> None:
    """Validate WsSingle under combined BTC/ETH/SOL bookTicker+trade load."""
    symbols = live_test_config["test_symbols"]
    stream_kinds = live_test_config["stream_kinds"]
    url_streams, expected_streams = _build_expected_streams(symbols, stream_kinds)
    expected_symbols = {symbol.upper() for symbol in symbols}
    combined_url = (
        f"{live_test_config['binance_futures_combined_base']}{'/'.join(url_streams)}"
    )

    ws = WsSingle(WsConnectionConfig.default(combined_url))
    async with ws:
        await _wait_for_state(ws.get_state, ConnectionState.CONNECTED, timeout_s=10.0)
        counts = await _collect_until_coverage(
            ws,
            expected_streams=expected_streams,
            expected_symbols=expected_symbols,
            timeout_s=max(10.0, live_timeout_s),
            freshness_budget_ms=live_test_config["freshness_budget_ms"],
            clock_future_drift_ms=live_test_config["clock_future_drift_ms"],
        )

    assert sum(counts.values()) >= len(expected_streams)


@pytest.mark.asyncio
@pytest.mark.live
@pytest.mark.stress
async def test_pool_combined_stream_load_realtime(
    live_test_config: dict[str, Any], live_timeout_s: float
) -> None:
    """Validate WsPool under combined BTC/ETH/SOL bookTicker+trade load."""
    symbols = live_test_config["test_symbols"]
    stream_kinds = live_test_config["stream_kinds"]
    url_streams, expected_streams = _build_expected_streams(symbols, stream_kinds)
    expected_symbols = {symbol.upper() for symbol in symbols}
    combined_url = (
        f"{live_test_config['binance_futures_combined_base']}{'/'.join(url_streams)}"
    )

    pool = await WsPool.new(
        config=WsConnectionConfig.default(combined_url),
        on_message=_noop_message_handler,
        pool_config=WsPoolConfig(num_connections=3, evict_interval_s=60),
    )
    async with pool:
        await _wait_for_state(
            pool.get_state, ConnectionState.CONNECTED, timeout_s=10.0
        )
        await _wait_for_pool_connections(pool, timeout_s=10.0)
        counts = await _collect_until_coverage(
            pool,
            expected_streams=expected_streams,
            expected_symbols=expected_symbols,
            timeout_s=max(10.0, live_timeout_s),
            freshness_budget_ms=live_test_config["freshness_budget_ms"],
            clock_future_drift_ms=live_test_config["clock_future_drift_ms"],
        )

    assert sum(counts.values()) >= len(expected_streams)
