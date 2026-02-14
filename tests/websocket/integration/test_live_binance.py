"""Live Binance Futures integration tests for websocket components.

Run with: pytest tests/websocket/integration/test_live_binance.py --run-live
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from time import time_ns
from typing import Any, Callable

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


@dataclass(slots=True)
class LatencyStats:
    """Latency summary statistics in milliseconds."""

    name: str
    count: int
    min_ms: int
    p50_ms: float
    p90_ms: float
    p99_ms: float
    max_ms: int
    mean_ms: float


class _StartBarrier:
    """Simple async barrier used to align side-by-side sample windows."""

    def __init__(self, parties: int) -> None:
        self._parties = parties
        self._count = 0
        self._lock = asyncio.Lock()
        self._event = asyncio.Event()

    async def ready_and_wait(self) -> float:
        """Block until all parties are ready, then return aligned start time."""
        async with self._lock:
            self._count += 1
            if self._count == self._parties:
                self._event.set()
        await self._event.wait()
        return asyncio.get_running_loop().time()


class _LiveBinanceHarness:
    """Shared helpers for live Binance websocket integration tests."""

    BOOK_TICKER_DECODER = msgspec.json.Decoder(
        type=BinanceFuturesBookTicker,
        strict=True,
    )
    TRADE_DECODER = msgspec.json.Decoder(type=BinanceFuturesTrade, strict=True)
    COMBINED_ENVELOPE_DECODER = msgspec.json.Decoder(
        type=BinanceCombinedStreamEnvelope,
        strict=True,
    )

    @staticmethod
    def _noop_message_handler(msg: bytes) -> None:
        """No-op callback required by WsPool constructor."""
        return None

    @staticmethod
    def _to_decimal(value: str, field: str) -> Decimal:
        """Convert a numeric string to Decimal for strict numeric checks."""
        try:
            return Decimal(value)
        except InvalidOperation as exc:
            raise AssertionError(f"Invalid decimal in field '{field}': {value}") from exc

    @staticmethod
    def _now_ms() -> int:
        """Current unix timestamp in milliseconds."""
        return time_ns() // 1_000_000

    @classmethod
    def _validate_timestamps(
        cls,
        event_time_ms: int,
        transaction_time_ms: int,
        *,
        freshness_budget_ms: int,
        clock_future_drift_ms: int,
    ) -> None:
        """Validate event freshness and basic clock sanity."""
        recv_ms = cls._now_ms()
        assert event_time_ms > 0
        assert transaction_time_ms > 0
        assert event_time_ms <= recv_ms + clock_future_drift_ms
        assert transaction_time_ms <= recv_ms + clock_future_drift_ms
        assert recv_ms - event_time_ms <= freshness_budget_ms
        assert recv_ms - transaction_time_ms <= freshness_budget_ms

    @classmethod
    def _validate_book_ticker(
        cls,
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
        cls._validate_timestamps(
            data.E,
            data.T,
            freshness_budget_ms=freshness_budget_ms,
            clock_future_drift_ms=clock_future_drift_ms,
        )

        bid = cls._to_decimal(data.b, "b")
        bid_qty = cls._to_decimal(data.B, "B")
        ask = cls._to_decimal(data.a, "a")
        ask_qty = cls._to_decimal(data.A, "A")
        assert bid > 0
        assert ask > 0
        assert bid <= ask
        assert bid_qty >= 0
        assert ask_qty >= 0

    @classmethod
    def _validate_trade(
        cls,
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
        cls._validate_timestamps(
            data.E,
            data.T,
            freshness_budget_ms=freshness_budget_ms,
            clock_future_drift_ms=clock_future_drift_ms,
        )

        price = cls._to_decimal(data.p, "p")
        qty = cls._to_decimal(data.q, "q")
        if price == 0 or qty == 0:
            # Binance futures occasionally emits placeholder trade records on this
            # stream shape. Treat them as valid only when fully zeroed and marked
            # with the expected placeholder trade type.
            assert price == 0
            assert qty == 0
            assert data.X == "NA"
        else:
            assert price > 0
            assert qty > 0

    @staticmethod
    async def _wait_for_state(
        get_state: Callable[[], ConnectionState],
        expected: ConnectionState,
        *,
        timeout_s: float,
    ) -> None:
        """Wait for a websocket wrapper to reach an expected state."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        while loop.time() < deadline:
            if get_state() == expected:
                return
            await asyncio.sleep(0.05)
        raise AssertionError(f"Timed out waiting for state={expected}")

    @staticmethod
    async def _wait_for_pool_connections(pool: WsPool, *, timeout_s: float) -> None:
        """Wait until a pool reports at least one connected websocket."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        while loop.time() < deadline:
            if pool.get_connection_count() > 0:
                return
            await asyncio.sleep(0.05)
        raise AssertionError("Timed out waiting for active pool connections")

    @staticmethod
    async def _next_message(stream: WsSingle | WsPool, *, timeout_s: float) -> bytes:
        """Receive one websocket payload from an async iterator wrapper."""
        return await asyncio.wait_for(stream.__anext__(), timeout=timeout_s)

    @staticmethod
    def _build_expected_streams(
        symbols: list[str], stream_kinds: list[str]
    ) -> tuple[list[str], set[str]]:
        """Create combined-stream URL params and normalized expected stream names."""
        url_streams = [f"{symbol}@{kind}" for symbol in symbols for kind in stream_kinds]
        expected_streams = {stream.lower() for stream in url_streams}
        return url_streams, expected_streams

    @classmethod
    def _decode_and_validate_combined_payload(
        cls,
        payload: bytes,
        *,
        expected_streams: set[str],
        expected_symbols: set[str],
        freshness_budget_ms: int,
        clock_future_drift_ms: int,
    ) -> str:
        """Decode one combined-stream payload and validate by stream type."""
        stream_name, _ = cls._decode_and_validate_combined_payload_with_event_time(
            payload,
            expected_streams=expected_streams,
            expected_symbols=expected_symbols,
            freshness_budget_ms=freshness_budget_ms,
            clock_future_drift_ms=clock_future_drift_ms,
        )
        return stream_name

    @classmethod
    def _decode_and_validate_combined_payload_with_event_time(
        cls,
        payload: bytes,
        *,
        expected_streams: set[str],
        expected_symbols: set[str],
        freshness_budget_ms: int,
        clock_future_drift_ms: int,
        strict_trade_values: bool = True,
    ) -> tuple[str, int]:
        """Decode payload and return normalized stream and event timestamp."""
        envelope = cls.COMBINED_ENVELOPE_DECODER.decode(payload)
        stream_name = envelope.stream.lower()
        assert stream_name in expected_streams

        inner_payload = bytes(envelope.data)
        if stream_name.endswith("@bookticker"):
            event = cls.BOOK_TICKER_DECODER.decode(inner_payload)
            cls._validate_book_ticker(
                event,
                expected_symbols=expected_symbols,
                freshness_budget_ms=freshness_budget_ms,
                clock_future_drift_ms=clock_future_drift_ms,
            )
            return stream_name, event.E

        if stream_name.endswith("@trade"):
            event = cls.TRADE_DECODER.decode(inner_payload)
            if strict_trade_values:
                cls._validate_trade(
                    event,
                    expected_symbols=expected_symbols,
                    freshness_budget_ms=freshness_budget_ms,
                    clock_future_drift_ms=clock_future_drift_ms,
                )
            else:
                # Live futures feeds can emit typed placeholder trades with zero
                # price/quantity. Keep strict schema/timestamp checks but allow
                # non-negative economics for latency benchmarking.
                assert event.e == "trade"
                assert event.s in expected_symbols
                assert event.t > 0
                assert event.X != ""
                cls._validate_timestamps(
                    event.E,
                    event.T,
                    freshness_budget_ms=freshness_budget_ms,
                    clock_future_drift_ms=clock_future_drift_ms,
                )
                assert cls._to_decimal(event.p, "p") >= 0
                assert cls._to_decimal(event.q, "q") >= 0
            return stream_name, event.E

        raise AssertionError(f"Unexpected stream received: {stream_name}")

    @staticmethod
    def _percentile(sorted_samples: list[int], quantile: float) -> float:
        """Linear-interpolated percentile over pre-sorted integer samples."""
        if not sorted_samples:
            raise AssertionError("Cannot compute percentile from empty samples")
        assert 0.0 <= quantile <= 1.0
        if len(sorted_samples) == 1:
            return float(sorted_samples[0])

        rank = (len(sorted_samples) - 1) * quantile
        lo = int(rank)
        hi = min(lo + 1, len(sorted_samples) - 1)
        weight = rank - lo
        return float(sorted_samples[lo]) * (1.0 - weight) + float(
            sorted_samples[hi]
        ) * weight

    @classmethod
    def _build_latency_stats(cls, name: str, samples_ms: list[int]) -> LatencyStats:
        """Build latency summary statistics from millisecond samples."""
        if not samples_ms:
            raise AssertionError(f"No latency samples collected for {name}")
        sorted_samples = sorted(samples_ms)
        return LatencyStats(
            name=name,
            count=len(sorted_samples),
            min_ms=sorted_samples[0],
            p50_ms=cls._percentile(sorted_samples, 0.50),
            p90_ms=cls._percentile(sorted_samples, 0.90),
            p99_ms=cls._percentile(sorted_samples, 0.99),
            max_ms=sorted_samples[-1],
            mean_ms=float(sum(sorted_samples)) / float(len(sorted_samples)),
        )

    @staticmethod
    def _format_latency_stats(stats: LatencyStats) -> str:
        """Format latency statistics for human-readable live output."""
        return (
            f"{stats.name}: n={stats.count} mean={stats.mean_ms:.2f}ms "
            f"p50={stats.p50_ms:.2f}ms p90={stats.p90_ms:.2f}ms "
            f"p99={stats.p99_ms:.2f}ms min={stats.min_ms}ms max={stats.max_ms}ms"
        )

    @classmethod
    async def _collect_until_coverage(
        cls,
        stream: WsSingle | WsPool,
        *,
        expected_streams: set[str],
        expected_symbols: set[str],
        timeout_s: float,
        freshness_budget_ms: int,
        clock_future_drift_ms: int,
    ) -> dict[str, int]:
        """Collect combined stream messages until all expected streams are observed."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        counts: dict[str, int] = {}

        while loop.time() < deadline and len(counts) < len(expected_streams):
            remaining = deadline - loop.time()
            payload = await cls._next_message(stream, timeout_s=max(0.1, remaining))
            stream_name = cls._decode_and_validate_combined_payload(
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

    @classmethod
    async def _collect_latency_samples(
        cls,
        stream: WsSingle | WsPool,
        *,
        expected_streams: set[str],
        expected_symbols: set[str],
        sample_window_s: float,
        freshness_budget_ms: int,
        clock_future_drift_ms: int,
        barrier: _StartBarrier,
    ) -> tuple[list[int], dict[str, int]]:
        """Collect per-message latency samples over a synchronized time window."""
        loop = asyncio.get_running_loop()
        start = await barrier.ready_and_wait()
        deadline = start + sample_window_s
        latencies_ms: list[int] = []
        counts: dict[str, int] = {}

        while loop.time() < deadline:
            remaining = deadline - loop.time()
            payload = await cls._next_message(
                stream,
                timeout_s=max(0.1, min(2.0, remaining)),
            )
            stream_name, event_time_ms = (
                cls._decode_and_validate_combined_payload_with_event_time(
                    payload,
                    expected_streams=expected_streams,
                    expected_symbols=expected_symbols,
                    freshness_budget_ms=freshness_budget_ms,
                    clock_future_drift_ms=clock_future_drift_ms,
                    strict_trade_values=False,
                )
            )
            recv_ms = cls._now_ms()
            latency_ms = recv_ms - event_time_ms
            assert latency_ms >= -clock_future_drift_ms
            assert latency_ms <= freshness_budget_ms
            latencies_ms.append(latency_ms)
            counts[stream_name] = counts.get(stream_name, 0) + 1

        return latencies_ms, counts


@pytest.mark.asyncio
@pytest.mark.live
class TestLiveBinanceFutures(_LiveBinanceHarness):
    """Live Binance futures tests for WsSingle and WsPool."""

    async def test_single_btc_bookticker_smoke(
        self,
        live_test_config: dict[str, Any],
        live_timeout_s: float,
    ) -> None:
        """Validate WsSingle on real BTC futures @bookTicker feed."""
        config = WsConnectionConfig.default(
            f"{live_test_config['binance_futures_base']}/btcusdt@bookTicker"
        )
        ws = WsSingle(config)
        timeout_s = max(6.0, min(live_timeout_s, 20.0))

        valid_count = 0
        async with ws:
            await self._wait_for_state(
                ws.get_state,
                ConnectionState.CONNECTED,
                timeout_s=10.0,
            )
            deadline = asyncio.get_running_loop().time() + timeout_s
            while asyncio.get_running_loop().time() < deadline and valid_count < 3:
                payload = await self._next_message(ws, timeout_s=1.0)
                event = self.BOOK_TICKER_DECODER.decode(payload)
                self._validate_book_ticker(
                    event,
                    expected_symbols={"BTCUSDT"},
                    freshness_budget_ms=live_test_config["freshness_budget_ms"],
                    clock_future_drift_ms=live_test_config["clock_future_drift_ms"],
                )
                valid_count += 1

        assert valid_count >= 3

    async def test_pool_btc_bookticker_smoke(
        self,
        live_test_config: dict[str, Any],
        live_timeout_s: float,
    ) -> None:
        """Validate WsPool on real BTC futures @bookTicker feed."""
        config = WsConnectionConfig.default(
            f"{live_test_config['binance_futures_base']}/btcusdt@bookTicker"
        )
        pool = await WsPool.new(
            config=config,
            on_message=self._noop_message_handler,
            pool_config=WsPoolConfig(num_connections=2, evict_interval_s=60),
        )
        timeout_s = max(6.0, min(live_timeout_s, 20.0))

        valid_count = 0
        async with pool:
            await self._wait_for_state(
                pool.get_state,
                ConnectionState.CONNECTED,
                timeout_s=10.0,
            )
            await self._wait_for_pool_connections(pool, timeout_s=10.0)
            deadline = asyncio.get_running_loop().time() + timeout_s
            while asyncio.get_running_loop().time() < deadline and valid_count < 3:
                payload = await self._next_message(pool, timeout_s=1.0)
                event = self.BOOK_TICKER_DECODER.decode(payload)
                self._validate_book_ticker(
                    event,
                    expected_symbols={"BTCUSDT"},
                    freshness_budget_ms=live_test_config["freshness_budget_ms"],
                    clock_future_drift_ms=live_test_config["clock_future_drift_ms"],
                )
                valid_count += 1

        assert valid_count >= 3

    @pytest.mark.stress
    async def test_single_combined_stream_load_realtime(
        self,
        live_test_config: dict[str, Any],
        live_timeout_s: float,
    ) -> None:
        """Validate WsSingle under combined BTC/ETH/SOL bookTicker+trade load."""
        symbols = live_test_config["test_symbols"]
        stream_kinds = live_test_config["stream_kinds"]
        url_streams, expected_streams = self._build_expected_streams(
            symbols,
            stream_kinds,
        )
        expected_symbols = {symbol.upper() for symbol in symbols}
        combined_url = (
            f"{live_test_config['binance_futures_combined_base']}{'/'.join(url_streams)}"
        )

        ws = WsSingle(WsConnectionConfig.default(combined_url))
        async with ws:
            await self._wait_for_state(
                ws.get_state,
                ConnectionState.CONNECTED,
                timeout_s=10.0,
            )
            counts = await self._collect_until_coverage(
                ws,
                expected_streams=expected_streams,
                expected_symbols=expected_symbols,
                timeout_s=max(10.0, live_timeout_s),
                freshness_budget_ms=live_test_config["freshness_budget_ms"],
                clock_future_drift_ms=live_test_config["clock_future_drift_ms"],
            )

        assert sum(counts.values()) >= len(expected_streams)

    @pytest.mark.stress
    async def test_pool_combined_stream_load_realtime(
        self,
        live_test_config: dict[str, Any],
        live_timeout_s: float,
    ) -> None:
        """Validate WsPool under combined BTC/ETH/SOL bookTicker+trade load."""
        symbols = live_test_config["test_symbols"]
        stream_kinds = live_test_config["stream_kinds"]
        url_streams, expected_streams = self._build_expected_streams(
            symbols,
            stream_kinds,
        )
        expected_symbols = {symbol.upper() for symbol in symbols}
        combined_url = (
            f"{live_test_config['binance_futures_combined_base']}{'/'.join(url_streams)}"
        )

        pool = await WsPool.new(
            config=WsConnectionConfig.default(combined_url),
            on_message=self._noop_message_handler,
            pool_config=WsPoolConfig(num_connections=3, evict_interval_s=60),
        )
        async with pool:
            await self._wait_for_state(
                pool.get_state,
                ConnectionState.CONNECTED,
                timeout_s=10.0,
            )
            await self._wait_for_pool_connections(pool, timeout_s=10.0)
            counts = await self._collect_until_coverage(
                pool,
                expected_streams=expected_streams,
                expected_symbols=expected_symbols,
                timeout_s=max(10.0, live_timeout_s),
                freshness_budget_ms=live_test_config["freshness_budget_ms"],
                clock_future_drift_ms=live_test_config["clock_future_drift_ms"],
            )

        assert sum(counts.values()) >= len(expected_streams)

    @pytest.mark.stress
    async def test_side_by_side_latency_stats_single_vs_pool(
        self,
        live_test_config: dict[str, Any],
        live_timeout_s: float,
    ) -> None:
        """Measure side-by-side event-time latency stats for single vs pool."""
        symbols = live_test_config["test_symbols"]
        stream_kinds = live_test_config["stream_kinds"]
        url_streams, expected_streams = self._build_expected_streams(
            symbols,
            stream_kinds,
        )
        expected_symbols = {symbol.upper() for symbol in symbols}
        combined_url = (
            f"{live_test_config['binance_futures_combined_base']}{'/'.join(url_streams)}"
        )

        sample_window_s = max(8.0, min(live_timeout_s, 20.0))
        barrier = _StartBarrier(parties=2)

        ws = WsSingle(WsConnectionConfig.default(combined_url))
        pool = await WsPool.new(
            config=WsConnectionConfig.default(combined_url),
            on_message=self._noop_message_handler,
            pool_config=WsPoolConfig(num_connections=3, evict_interval_s=60),
        )

        async with ws, pool:
            await self._wait_for_state(
                ws.get_state,
                ConnectionState.CONNECTED,
                timeout_s=10.0,
            )
            await self._wait_for_state(
                pool.get_state,
                ConnectionState.CONNECTED,
                timeout_s=10.0,
            )
            await self._wait_for_pool_connections(pool, timeout_s=10.0)

            single_result, pool_result = await asyncio.gather(
                self._collect_latency_samples(
                    ws,
                    expected_streams=expected_streams,
                    expected_symbols=expected_symbols,
                    sample_window_s=sample_window_s,
                    freshness_budget_ms=live_test_config["freshness_budget_ms"],
                    clock_future_drift_ms=live_test_config["clock_future_drift_ms"],
                    barrier=barrier,
                ),
                self._collect_latency_samples(
                    pool,
                    expected_streams=expected_streams,
                    expected_symbols=expected_symbols,
                    sample_window_s=sample_window_s,
                    freshness_budget_ms=live_test_config["freshness_budget_ms"],
                    clock_future_drift_ms=live_test_config["clock_future_drift_ms"],
                    barrier=barrier,
                ),
            )

        single_latencies, single_counts = single_result
        pool_latencies, pool_counts = pool_result

        missing_single = expected_streams - set(single_counts)
        missing_pool = expected_streams - set(pool_counts)
        assert not missing_single, f"Single missing streams: {sorted(missing_single)}"
        assert not missing_pool, f"Pool missing streams: {sorted(missing_pool)}"

        single_stats = self._build_latency_stats("single", single_latencies)
        pool_stats = self._build_latency_stats("pool", pool_latencies)

        # Comparison is informative (internet jitter can dominate in any one run);
        # pass criteria remain deterministic schema/freshness and stream coverage.
        print("Latency comparison (event time -> local receive time):")
        print(self._format_latency_stats(single_stats))
        print(self._format_latency_stats(pool_stats))
        print(
            "delta (single - pool) ms: "
            f"p50={single_stats.p50_ms - pool_stats.p50_ms:.2f}, "
            f"p90={single_stats.p90_ms - pool_stats.p90_ms:.2f}, "
            f"p99={single_stats.p99_ms - pool_stats.p99_ms:.2f}"
        )

        assert single_stats.p99_ms <= live_test_config["freshness_budget_ms"]
        assert pool_stats.p99_ms <= live_test_config["freshness_budget_ms"]
