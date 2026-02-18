"""Benchmarks live Binance Futures websocket throughput and latency.

Usage:
    uv run python benchmarks/websocket/benchmark_live_binance.py
"""

from __future__ import annotations

import asyncio
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

import msgspec

from mm_toolbox.websocket.connection import ConnectionState, WsConnectionConfig
from mm_toolbox.websocket.pool import WsPool, WsPoolConfig
from mm_toolbox.websocket.single import WsSingle

try:
    from benchmarks.core import (
        BaseBenchmarkConfig,
        BenchmarkCLI,
        BenchmarkReporter,
        BenchmarkRunner,
    )
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from benchmarks.core import (
        BaseBenchmarkConfig,
        BenchmarkCLI,
        BenchmarkReporter,
        BenchmarkRunner,
    )


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


@dataclass
class LiveBinanceBenchmarkConfig(BaseBenchmarkConfig):
    """Configuration for live websocket benchmark runs."""

    num_operations: int = 1
    warmup_operations: int = 0
    combined_base_url: str = "wss://fstream.binance.com/stream?streams="
    symbols: list[str] = field(default_factory=lambda: ["btcusdt", "ethusdt", "solusdt"])
    stream_kinds: list[str] = field(default_factory=lambda: ["bookTicker", "trade"])
    connection_timeout_s: float = 10.0
    sample_window_s: float = 15.0
    pool_connections: int = 3
    pool_evict_interval_s: int = 60
    freshness_budget_ms: int = 15_000
    clock_future_drift_ms: int = 2_000


class _LiveBinanceHarness:
    """Shared helpers for live Binance websocket validation and collection."""

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
        return time.time_ns() // 1_000_000

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
        strict_values: bool,
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

        if strict_values:
            if price == 0 or qty == 0:
                assert price == 0
                assert qty == 0
                assert data.X == "NA"
            else:
                assert price > 0
                assert qty > 0
            return

        assert price >= 0
        assert qty >= 0

    @staticmethod
    async def _wait_for_state(
        get_state,
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
        strict_trade_values: bool,
    ) -> tuple[str, int, str]:
        """Decode payload and return stream name, event time, and event key."""
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
            return stream_name, event.E, f"{stream_name}|u:{event.u}"

        if stream_name.endswith("@trade"):
            event = cls.TRADE_DECODER.decode(inner_payload)
            cls._validate_trade(
                event,
                expected_symbols=expected_symbols,
                freshness_budget_ms=freshness_budget_ms,
                clock_future_drift_ms=clock_future_drift_ms,
                strict_values=strict_trade_values,
            )
            return stream_name, event.E, f"{stream_name}|t:{event.t}"

        raise AssertionError(f"Unexpected stream received: {stream_name}")

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
    ) -> tuple[list[int], list[tuple[str, int]], list[int], dict[str, int]]:
        """Collect latency samples, keyed latencies, throughput bins, and stream counts."""
        loop = asyncio.get_running_loop()
        start = await barrier.ready_and_wait()
        deadline = start + sample_window_s
        latencies_ms: list[int] = []
        keyed_latencies_ms: list[tuple[str, int]] = []
        counts: dict[str, int] = {}

        num_bins = max(1, int(math.ceil(sample_window_s)))
        bins = [0 for _ in range(num_bins)]

        while loop.time() < deadline:
            remaining = deadline - loop.time()
            payload = await cls._next_message(
                stream,
                timeout_s=max(0.1, min(2.0, remaining)),
            )
            stream_name, event_time_ms, event_key = cls._decode_and_validate_combined_payload(
                payload,
                expected_streams=expected_streams,
                expected_symbols=expected_symbols,
                freshness_budget_ms=freshness_budget_ms,
                clock_future_drift_ms=clock_future_drift_ms,
                strict_trade_values=False,
            )

            recv_ms = cls._now_ms()
            latency_ms = recv_ms - event_time_ms
            assert latency_ms >= -clock_future_drift_ms
            assert latency_ms <= freshness_budget_ms

            latencies_ms.append(latency_ms)
            keyed_latencies_ms.append((event_key, latency_ms))
            counts[stream_name] = counts.get(stream_name, 0) + 1

            elapsed_s = loop.time() - start
            if elapsed_s < sample_window_s:
                bin_idx = min(int(elapsed_s), num_bins - 1)
                bins[bin_idx] += 1

        return latencies_ms, keyed_latencies_ms, bins, counts


class LiveBinanceBenchmarkReporter(BenchmarkReporter):
    """Custom reporter with dedicated tables for throughput and latency."""

    THROUGHPUT_SCALE = 1_000_000

    def print_incoming_data_table(self, stats) -> None:
        """Print throughput-per-second percentile table."""
        print("Incoming Data Stats")
        print("-" * 100)
        print(
            f"{'Operation':<36} {'Count':>8} {'Mean msg/s':>13} "
            f"{'P50':>10} {'P90':>10} {'P95':>10} {'P99':>10}"
        )

        metrics = stats.get_operation("incoming_msgs_per_sec")
        if metrics is None or not metrics.latencies_ns:
            print(f"{'incoming_msgs_per_sec':<36} {0:>8} {0.0:>13.1f} {0.0:>10.1f} {0.0:>10.1f} {0.0:>10.1f} {0.0:>10.1f}")
            print("=" * 100)
            return

        pcts = metrics.compute_percentiles()

        def unscale(value: float) -> float:
            return value / self.THROUGHPUT_SCALE

        print(
            f"{'incoming_msgs_per_sec':<36} "
            f"{pcts['count']:>8} "
            f"{unscale(pcts['mean']):>13.1f} "
            f"{unscale(pcts['p50']):>10.1f} "
            f"{unscale(pcts['p90']):>10.1f} "
            f"{unscale(pcts['p95']):>10.1f} "
            f"{unscale(pcts['p99']):>10.1f}"
        )
        print("=" * 100)

    def print_latency_table(self, stats) -> None:
        """Print latency summary table in milliseconds for single/pool/delta."""
        print("Latency Stats (ms)")
        print("-" * 100)
        print(
            f"{'Path':<28} {'Count':>8} {'Mean ms':>11} {'P50':>10} "
            f"{'P90':>10} {'P95':>10} {'P99':>10} {'Min':>8} {'Max':>8}"
        )

        def print_row(label: str, op_name: str) -> None:
            metrics = stats.get_operation(op_name)
            if metrics is None or not metrics.latencies_ns:
                print(
                    f"{label:<28} {0:>8} {0.0:>11.2f} {0.0:>10.2f} "
                    f"{0.0:>10.2f} {0.0:>10.2f} {0.0:>10.2f} {0.0:>8.2f} {0.0:>8.2f}"
                )
                return

            pcts = metrics.compute_percentiles()
            min_ms = min(metrics.latencies_ns) / 1_000_000
            max_ms = max(metrics.latencies_ns) / 1_000_000

            print(
                f"{label:<28} "
                f"{pcts['count']:>8} "
                f"{(pcts['mean'] / 1_000_000):>11.2f} "
                f"{(pcts['p50'] / 1_000_000):>10.2f} "
                f"{(pcts['p90'] / 1_000_000):>10.2f} "
                f"{(pcts['p95'] / 1_000_000):>10.2f} "
                f"{(pcts['p99'] / 1_000_000):>10.2f} "
                f"{min_ms:>8.2f} "
                f"{max_ms:>8.2f}"
            )

        print_row("single", "single.event_latency")
        print_row("pool", "pool.event_latency")
        print_row("single_vs_pool", "single_vs_pool.delta_latency")
        print("=" * 100)


class LiveBinanceWebSocketBenchmark(
    _LiveBinanceHarness, BenchmarkRunner[LiveBinanceBenchmarkConfig]
):
    """Benchmark runner for live Binance websocket throughput and latency."""

    THROUGHPUT_SCALE = 1_000_000

    def __init__(self, config: LiveBinanceBenchmarkConfig) -> None:
        """Initialize benchmark runner."""
        super().__init__(config)
        self.url_streams, self.expected_streams = self._build_expected_streams(
            config.symbols,
            config.stream_kinds,
        )
        self.expected_symbols = {symbol.upper() for symbol in config.symbols}
        self.combined_url = f"{config.combined_base_url}{'/'.join(self.url_streams)}"

    def _create_subject(self) -> None:
        """No shared subject is created for this live-network benchmark."""
        return None

    def _run_benchmark_suite(self, _subject: None) -> None:
        """Run throughput and side-by-side latency benchmark."""
        print("Running incoming-data and latency benchmark...")
        self._benchmark_side_by_side_window()

    @staticmethod
    def _pair_latency_deltas(
        single_samples: list[tuple[str, int]],
        pool_samples: list[tuple[str, int]],
    ) -> list[int]:
        """Pair keyed event latencies and return single-pool delta list in ms."""
        by_key_single: dict[str, list[int]] = defaultdict(list)
        by_key_pool: dict[str, list[int]] = defaultdict(list)

        for key, latency_ms in single_samples:
            by_key_single[key].append(latency_ms)
        for key, latency_ms in pool_samples:
            by_key_pool[key].append(latency_ms)

        deltas_ms: list[int] = []
        for key in set(by_key_single) & set(by_key_pool):
            single_vals = by_key_single[key]
            pool_vals = by_key_pool[key]
            count = min(len(single_vals), len(pool_vals))
            for i in range(count):
                deltas_ms.append(single_vals[i] - pool_vals[i])

        return deltas_ms

    def _benchmark_side_by_side_window(self) -> None:
        """Benchmark incoming throughput and latency distributions in one window."""
        if self.stats is None:
            raise RuntimeError("Statistics not initialized")

        incoming_metrics = self.stats.add_operation("incoming_msgs_per_sec")
        single_metrics = self.stats.add_operation("single.event_latency")
        pool_metrics = self.stats.add_operation("pool.event_latency")
        delta_metrics = self.stats.add_operation("single_vs_pool.delta_latency")

        for _ in range(self.config.warmup_operations):
            asyncio.run(self._run_side_by_side_window_once())

        for _ in range(self.config.num_operations):
            result = asyncio.run(self._run_side_by_side_window_once())
            single_latencies_ms, pool_latencies_ms, deltas_ms, incoming_bins = result

            for value in incoming_bins:
                incoming_metrics.add_latency(int(round(value * self.THROUGHPUT_SCALE)))
            for value in single_latencies_ms:
                single_metrics.add_latency(int(round(value * 1_000_000)))
            for value in pool_latencies_ms:
                pool_metrics.add_latency(int(round(value * 1_000_000)))
            for value in deltas_ms:
                delta_metrics.add_latency(int(round(value * 1_000_000)))

    async def _run_side_by_side_window_once(
        self,
    ) -> tuple[list[int], list[int], list[int], list[int]]:
        """Run one synchronized side-by-side sampling window."""
        barrier = _StartBarrier(parties=2)
        ws = WsSingle(WsConnectionConfig.default(self.combined_url))
        pool = await WsPool.new(
            config=WsConnectionConfig.default(self.combined_url),
            on_message=self._noop_message_handler,
            pool_config=WsPoolConfig(
                num_connections=self.config.pool_connections,
                evict_interval_s=self.config.pool_evict_interval_s,
            ),
        )

        async with ws, pool:
            await self._wait_for_state(
                ws.get_state,
                ConnectionState.CONNECTED,
                timeout_s=self.config.connection_timeout_s,
            )
            await self._wait_for_state(
                pool.get_state,
                ConnectionState.CONNECTED,
                timeout_s=self.config.connection_timeout_s,
            )
            await self._wait_for_pool_connections(
                pool,
                timeout_s=self.config.connection_timeout_s,
            )

            single_result, pool_result = await asyncio.gather(
                self._collect_latency_samples(
                    ws,
                    expected_streams=self.expected_streams,
                    expected_symbols=self.expected_symbols,
                    sample_window_s=self.config.sample_window_s,
                    freshness_budget_ms=self.config.freshness_budget_ms,
                    clock_future_drift_ms=self.config.clock_future_drift_ms,
                    barrier=barrier,
                ),
                self._collect_latency_samples(
                    pool,
                    expected_streams=self.expected_streams,
                    expected_symbols=self.expected_symbols,
                    sample_window_s=self.config.sample_window_s,
                    freshness_budget_ms=self.config.freshness_budget_ms,
                    clock_future_drift_ms=self.config.clock_future_drift_ms,
                    barrier=barrier,
                ),
            )

        single_latencies_ms, single_keyed, single_bins, single_counts = single_result
        pool_latencies_ms, pool_keyed, pool_bins, pool_counts = pool_result

        missing_single = self.expected_streams - set(single_counts)
        missing_pool = self.expected_streams - set(pool_counts)
        assert not missing_single, f"Single missing streams: {sorted(missing_single)}"
        assert not missing_pool, f"Pool missing streams: {sorted(missing_pool)}"

        deltas_ms = self._pair_latency_deltas(single_keyed, pool_keyed)
        assert deltas_ms, "No overlapping events between single and pool for delta latency"

        merged_bins: list[int] = []
        for single_rate, pool_rate in zip(single_bins, pool_bins):
            merged_bins.append(int(round((single_rate + pool_rate) / 2.0)))

        return single_latencies_ms, pool_latencies_ms, deltas_ms, merged_bins


def _parse_csv(value: str) -> list[str]:
    """Parse a comma-separated CLI string into non-empty values."""
    parts = [part.strip() for part in value.split(",")]
    return [part for part in parts if part]


def main() -> None:
    """Main entry point."""
    cli = BenchmarkCLI("Benchmark live Binance Futures websocket performance")
    cli.parser.set_defaults(operations=1, warmup=0)
    cli.parser.add_argument(
        "--combined-base-url",
        default="wss://fstream.binance.com/stream?streams=",
        help=(
            "Combined stream base URL "
            "(default: wss://fstream.binance.com/stream?streams=)"
        ),
    )
    cli.parser.add_argument(
        "--symbols",
        default="btcusdt,ethusdt,solusdt",
        help="Comma-separated symbols (default: btcusdt,ethusdt,solusdt)",
    )
    cli.parser.add_argument(
        "--stream-kinds",
        default="bookTicker,trade",
        help="Comma-separated stream kinds (default: bookTicker,trade)",
    )
    cli.parser.add_argument(
        "--connection-timeout-s",
        type=float,
        default=10.0,
        help="Connection timeout in seconds (default: 10.0)",
    )
    cli.parser.add_argument(
        "--sample-window-s",
        type=float,
        default=15.0,
        help="Latency/throughput sample window in seconds (default: 15.0)",
    )
    cli.parser.add_argument(
        "--pool-connections",
        type=int,
        default=3,
        help="Number of pool websocket connections (default: 3)",
    )
    cli.parser.add_argument(
        "--pool-evict-interval-s",
        type=int,
        default=60,
        help="Pool eviction interval in seconds (default: 60)",
    )
    cli.parser.add_argument(
        "--freshness-budget-ms",
        type=int,
        default=15_000,
        help="Freshness budget in milliseconds (default: 15000)",
    )
    cli.parser.add_argument(
        "--clock-future-drift-ms",
        type=int,
        default=2_000,
        help="Allowed future clock drift in milliseconds (default: 2000)",
    )

    args = cli.parse()

    if args.multi_size:
        print("Multi-size benchmarking is not supported for this live benchmark")
        return

    symbols = [symbol.lower() for symbol in _parse_csv(args.symbols)]
    stream_kinds = _parse_csv(args.stream_kinds)
    if not symbols:
        raise ValueError("At least one symbol is required")
    if not stream_kinds:
        raise ValueError("At least one stream kind is required")

    config = LiveBinanceBenchmarkConfig(
        num_operations=max(1, int(args.operations)),
        warmup_operations=max(0, int(args.warmup)),
        combined_base_url=args.combined_base_url,
        symbols=symbols,
        stream_kinds=stream_kinds,
        connection_timeout_s=float(args.connection_timeout_s),
        sample_window_s=float(args.sample_window_s),
        pool_connections=max(1, int(args.pool_connections)),
        pool_evict_interval_s=max(1, int(args.pool_evict_interval_s)),
        freshness_budget_ms=max(1, int(args.freshness_budget_ms)),
        clock_future_drift_ms=max(0, int(args.clock_future_drift_ms)),
    )

    benchmark = LiveBinanceWebSocketBenchmark(config)
    stats = benchmark.run()

    reporter = LiveBinanceBenchmarkReporter(
        "Live Binance WebSocket Benchmark Results",
        {
            "Symbols": ",".join(config.symbols),
            "Streams": ",".join(config.stream_kinds),
            "Pool connections": config.pool_connections,
            "Sample window (s)": config.sample_window_s,
        },
    )
    reporter.print_header(stats, warmup=config.warmup_operations)
    reporter.print_incoming_data_table(stats)
    print()
    reporter.print_latency_table(stats)


if __name__ == "__main__":
    main()
