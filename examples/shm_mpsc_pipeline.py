"""Multi-process Binance streaming with SHM MPSC and per-symbol AdvancedOrderbook.

This example demonstrates:
- Spawning N worker processes, each connecting to a different Binance symbol
  via WsSingle and pushing market updates through a shared ShmMpscProducer.
- A single master process consuming from ShmMpscConsumer and maintaining
  per-symbol AdvancedOrderbook instances.
- Fetching tick/lot sizes from the Binance REST API with RateLimiter.
- Logging trades against the current BBO using WorkerLogger (workers) and
  MasterLogger (master).
- msgspec encoding/decoding for IPC.
- Clean shutdown on Ctrl+C.

Snapshots seed the demonstration books; only the latest BBO is maintained live.
This example does not subscribe to depth deltas or maintain synchronized depth.
It stops the whole pipeline on transport loss or disconnected continuity.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import multiprocessing
import os
import signal
import sys
import tempfile
import time
import urllib.request
from typing import Any

import msgspec

from mm_toolbox.logging.advanced import (
    LogLevel,
    LoggerConfig,
    MasterLogger,
    WorkerLogger,
)
from mm_toolbox.logging.advanced.handlers.base import BaseLogHandler
from mm_toolbox.logging.advanced.pylog import PyLog
from mm_toolbox.orderbook.advanced import (
    AdvancedOrderbook,
    OrderbookLevel,
    OrderbookLevels,
)
from mm_toolbox.rate_limiter import RateLimiter
from mm_toolbox.ringbuffer.shm import ShmMpscConsumer, ShmMpscProducer
from mm_toolbox.websocket import WsConnectionConfig, WsSingle

if __package__:
    from ._stream_guard import (
        FeedIntegrityError,
        assert_no_receive_loss,
        assert_no_transport_loss,
        consume_batch,
        forward_message,
        run_feeds,
    )
else:
    from _stream_guard import (
        FeedIntegrityError,
        assert_no_receive_loss,
        assert_no_transport_loss,
        consume_batch,
        forward_message,
        run_feeds,
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "HYPEUSDT"]
EXCHANGE_INFO_URL = "https://fapi.binance.com/fapi/v1/exchangeInfo"
SHM_CAPACITY_BYTES = 64 * 1024 * 1024


# ---------------------------------------------------------------------------
# Cross-process synchronization
# ---------------------------------------------------------------------------


class StartupEvent:
    """Simple cross-process synchronization using a file."""

    def __init__(self, path: str) -> None:
        """Initialize the startup event.

        Args:
            path: Filesystem path used as the synchronization signal.
        """
        self.path = path

    def set(self) -> None:
        """Signal that startup is complete."""
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("ready")

    def wait(self, timeout: float = 30.0, poll_interval: float = 0.01) -> bool:
        """Wait for the startup signal.

        Args:
            timeout: Maximum seconds to wait.
            poll_interval: Seconds between polls.

        Returns:
            True if the signal was received, False on timeout.
        """
        start = time.time()
        while time.time() - start < timeout:
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    if f.read().strip() == "ready":
                        return True
            except FileNotFoundError:
                pass
            time.sleep(poll_interval)
        return False


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------


class StdoutLogHandler(BaseLogHandler):
    """Simple stdout handler for logging."""

    def push(self, logs: list[PyLog]) -> None:
        """Push logs to stdout.

        Args:
            logs: Batch of log records to emit.
        """
        try:
            for log in logs:
                formatted = self.format_log(log)
                print(formatted, flush=True)
        except Exception as e:
            print(f"Failed to write logs to stdout; {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# IPC message struct
# ---------------------------------------------------------------------------


class MarketUpdate(msgspec.Struct):
    """Unified market update message for SHM MPSC communication."""

    msg_type: str  # "trade" or "bbo"
    event_time: int
    symbol: str
    price: float = 0.0
    quantity: float = 0.0
    trade_time: int = 0
    is_buyer_maker: bool = False
    best_bid_price: float = 0.0
    best_bid_qty: float = 0.0
    best_ask_price: float = 0.0
    best_ask_qty: float = 0.0


# ---------------------------------------------------------------------------
# REST helpers
# ---------------------------------------------------------------------------


def _acquire_rate_limit_token(limiter: RateLimiter) -> None:
    """Block until a rate-limit token can be consumed.

    Args:
        limiter: The RateLimiter instance to consume from.
    """
    while True:
        result = limiter.try_consume()
        if result.allowed:
            break
        time.sleep(0.05)


def fetch_symbol_info(symbols: list[str]) -> dict[str, tuple[float, float]]:
    """Fetch tick and lot sizes from the Binance futures exchange info endpoint.

    Uses a RateLimiter configured for 1200 weight per minute.

    Args:
        symbols: List of symbol strings to look up.

    Returns:
        Mapping from symbol to (tick_size, lot_size).
    """
    limiter = RateLimiter.per_minute(capacity=1200)
    _acquire_rate_limit_token(limiter)

    with urllib.request.urlopen(EXCHANGE_INFO_URL, timeout=20) as resp:
        data = resp.read()

    payload = msgspec.json.decode(data, type=dict)
    symbol_info: dict[str, tuple[float, float]] = {}

    for sym_data in payload.get("symbols", []):
        symbol = sym_data.get("symbol")
        if symbol not in symbols:
            continue

        tick_size: float | None = None
        lot_size: float | None = None

        for filt in sym_data.get("filters", []):
            if filt.get("filterType") == "PRICE_FILTER":
                tick_size = float(filt.get("tickSize", 0))
            elif filt.get("filterType") == "LOT_SIZE":
                lot_size = float(filt.get("stepSize", 0))

        if tick_size is not None and lot_size is not None:
            symbol_info[symbol] = (tick_size, lot_size)

    return symbol_info


def fetch_orderbook_snapshot(symbol: str, limiter: RateLimiter) -> dict[str, Any]:
    """Fetch a Binance futures orderbook snapshot for a single symbol.

    Args:
        symbol: The trading pair symbol (e.g. BTCUSDT).
        limiter: RateLimiter to throttle the REST call.

    Returns:
        Parsed JSON response as a dict.
    """
    _acquire_rate_limit_token(limiter)
    url = f"https://fapi.binance.com/fapi/v1/depth?symbol={symbol}&limit=100"
    with urllib.request.urlopen(url, timeout=20) as resp:
        data = resp.read()
    return msgspec.json.decode(data, type=dict)


# ---------------------------------------------------------------------------
# Worker process
# ---------------------------------------------------------------------------


class WorkerProcess:
    """Handles WebSocket streams for one symbol and pushes updates via SHM MPSC."""

    def __init__(self, symbol: str, logger_path: str, data_path: str) -> None:
        """Initialize the worker process.

        Args:
            symbol: Trading pair to stream (e.g. BTCUSDT).
            logger_path: Filesystem path for the logger SHM.
            data_path: Filesystem path for the MPSC SHM ringbuffer.
        """
        self.symbol = symbol
        self.logger_path = logger_path
        self.data_path = data_path

        logger_config = LoggerConfig(
            base_level=LogLevel.INFO,
            path=logger_path,
            flush_interval_s=0.5,
            emit_internal=False,
        )
        self.logger = WorkerLogger(config=logger_config, name=f"Worker-{symbol}")

        try:
            self.data_producer = ShmMpscProducer(
                path=data_path,
                capacity_bytes=SHM_CAPACITY_BYTES,
                create=False,
                unlink_on_close=False,
            )
        except BaseException:
            self.logger.shutdown()
            raise

        self.encoder = msgspec.json.Encoder()
        self.ws_trade: WsSingle | None = None
        self.ws_bbo: WsSingle | None = None
        self._shutdown = False

    def _process_trade_message(self, msg: bytes) -> None:
        """Decode a Binance @trade message and push it to the MPSC ring."""
        decoded = msgspec.json.decode(msg, type=dict)
        update = MarketUpdate(
            msg_type="trade",
            event_time=decoded["E"],
            symbol=decoded["s"],
            price=float(decoded["p"]),
            quantity=float(decoded["q"]),
            trade_time=decoded["T"],
            is_buyer_maker=decoded["m"],
        )
        forward_message(self.data_producer, self.encoder.encode(update), self.symbol)

    def _process_bbo_message(self, msg: bytes) -> None:
        """Decode a Binance @bookTicker message and push it to the MPSC ring."""
        decoded = msgspec.json.decode(msg, type=dict)
        update = MarketUpdate(
            msg_type="bbo",
            event_time=decoded["E"],
            symbol=decoded["s"],
            best_bid_price=float(decoded["b"]),
            best_bid_qty=float(decoded["B"]),
            best_ask_price=float(decoded["a"]),
            best_ask_qty=float(decoded["A"]),
        )
        forward_message(self.data_producer, self.encoder.encode(update), self.symbol)

    async def _run_streams(self) -> None:
        """Start WebSocket streams for trades and BBO."""
        self.logger.info(f"Starting streams for {self.symbol}".encode("utf-8"))

        trade_config = WsConnectionConfig.default(
            wss_url=f"wss://fstream.binance.com/ws/{self.symbol.lower()}@trade",
            auto_reconnect=False,
        )
        bbo_config = WsConnectionConfig.default(
            wss_url=f"wss://fstream.binance.com/ws/{self.symbol.lower()}@bookTicker",
            auto_reconnect=False,
        )

        self.ws_trade = WsSingle(config=trade_config)
        self.ws_bbo = WsSingle(config=bbo_config)
        try:
            async with self.ws_trade, self.ws_bbo:
                await run_feeds(
                    [
                        (
                            self.ws_trade,
                            f"{self.symbol} trades",
                            self._process_trade_message,
                        ),
                        (self.ws_bbo, f"{self.symbol} BBO", self._process_bbo_message),
                    ]
                )
        finally:
            assert_no_receive_loss(self.ws_trade, f"{self.symbol} trades")
            assert_no_receive_loss(self.ws_bbo, f"{self.symbol} BBO")

    def run(self) -> None:
        """Run the worker event loop."""
        try:
            asyncio.run(self._run_streams())
        except KeyboardInterrupt:
            self.logger.info("Worker interrupted".encode("utf-8"))
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Close WebSockets, ringbuffer, and logger."""
        if self._shutdown:
            return
        self._shutdown = True
        if self.ws_trade is not None:
            self.ws_trade.close()
        if self.ws_bbo is not None:
            self.ws_bbo.close()
        try:
            assert_no_transport_loss(self.data_producer, self.symbol)
        finally:
            self.data_producer.close()
            self.logger.shutdown()


# ---------------------------------------------------------------------------
# Master process
# ---------------------------------------------------------------------------


class MasterProcess:
    """Consumes MPSC messages, maintains orderbooks, and logs against BBO."""

    def __init__(
        self,
        symbols: list[str],
        symbol_info: dict[str, tuple[float, float]],
        logger_path: str,
        data_path: str,
    ) -> None:
        """Initialize the master process.

        Args:
            symbols: List of symbols being streamed.
            symbol_info: Mapping from symbol to (tick_size, lot_size).
            logger_path: Filesystem path for the logger SHM.
            data_path: Filesystem path for the MPSC SHM ringbuffer.
        """
        self.symbols = symbols
        self.symbol_info = symbol_info
        self.logger_path = logger_path
        self.data_path = data_path

        logger_config = LoggerConfig(
            base_level=LogLevel.INFO,
            path=logger_path,
            flush_interval_s=0.1,
            emit_internal=False,
        )
        self.logger = MasterLogger(
            config=logger_config, log_handlers=[StdoutLogHandler()]
        )

        # Create the MPSC backing file so workers can attach.
        try:
            self._data_producer = ShmMpscProducer(
                path=data_path,
                capacity_bytes=SHM_CAPACITY_BYTES,
                # Reserve the master's creator shard as well as worker shards.
                num_rings=len(symbols) + 1,
                create=True,
                unlink_on_close=True,
            )
            try:
                self.data_consumer = ShmMpscConsumer(path=data_path)
            except BaseException:
                self._data_producer.close()
                raise
        except BaseException:
            self.logger.shutdown()
            raise

        self.decoder = msgspec.json.Decoder(type=MarketUpdate)

        self.orderbooks: dict[str, AdvancedOrderbook] = {}
        for symbol in symbols:
            tick_size, lot_size = symbol_info[symbol]
            self.orderbooks[symbol] = AdvancedOrderbook(
                tick_size=tick_size,
                lot_size=lot_size,
                num_levels=100,
            )

        self._trade_counts: dict[str, int] = {s: 0 for s in symbols}
        self._bbo_ready: set[str] = set()
        self._shutdown = False

    def _load_snapshots(self) -> None:
        """Fetch and apply initial orderbook snapshots via REST."""
        limiter = RateLimiter.per_minute(capacity=1200)
        for symbol in self.symbols:
            payload = fetch_orderbook_snapshot(symbol, limiter)
            bids_data = payload["bids"]
            asks_data = payload["asks"]
            bids = OrderbookLevels.from_list(
                prices=[float(p) for p, _ in bids_data],
                sizes=[float(q) for _, q in bids_data],
            )
            asks = OrderbookLevels.from_list(
                prices=[float(p) for p, _ in asks_data],
                sizes=[float(q) for _, q in asks_data],
            )
            self.orderbooks[symbol].consume_snapshot(asks=asks, bids=bids)
            self.logger.info(
                f"[{symbol}] Initial snapshot (BBO updates only): {len(bids_data)} bids, "
                f"{len(asks_data)} asks".encode("utf-8")
            )

    def _handle_bbo(self, update: MarketUpdate) -> None:
        """Apply a BBO update to the relevant orderbook."""
        ob = self.orderbooks.get(update.symbol)
        if ob is None:
            raise FeedIntegrityError(f"Unknown symbol: {update.symbol}")
        bid = OrderbookLevel(
            price=update.best_bid_price, size=update.best_bid_qty, norders=1
        )
        ask = OrderbookLevel(
            price=update.best_ask_price, size=update.best_ask_qty, norders=1
        )
        ob.consume_bbo(ask=ask, bid=bid)
        self._bbo_ready.add(update.symbol)

    def _handle_trade(self, update: MarketUpdate) -> None:
        """Log a trade against the current BBO."""
        symbol = update.symbol
        if not (
            math.isfinite(update.price)
            and update.price > 0
            and math.isfinite(update.quantity)
            and update.quantity > 0
        ):
            self.logger.info(
                b"Skipping trade with nonpositive or nonfinite price/quantity"
            )
            return
        self._trade_counts[symbol] += 1

        ob = self.orderbooks.get(symbol)
        if ob is None or symbol not in self._bbo_ready:
            return

        count = self._trade_counts[symbol]
        side = "SELL" if update.is_buyer_maker else "BUY"

        if count % 100 == 0:
            try:
                best_bid, best_ask = ob.get_bbo()
                self.logger.info(
                    f"[{symbol}] Trade #{count}: {side} "
                    f"{update.quantity:.4f} @ {update.price:.2f} | "
                    f"BBO: {best_bid.price:.2f}/{best_ask.price:.2f}".encode("utf-8")
                )
            except RuntimeError:
                self.logger.info(
                    f"[{symbol}] Trade #{count}: {side} "
                    f"{update.quantity:.4f} @ {update.price:.2f} | "
                    f"BBO: N/A".encode("utf-8")
                )

    def _process_message(self, msg_bytes: bytes) -> None:
        """Decode and dispatch a single MPSC message."""
        update = self.decoder.decode(msg_bytes)
        if update.msg_type == "bbo":
            self._handle_bbo(update)
        elif update.msg_type == "trade":
            self._handle_trade(update)
        else:
            raise FeedIntegrityError(f"Unknown market message: {update.msg_type}")

    def run(self) -> None:
        """Run the master consumption loop."""
        self.logger.info("Master process started".encode("utf-8"))
        try:
            while True:
                messages = consume_batch(self.data_consumer, "market data")
                if not messages:
                    time.sleep(0.001)
                    continue

                for msg_bytes in messages:
                    assert_no_transport_loss(self.data_consumer, "market data")
                    self._process_message(msg_bytes)

        except KeyboardInterrupt:
            self.logger.info("Master interrupted, shutting down...".encode("utf-8"))
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Close consumer, producer, and logger."""
        if self._shutdown:
            return
        self._shutdown = True
        try:
            assert_no_transport_loss(self.data_consumer, "market data")
        finally:
            self.data_consumer.close()
            self._data_producer.close()
            self.logger.shutdown()


# ---------------------------------------------------------------------------
# Process entry points
# ---------------------------------------------------------------------------


def worker_entry(
    symbol: str,
    logger_path: str,
    data_path: str,
    startup_event_path: str,
) -> None:
    """Entry point for a worker process.

    Args:
        symbol: Trading pair to stream.
        logger_path: Path for logger SHM.
        data_path: Path for MPSC SHM.
        startup_event_path: Path for the startup synchronization file.
    """
    startup = StartupEvent(startup_event_path)
    if not startup.wait(timeout=30.0):
        print(f"Worker {symbol}: timeout waiting for master", file=sys.stderr)
        sys.exit(1)

    worker = WorkerProcess(symbol, logger_path, data_path)

    def _signal_handler(signum: int, frame: Any) -> None:
        worker.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    worker.run()


def master_entry(
    symbols: list[str],
    symbol_info: dict[str, tuple[float, float]],
    logger_path: str,
    data_path: str,
    startup_event_path: str,
) -> None:
    """Entry point for the master process.

    Args:
        symbols: List of trading pairs being streamed.
        symbol_info: Mapping from symbol to (tick_size, lot_size).
        logger_path: Path for logger SHM.
        data_path: Path for MPSC SHM.
        startup_event_path: Path for the startup synchronization file.
    """
    master = MasterProcess(symbols, symbol_info, logger_path, data_path)

    def _signal_handler(signum: int, frame: Any) -> None:
        master.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    try:
        # Publish readiness only after every requested snapshot succeeds.
        master._load_snapshots()
        startup = StartupEvent(startup_event_path)
        startup.set()
        master.run()
    finally:
        master.shutdown()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _run_pipeline(duration_s: float | None, runtime_dir: str) -> None:
    """Run workers and master with shared files owned by runtime_dir."""
    logger_path = os.path.join(runtime_dir, "binance_mpsc_logger")
    data_path = os.path.join(runtime_dir, "binance_mpsc_data")
    startup_event_path = os.path.join(runtime_dir, "binance_mpsc_startup")

    print("Fetching symbol info from Binance...")
    symbol_info = fetch_symbol_info(SYMBOLS)
    for symbol in SYMBOLS:
        if symbol not in symbol_info:
            raise RuntimeError(f"Failed to retrieve info for {symbol}")
        tick, lot = symbol_info[symbol]
        print(f"  {symbol}: tick={tick}, lot={lot}")

    # Create and start master first so it creates the SHM backing file.
    master_proc = multiprocessing.Process(
        target=master_entry,
        args=(SYMBOLS, symbol_info, logger_path, data_path, startup_event_path),
        daemon=True,
    )
    master_proc.start()

    # Wait for master to initialize before spawning workers.
    startup = StartupEvent(startup_event_path)
    workers: list[multiprocessing.Process] = []
    startup_deadline = time.monotonic() + 30.0
    unexpected_exit = False
    try:
        ready = False
        while time.monotonic() < startup_deadline:
            if startup.wait(timeout=0.05):
                ready = True
                break
            if not master_proc.is_alive():
                break
        if not ready:
            raise TimeoutError("Timed out waiting for the master to start")

        # Spawn workers only after the master has created logger and data SHM.
        for symbol in SYMBOLS:
            proc = multiprocessing.Process(
                target=worker_entry,
                args=(symbol, logger_path, data_path, startup_event_path),
                daemon=True,
            )
            proc.start()
            workers.append(proc)
            print(f"Started worker for {symbol}")

        print("All processes running. Press Ctrl+C to stop...")
        start = time.monotonic()
        while master_proc.is_alive():
            if any(proc.exitcode is not None for proc in workers):
                unexpected_exit = True
                print("A worker exited unexpectedly; stopping the example.")
                break
            if duration_s is not None and time.monotonic() - start >= duration_s:
                print(f"Duration limit reached ({duration_s:g}s).")
                break
            time.sleep(0.1)
        else:
            unexpected_exit = True
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        # Stop the WebSocket workers first so the master can drain final logs.
        for w in workers:
            if w.is_alive() and w.pid is not None:
                try:
                    os.kill(w.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for w in workers:
            w.join(timeout=5)
        if master_proc.is_alive() and master_proc.pid is not None:
            try:
                os.kill(master_proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        master_proc.join(timeout=5)

        for proc in [*workers, master_proc]:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2)

    failed = [
        proc.name for proc in [master_proc, *workers] if proc.exitcode not in (0, None)
    ]
    if failed:
        raise RuntimeError(f"Example child process failed: {', '.join(failed)}")
    if unexpected_exit:
        raise RuntimeError("Example child process exited before shutdown was requested")


def main(duration_s: float | None = None) -> None:
    """Run the pipeline in a temporary directory owned by this invocation."""
    with tempfile.TemporaryDirectory(prefix="mm-binance-mpsc-") as runtime_dir:
        _run_pipeline(duration_s, runtime_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stream public Binance futures market data with SHM MPSC"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Stop after this many seconds (default: run until interrupted)",
    )
    args = parser.parse_args()
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration must be > 0")
    main(duration_s=args.duration)
