"""Main runner for the multi-venue streaming example.

Spawns one stream process per venue, consumes normalized IPC messages,
updates orderbooks, builds candles, and prints summary statistics.
"""

from __future__ import annotations

import argparse
import multiprocessing
import sys
import time
from dataclasses import dataclass

import msgspec

from examples.stream.core import (
    BBOUpdate,
    MsgType,
    OrderbookLevel as CoreOrderbookLevel,
    OrderbookMsg,
    StreamMessage,
    TradeMsg,
)
from examples.stream.core.stats import StatsCollector
from examples.stream.runner import stream_process_entry
from mm_toolbox.candles import Candle, TickCandles, Trade as CandleTrade, VolumeCandles
from mm_toolbox.logging.advanced import LogLevel, LoggerConfig, MasterLogger
from mm_toolbox.logging.advanced.handlers.base import BaseLogHandler
from mm_toolbox.logging.advanced.pylog import PyLog
from mm_toolbox.orderbook.standard import Orderbook, OrderbookLevel
from mm_toolbox.ringbuffer.ipc import IPCRingBufferConfig, IPCRingBufferConsumer
from mm_toolbox.time import time_iso8601


class StdoutLogHandler(BaseLogHandler):
    """Simple stdout handler for logging."""

    def format_log(self, log: PyLog) -> str:
        """Format a log message.

        Args:
            log: Log entry to format.

        Returns:
            str: Formatted log message.
        """
        name = log.name.decode()
        if name == "stream":
            return (
                f"{time_iso8601(float(log.timestamp_ns) / 1_000_000_000.0)} "
                f"[{log.level.name}] {log.message.decode()}"
            )
        return super().format_log(log)

    def push(self, logs: list[PyLog]) -> None:
        """Push logs to stdout.

        Args:
            logs: List of log entries.
        """
        try:
            for log in logs:
                formatted = self.format_log(log)
                print(formatted, flush=True)
        except Exception as exc:
            print(f"Failed to write logs to stdout; {exc}", file=sys.stderr)


@dataclass
class VenueConfig:
    """Configuration for a single venue.

    Attributes:
        symbol: Venue symbol.
        tick_size: Tick size for orderbook precision.
        lot_size: Lot size for orderbook precision.
    """

    symbol: str
    tick_size: float
    lot_size: float


@dataclass
class Config:
    """Configuration for the multi-venue runner.

    Attributes:
        venues: Venues to run.
        ticks_per_bucket: Tick candle threshold.
        volume_per_bucket: Volume candle threshold in quote currency.
        log_path: IPC path for logging.
        data_path: IPC path template for data (use {venue}).
    """

    venues: list[str]
    ticks_per_bucket: int
    volume_per_bucket: float
    log_path: str
    data_path: str


@dataclass
class OrderbookState:
    """Track per-venue orderbook state.

    Attributes:
        tick_size: Tick size for orderbook precision.
        lot_size: Lot size for orderbook precision.
        orderbook: Orderbook instance.
        size: Snapshot size used for the orderbook.
        snapshot_received: True once a snapshot is processed.
    """

    tick_size: float
    lot_size: float
    orderbook: Orderbook | None = None
    size: int | None = None
    snapshot_received: bool = False


class MultiVenueDataProcessor:
    """Processes streams from all venues."""

    def __init__(self, config: Config, venue_configs: dict[str, VenueConfig]) -> None:
        """Initialize the data processor.

        Args:
            config: Runner configuration.
            venue_configs: Per-venue settings.
        """
        self._config = config
        self._venue_configs = venue_configs
        self._decoder = msgspec.json.Decoder(type=StreamMessage)

        logger_config = LoggerConfig(
            base_level=LogLevel.INFO,
            path=config.log_path,
            flush_interval_s=0.1,
            emit_internal=False,
        )
        self.logger = MasterLogger(config=logger_config, log_handlers=[StdoutLogHandler()])

        self.consumers: dict[str, IPCRingBufferConsumer] = {}
        self.orderbooks: dict[str, OrderbookState] = {}

        for venue in config.venues:
            data_path = config.data_path.format(venue=venue)
            self.consumers[venue] = IPCRingBufferConsumer(
                IPCRingBufferConfig(
                    path=data_path,
                    backlog=10000,
                    num_producers=1,
                    num_consumers=1,
                    linger_ms=0,
                )
            )
            venue_config = venue_configs[venue]
            self.orderbooks[venue] = OrderbookState(
                tick_size=venue_config.tick_size,
                lot_size=venue_config.lot_size,
            )

        self.tick_candles = TickCandles(config.ticks_per_bucket, num_candles=100)
        self.volume_candles = VolumeCandles(config.volume_per_bucket, num_candles=100)
        self.stats = StatsCollector()
        self._last_vwap_print = 0.0

    def run(self) -> None:
        """Main processing loop."""
        self.logger.info("Multi-venue data processor started")
        try:
            while True:
                for venue, consumer in self.consumers.items():
                    messages = consumer.consume_all()
                    for msg_bytes in messages:
                        try:
                            stream_msg = self._decoder.decode(msg_bytes)
                        except Exception as exc:
                            self.logger.error(f"Failed to decode IPC message: {exc}")
                            continue
                        try:
                            self._process_message(venue, stream_msg)
                        except Exception as exc:
                            self.logger.error(f"Failed to process message: {exc}")
                            continue
                        end_time = time.time_ns() // 1_000_000
                        self.stats.get_or_create(venue).record_processing_time(
                            end_time - stream_msg.data.local_time_ms
                        )

                now = time.time()
                if now - self._last_vwap_print >= 1.0:
                    self._print_vwap()
                    self._last_vwap_print = now

                time.sleep(0.001)
        except KeyboardInterrupt:
            self.logger.info("Data processor interrupted, shutting down")
        finally:
            self.shutdown()

    def _process_message(self, venue: str, msg: StreamMessage) -> None:
        """Process a normalized message.

        Args:
            venue: Venue name.
            msg: Normalized stream message.
        """
        latency = msg.data.local_time_ms - msg.data.venue_time_ms
        venue_stats = self.stats.get_or_create(venue)
        venue_stats.record_latency(msg.msg_type, latency)
        venue_stats.record_message(msg.msg_type)

        match msg.msg_type:
            case MsgType.BBO:
                self._handle_bbo(venue, msg.data)
            case MsgType.TRADE:
                self._handle_trade(venue, msg.data)
            case MsgType.ORDERBOOK:
                self._handle_orderbook(venue, msg.data)

    def _handle_trade(self, venue: str, trade_msg: TradeMsg) -> None:
        """Handle trade updates.

        Args:
            venue: Venue name.
            trade_msg: Normalized trade message.
        """
        tick_count_before = len(self.tick_candles)
        vol_count_before = len(self.volume_candles)

        for trade in trade_msg.trades:
            candle_trade = CandleTrade(
                time_ms=trade.time_ms,
                is_buy=trade.is_buy,
                price=trade.price,
                size=trade.size,
            )
            self.tick_candles.process_trade(candle_trade)
            self.volume_candles.process_trade(candle_trade)

        if len(self.tick_candles) > tick_count_before and len(self.tick_candles) > 1:
            candle = self.tick_candles[-2]
            self._log_candle("Tick", candle)

        if len(self.volume_candles) > vol_count_before and len(self.volume_candles) > 1:
            candle = self.volume_candles[-2]
            self._log_candle("Volume", candle)

    def _handle_orderbook(self, venue: str, orderbook_msg: OrderbookMsg) -> None:
        """Handle orderbook updates.

        Args:
            venue: Venue name.
            orderbook_msg: Normalized orderbook update.
        """
        state = self.orderbooks[venue]
        match (orderbook_msg.is_bbo, orderbook_msg.is_snapshot):
            case (True, _):
                self._handle_orderbook_bbo(venue, orderbook_msg)
                return
            case (False, True):
                self._handle_orderbook_snapshot(venue, orderbook_msg)
                return
            case (False, False):
                if not state.snapshot_received or state.orderbook is None:
                    return
                bids = self._build_orderbook_levels(state, orderbook_msg.bids)
                asks = self._build_orderbook_levels(state, orderbook_msg.asks)
                state.orderbook.consume_deltas(asks=asks, bids=bids)

    def _handle_orderbook_snapshot(self, venue: str, snapshot: OrderbookMsg) -> None:
        """Handle orderbook snapshots.

        Args:
            venue: Venue name.
            snapshot: Normalized orderbook snapshot.
        """
        state = self.orderbooks[venue]
        size = min(len(snapshot.bids), len(snapshot.asks))
        if size <= 0:
            return
        if state.orderbook is None or state.size != size:
            state.orderbook = Orderbook(
                tick_size=state.tick_size,
                lot_size=state.lot_size,
                size=size,
            )
            state.size = size

        bids = self._build_orderbook_levels(state, snapshot.bids)
        asks = self._build_orderbook_levels(state, snapshot.asks)
        state.orderbook.consume_snapshot(asks=asks, bids=bids)
        state.snapshot_received = True
        self.logger.info(
            f"[{venue}] Orderbook snapshot received: {len(bids)} bids, {len(asks)} asks"
        )

    def _handle_orderbook_bbo(self, venue: str, orderbook_msg: OrderbookMsg) -> None:
        """Handle BBO updates embedded in orderbook messages.

        Args:
            venue: Venue name.
            orderbook_msg: Normalized orderbook message flagged as BBO.
        """
        state = self.orderbooks[venue]
        if not state.snapshot_received or state.orderbook is None:
            return
        if not orderbook_msg.bids or not orderbook_msg.asks:
            return

        best_bid_level = max(orderbook_msg.bids, key=lambda x: x.price)
        best_ask_level = min(orderbook_msg.asks, key=lambda x: x.price)

        best_bid = OrderbookLevel.from_values(
            price=best_bid_level.price,
            size=best_bid_level.size,
            norders=best_bid_level.num_orders,
            tick_size=state.tick_size,
            lot_size=state.lot_size,
        )
        best_ask = OrderbookLevel.from_values(
            price=best_ask_level.price,
            size=best_ask_level.size,
            norders=best_ask_level.num_orders,
            tick_size=state.tick_size,
            lot_size=state.lot_size,
        )
        state.orderbook.consume_bbo(ask=best_ask, bid=best_bid)

    def _handle_bbo(self, venue: str, bbo: BBOUpdate) -> None:
        """Handle BBO updates.

        Args:
            venue: Venue name.
            bbo: Normalized BBO update.
        """
        state = self.orderbooks[venue]
        if not state.snapshot_received or state.orderbook is None:
            return

        best_bid = OrderbookLevel.from_values(
            price=bbo.bid_price,
            size=bbo.bid_size,
            norders=1,
            tick_size=state.tick_size,
            lot_size=state.lot_size,
        )
        best_ask = OrderbookLevel.from_values(
            price=bbo.ask_price,
            size=bbo.ask_size,
            norders=1,
            tick_size=state.tick_size,
            lot_size=state.lot_size,
        )
        state.orderbook.consume_bbo(ask=best_ask, bid=best_bid)

    def _build_orderbook_levels(
        self, state: OrderbookState, levels: list[CoreOrderbookLevel]
    ) -> list[OrderbookLevel]:
        """Build orderbook levels from normalized orderbook levels.

        Args:
            state: Orderbook state with precision settings.
            levels: Normalized orderbook levels.

        Returns:
            list[OrderbookLevel]: Orderbook levels.
        """
        return [
            OrderbookLevel.from_values(
                price=level.price,
                size=level.size,
                norders=level.num_orders,
                tick_size=state.tick_size,
                lot_size=state.lot_size,
            )
            for level in levels
        ]

    def _log_candle(self, label: str, candle: Candle) -> None:
        """Log a completed candle.

        Args:
            label: Candle label.
            candle: Candle instance.
        """
        self.logger.info(
            f"{label} Candle: O={candle.open_price:.2f} "
            f"H={candle.high_price:.2f} "
            f"L={candle.low_price:.2f} "
            f"C={candle.close_price:.2f} "
            f"VWAP={candle.vwap_price:.2f}"
        )

    def _print_vwap(self) -> None:
        """Print $10K volume-weighted mid price for each venue."""
        for venue, state in self.orderbooks.items():
            if state.orderbook is None or not state.snapshot_received:
                continue
            vwap = state.orderbook.get_volume_weighted_mid_price(
                size=10_000.0, is_base_currency=False
            )
            self.logger.info(f"[{venue}] $10K VWAP: {vwap:.2f}")

    def shutdown(self) -> None:
        """Shutdown the data processor and print statistics."""
        for consumer in self.consumers.values():
            consumer.stop()
        self.stats.set_candle_counts(len(self.tick_candles), len(self.volume_candles))
        self.stats.print_summary()
        self.logger.shutdown()


def parse_args() -> Config:
    """Parse CLI arguments.

    Returns:
        Config: Parsed configuration.
    """
    parser = argparse.ArgumentParser(description="Multi-venue streaming runner")
    parser.add_argument("--venues", nargs="+", default=["binance", "bybit", "okx"])
    parser.add_argument("--ticks", type=int, default=100)
    parser.add_argument("--volume", type=float, default=10_000.0)
    parser.add_argument("--log-path", default="ipc:///tmp/stream_logger")
    parser.add_argument("--data-path", default="ipc:///tmp/stream_{venue}.ipc")
    args = parser.parse_args()

    return Config(
        venues=[venue.lower() for venue in args.venues],
        ticks_per_bucket=args.ticks,
        volume_per_bucket=args.volume,
        log_path=args.log_path,
        data_path=args.data_path,
    )


def get_default_venue_configs() -> dict[str, VenueConfig]:
    """Return default configuration for supported venues.

    Returns:
        dict[str, VenueConfig]: Venue config mapping.
    """
    return {
        "binance": VenueConfig(symbol="BTCUSDT", tick_size=0.01, lot_size=0.001),
        "bybit": VenueConfig(symbol="BTCUSDT", tick_size=0.01, lot_size=0.001),
        "okx": VenueConfig(symbol="BTC-USDT-SWAP", tick_size=0.01, lot_size=0.001),
        "hyperliquid": VenueConfig(symbol="BTC", tick_size=0.01, lot_size=0.001),
        "lighter": VenueConfig(symbol="BTC-USDC", tick_size=0.01, lot_size=0.001),
    }


def main() -> None:
    """Main entry point."""
    config = parse_args()
    venue_configs = get_default_venue_configs()

    for venue in config.venues:
        if venue not in venue_configs:
            raise ValueError(f"Unsupported venue: {venue}")

    processes: list[multiprocessing.Process] = []
    for venue in config.venues:
        venue_config = venue_configs[venue]
        data_path = config.data_path.format(venue=venue)
        proc = multiprocessing.Process(
            target=stream_process_entry,
            args=(venue, venue_config.symbol, data_path, config.log_path),
            daemon=True,
        )
        proc.start()
        processes.append(proc)
        print(f"[{venue}] Stream process started (PID: {proc.pid})", flush=True)

    processor = MultiVenueDataProcessor(config, venue_configs)
    try:
        processor.run()
    finally:
        for proc in processes:
            if proc.is_alive():
                proc.terminate()
            proc.join(timeout=2)


if __name__ == "__main__":
    main()
