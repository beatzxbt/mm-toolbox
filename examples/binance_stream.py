"""Comprehensive example demonstrating multi-process Binance streaming with SHM SPSC.

This example shows:
- Using WsPool for BBO, orderbook depth, and trade streams
- Building time-based candles on mid price
- Building standard orderbook from snapshot + deltas
- Computing TEMA on mid price with tick-size rounding
- SHM SPSC communication between processes
- Worker logger in stream process, master logger in processing process
"""

from __future__ import annotations

import asyncio
import multiprocessing
import os
import sys
import time
import urllib.request
from typing import Any

import msgspec

from mm_toolbox.candles import Trade, TimeCandles
from mm_toolbox.logging.advanced import (
    LogLevel,
    LoggerConfig,
    MasterLogger,
    WorkerLogger,
)
from mm_toolbox.logging.advanced.handlers.base import BaseLogHandler
from mm_toolbox.logging.advanced.pylog import PyLog
from mm_toolbox.moving_average import TimeExponentialMovingAverage
from mm_toolbox.orderbook.standard import Orderbook, OrderbookLevel
from mm_toolbox.ringbuffer.shm.spsc import (
    ShmSpscConsumer,
    ShmSpscProducer,
)
from mm_toolbox.rounding import Rounder, RounderConfig
from mm_toolbox.websocket import WsConnectionConfig, WsPool, WsPoolConfig


class StartupEvent:
    """Simple cross-process synchronization using a file."""

    def __init__(self, path: str):
        self.path = path

    def set(self):
        """Signal that startup is complete."""
        with open(self.path, "w") as f:
            f.write("ready")

    def wait(self, timeout: float = 10.0, poll_interval: float = 0.01):
        """Wait for startup signal."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                with open(self.path, "r") as f:
                    if f.read().strip() == "ready":
                        return True
            except FileNotFoundError:
                pass
            time.sleep(poll_interval)
        return False


class StdoutLogHandler(BaseLogHandler):
    """Simple stdout handler for logging."""

    def push(self, logs: list[PyLog]) -> None:
        """Push logs to stdout."""
        try:
            for log in logs:
                formatted = self.format_log(log)
                print(formatted, flush=True)
        except Exception as e:
            print(f"Failed to write logs to stdout; {e}", file=sys.stderr)


# Message types for SHM SPSC communication
class BBOUpdate(msgspec.Struct):
    """Best bid/offer update message."""

    event_time: int
    symbol: str
    best_bid_price: float
    best_bid_qty: float
    best_ask_price: float
    best_ask_qty: float


class OrderbookSnapshot(msgspec.Struct):
    """Orderbook snapshot message."""

    last_update_id: int
    bids: list[tuple[float, float]]
    asks: list[tuple[float, float]]


class OrderbookDelta(msgspec.Struct):
    """Orderbook delta update message."""

    final_update_id: int
    first_update_id: int
    bids: list[tuple[float, float]]
    asks: list[tuple[float, float]]


class TradeUpdate(msgspec.Struct):
    """Aggregated trade update message."""

    event_time: int
    symbol: str
    price: float
    quantity: float
    trade_time: int
    is_buyer_maker: bool


class StreamMessage(msgspec.Struct):
    """Wrapper for stream messages."""

    msg_type: str  # "bbo", "trade", "depth"
    data: dict[str, Any]


class BinanceStreamProcessor:
    """Handles WebSocket streams and forwards data via SHM SPSC."""

    def __init__(
        self,
        symbol: str,
        logger_path: str,
        data_path: str,
    ) -> None:
        """Initialize the stream processor."""
        self.symbol = symbol
        self.logger_path = logger_path
        self.data_path = data_path

        # Initialize worker logger
        logger_config = LoggerConfig(
            base_level=LogLevel.INFO,
            path=logger_path,
            flush_interval_s=0.5,
            emit_internal=False,
        )
        self.logger = WorkerLogger(config=logger_config, name="StreamProcess")

        # Initialize SHM SPSC producer for data
        self.data_producer = ShmSpscProducer(
            path=data_path,
            capacity_bytes=16 * 1024 * 1024,
            create=True,
            unlink_on_close=False,
        )

        # Message encoder
        self.encoder = msgspec.json.Encoder()

        # WebSocket pools
        self.bbo_pool: WsPool | None = None
        self.trade_pool: WsPool | None = None
        self.depth_pool: WsPool | None = None
        self._trade_sent_count = 0

    def _process_bbo_message(self, msg: bytes) -> None:
        """Process bookTicker (BBO) messages."""
        try:
            decoded = msgspec.json.decode(msg, type=dict)
            bbo = BBOUpdate(
                event_time=decoded["E"],
                symbol=decoded["s"],
                best_bid_price=float(decoded["b"]),
                best_bid_qty=float(decoded["B"]),
                best_ask_price=float(decoded["a"]),
                best_ask_qty=float(decoded["A"]),
            )
            stream_msg = StreamMessage(msg_type="bbo", data=msgspec.to_builtins(bbo))
            self.data_producer.insert(self.encoder.encode(stream_msg))
        except Exception as e:
            self.logger.error(f"Error processing BBO message: {e}".encode("utf-8"))

    def _process_trade_message(self, msg: bytes) -> None:
        """Process aggregated trade messages."""
        try:
            decoded = msgspec.json.decode(msg, type=dict)
            trade = TradeUpdate(
                event_time=decoded["E"],
                symbol=decoded["s"],
                price=float(decoded["p"]),
                quantity=float(decoded["q"]),
                trade_time=decoded["T"],
                is_buyer_maker=decoded["m"],
            )
            stream_msg = StreamMessage(
                msg_type="trade", data=msgspec.to_builtins(trade)
            )
            self.data_producer.insert(self.encoder.encode(stream_msg))
            self._trade_sent_count += 1
            if self._trade_sent_count % 50 == 0:
                self.logger.info(
                    f"Sent {self._trade_sent_count} trades".encode("utf-8")
                )
        except Exception as e:
            self.logger.error(f"Error processing trade message: {e}".encode("utf-8"))

    def _process_depth_message(self, msg: bytes) -> None:
        """Process depth update messages."""
        try:
            decoded = msgspec.json.decode(msg, type=dict)
            delta = OrderbookDelta(
                final_update_id=decoded["u"],
                first_update_id=decoded["U"],
                bids=[(float(p), float(q)) for p, q in decoded.get("b", [])],
                asks=[(float(p), float(q)) for p, q in decoded.get("a", [])],
            )
            stream_msg = StreamMessage(
                msg_type="depth", data=msgspec.to_builtins(delta)
            )
            self.data_producer.insert(self.encoder.encode(stream_msg))
        except Exception as e:
            self.logger.error(f"Error processing depth message: {e}".encode("utf-8"))

    async def _run_streams(self) -> None:
        """Run WebSocket streams."""
        self.logger.info(f"Starting streams for {self.symbol}".encode("utf-8"))

        # BBO stream
        bbo_config = WsConnectionConfig.default(
            wss_url=f"wss://fstream.binance.com/ws/{self.symbol.lower()}@bookTicker"
        )
        self.bbo_pool = await WsPool.new(
            config=bbo_config,
            pool_config=WsPoolConfig.default(),
        )

        # Trade stream
        trade_config = WsConnectionConfig.default(
            wss_url=f"wss://fstream.binance.com/ws/{self.symbol.lower()}@trade"
        )
        self.trade_pool = await WsPool.new(
            config=trade_config,
            pool_config=WsPoolConfig.default(),
        )

        # Depth stream
        depth_config = WsConnectionConfig.default(
            wss_url=f"wss://fstream.binance.com/ws/{self.symbol.lower()}@depth@100ms"
        )
        self.depth_pool = await WsPool.new(
            config=depth_config,
            pool_config=WsPoolConfig.default(),
        )

        async with self.bbo_pool, self.trade_pool, self.depth_pool:
            self.logger.info(
                "Streams connected, processing BBO + trades + depth...".encode("utf-8")
            )
            try:

                async def consume_bbo():
                    async for msg in self.bbo_pool:
                        self._process_bbo_message(msg)

                async def consume_trades():
                    async for msg in self.trade_pool:
                        self._process_trade_message(msg)

                async def consume_depth():
                    async for msg in self.depth_pool:
                        self._process_depth_message(msg)

                await asyncio.gather(consume_bbo(), consume_trades(), consume_depth())
            except KeyboardInterrupt:
                self.logger.info("Stream interrupted, shutting down...".encode("utf-8"))

    def run(self) -> None:
        """Run the stream processor."""
        try:
            asyncio.run(self._run_streams())
        except KeyboardInterrupt:
            self.logger.info("Stream process interrupted".encode("utf-8"))
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Shutdown the stream processor."""
        if self.bbo_pool is not None:
            self.bbo_pool.close()
        if self.trade_pool is not None:
            self.trade_pool.close()
        if self.depth_pool is not None:
            self.depth_pool.close()
        self.data_producer.close()
        self.logger.shutdown()


class BinanceDataProcessor:
    """Handles data processing and candle generation from BBO + trades."""

    def __init__(
        self,
        symbol: str,
        tick_size: float,
        lot_size: float,
        logger_path: str,
        data_path: str,
    ) -> None:
        """Initialize the data processor."""
        self.symbol = symbol
        self.tick_size = tick_size
        self.lot_size = lot_size

        # Initialize master logger with stdout handler
        logger_config = LoggerConfig(
            base_level=LogLevel.INFO,
            path=logger_path,
            flush_interval_s=0.1,
            emit_internal=False,
        )
        self.logger = MasterLogger(
            config=logger_config, log_handlers=[StdoutLogHandler()]
        )

        # Initialize SHM SPSC consumer for data
        self.data_consumer = ShmSpscConsumer(path=data_path)

        # Initialize time candles (1 second candles)
        self.time_candles = TimeCandles(secs_per_bucket=1.0, num_candles=100)

        # Message decoder
        self.decoder = msgspec.json.Decoder(type=StreamMessage)

        # Orderbook state
        self.orderbook: Orderbook | None = None
        self.last_update_id: int = 0
        self._depth_started: bool = False
        self._depth_update_count: int = 0

        # TEMA and rounding
        self.tema = TimeExponentialMovingAverage(half_life_s=1.0)
        self.rounder = Rounder(
            RounderConfig.default(tick_size=tick_size, lot_size=lot_size)
        )

        # State
        self._last_candle_timestamp = -1.0
        self._bbo_count = 0
        self._trade_count = 0

    def _fetch_snapshot(self) -> None:
        """Fetch initial orderbook snapshot via REST."""
        url = f"https://fapi.binance.com/fapi/v1/depth?symbol={self.symbol}&limit=100"
        with urllib.request.urlopen(url, timeout=20) as resp:
            data = resp.read()

        payload = msgspec.json.decode(data, type=dict)
        snapshot = OrderbookSnapshot(
            last_update_id=payload["lastUpdateId"],
            bids=[(float(p), float(q)) for p, q in payload.get("bids", [])],
            asks=[(float(p), float(q)) for p, q in payload.get("asks", [])],
        )

        bids = [
            OrderbookLevel(price=price, size=size, norders=0)
            for price, size in snapshot.bids
        ]
        asks = [
            OrderbookLevel(price=price, size=size, norders=0)
            for price, size in snapshot.asks
        ]

        self.orderbook = Orderbook(
            tick_size=self.tick_size,
            lot_size=self.lot_size,
            size=100,
        )
        self.orderbook.consume_snapshot(asks=asks, bids=bids)
        self.last_update_id = snapshot.last_update_id
        self._depth_started = False

        self.logger.info(
            f"Orderbook snapshot received: {len(bids)} bids, {len(asks)} asks".encode(
                "utf-8"
            )
        )

    def _handle_bbo(self, bbo_data: BBOUpdate) -> None:
        """Handle BBO update - log mid price periodically."""
        mid_price = (bbo_data.best_bid_price + bbo_data.best_ask_price) / 2.0

        # Feed TEMA
        tema_value = self.tema.update(mid_price)

        self._bbo_count += 1
        if self._bbo_count % 50 == 0:
            rounded_tema = self.rounder.bid(tema_value)
            self.logger.info(
                f"TEMA(1s): {rounded_tema:.2f} | Mid: {mid_price:.2f}".encode("utf-8")
            )

    def _handle_trade(self, trade_data: TradeUpdate) -> None:
        """Handle trade update - build candles with real size."""
        self._trade_count += 1
        if self._trade_count % 100 == 0:
            side = "SELL" if trade_data.is_buyer_maker else "BUY"
            self.logger.info(
                f"Trade #{self._trade_count}: {side} {trade_data.quantity:.4f} @ {trade_data.price:.2f}".encode(
                    "utf-8"
                )
            )

        # Create trade object with real size from exchange
        trade = Trade(
            time_ms=trade_data.trade_time,
            is_buy=not trade_data.is_buyer_maker,  # buyer is maker = seller is taker
            price=trade_data.price,
            size=trade_data.quantity,
        )

        # Process trade for candles
        self.time_candles.process_trade(trade)

        # Check if a new candle was completed
        if len(self.time_candles) > 0:
            current_candle = self.time_candles[-1]
            current_open_time = current_candle.open_time_ms
            if (
                current_open_time != self._last_candle_timestamp
                and self._last_candle_timestamp >= 0
            ):
                # New candle bucket started, previous one is complete
                if len(self.time_candles) > 1:
                    completed_candle = self.time_candles[-2]
                    if completed_candle.num_trades > 0:
                        self.logger.info(
                            f"1s Candle: O={completed_candle.open_price:.2f} "
                            f"H={completed_candle.high_price:.2f} "
                            f"L={completed_candle.low_price:.2f} "
                            f"C={completed_candle.close_price:.2f} "
                            f"VWAP={completed_candle.vwap:.2f} "
                            f"| Trades={completed_candle.num_trades} "
                            f"| Vol={completed_candle.buy_size + completed_candle.sell_size:.4f}".encode(
                                "utf-8"
                            )
                        )
            self._last_candle_timestamp = current_open_time

    def _handle_depth(self, depth_data: OrderbookDelta) -> None:
        """Handle orderbook delta update."""
        if self.orderbook is None:
            return

        # Validate sequence continuity
        if depth_data.final_update_id <= self.last_update_id:
            return

        if not self._depth_started:
            if (
                depth_data.first_update_id
                <= self.last_update_id + 1
                <= depth_data.final_update_id
            ):
                self._depth_started = True
            else:
                return
        else:
            if depth_data.final_update_id < self.last_update_id + 1:
                return
            if depth_data.first_update_id > self.last_update_id + 1:
                self.logger.warning(
                    f"Orderbook sequence gap: expected {self.last_update_id + 1}, "
                    f"got [{depth_data.first_update_id}, {depth_data.final_update_id}]".encode(
                        "utf-8"
                    )
                )
                return

        # Apply deltas
        bids = [
            OrderbookLevel(price=price, size=size, norders=0)
            for price, size in depth_data.bids
        ]
        asks = [
            OrderbookLevel(price=price, size=size, norders=0)
            for price, size in depth_data.asks
        ]
        self.orderbook.consume_deltas(asks=asks, bids=bids)
        self.last_update_id = depth_data.final_update_id

        self._depth_update_count += 1
        if self._depth_update_count % 100 == 0:
            depth = len(self.orderbook.get_bids()) + len(self.orderbook.get_asks())
            self.logger.info(f"Orderbook depth: {depth} levels".encode("utf-8"))

    def _process_message(self, msg_bytes: bytes) -> None:
        """Process a single message."""
        try:
            stream_msg = self.decoder.decode(msg_bytes)

            if stream_msg.msg_type == "bbo":
                bbo_data = msgspec.convert(stream_msg.data, type=BBOUpdate)
                self._handle_bbo(bbo_data)
            elif stream_msg.msg_type == "trade":
                trade_data = msgspec.convert(stream_msg.data, type=TradeUpdate)
                self._handle_trade(trade_data)
            elif stream_msg.msg_type == "depth":
                depth_data = msgspec.convert(stream_msg.data, type=OrderbookDelta)
                self._handle_depth(depth_data)

        except Exception as e:
            self.logger.error(f"Error processing message: {e}".encode("utf-8"))

    def run(self) -> None:
        """Run the data processor."""
        self.logger.info(
            f"Processing process started for {self.symbol}".encode("utf-8")
        )

        try:
            while True:
                try:
                    # Consume all available messages
                    messages = self.data_consumer.consume_all()
                    if not messages:
                        time.sleep(0.01)  # Small sleep to avoid busy waiting
                        continue

                    for msg_bytes in messages:
                        self._process_message(msg_bytes)

                except KeyboardInterrupt:
                    self.logger.info(
                        "Processing interrupted, shutting down...".encode("utf-8")
                    )
                    break
                except Exception as e:
                    self.logger.error(f"Error in processing loop: {e}".encode("utf-8"))
                    time.sleep(0.1)

        except KeyboardInterrupt:
            self.logger.info("Processing process interrupted".encode("utf-8"))
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Shutdown the data processor."""
        self.data_consumer.close()
        self.logger.shutdown()


def stream_process_entry(
    symbol: str,
    tick_size: float,
    lot_size: float,
    logger_path: str,
    data_path: str,
    startup_event_path: str,
) -> None:
    """Entry point for stream process."""
    # Wait for processing process to initialize MasterLogger and snapshot
    startup = StartupEvent(startup_event_path)
    if not startup.wait(timeout=10.0):
        print("Timeout waiting for processing process to start", file=sys.stderr)
        sys.exit(1)

    processor = BinanceStreamProcessor(symbol, logger_path, data_path)
    processor.run()


def processing_process_entry(
    symbol: str,
    tick_size: float,
    lot_size: float,
    logger_path: str,
    data_path: str,
    startup_event_path: str,
) -> None:
    """Entry point for processing process."""
    processor = BinanceDataProcessor(
        symbol, tick_size, lot_size, logger_path, data_path
    )

    # Fetch snapshot before signaling ready so stream process does not
    # produce deltas we are unprepared to handle.
    processor._fetch_snapshot()

    # Signal that MasterLogger and orderbook snapshot are ready
    startup = StartupEvent(startup_event_path)
    startup.set()

    processor.run()


def main() -> None:
    """Main entry point."""
    symbol = "BTCUSDT"
    tick_size = 0.01
    lot_size = 0.001

    # Paths - logger and data ringbuffer use raw filesystem paths
    logger_path = "/tmp/binance_logger"
    data_path = "/tmp/binance_data"
    startup_event_path = "/tmp/binance_startup_ready"

    # Clean up any stale files
    for p in (startup_event_path, data_path):
        try:
            os.remove(p)
        except FileNotFoundError:
            pass

    # Create processes
    stream_proc = multiprocessing.Process(
        target=stream_process_entry,
        args=(symbol, tick_size, lot_size, logger_path, data_path, startup_event_path),
        daemon=True,
    )
    processing_proc = multiprocessing.Process(
        target=processing_process_entry,
        args=(symbol, tick_size, lot_size, logger_path, data_path, startup_event_path),
        daemon=True,
    )

    # Start processing first so MasterLogger creates shared memory before WorkerLogger attaches
    processing_proc.start()
    stream_proc.start()

    print(f"Started stream and processing processes for {symbol}")
    print("Press Ctrl+C to stop...")

    try:
        # Wait for processes
        stream_proc.join()
        processing_proc.join()
    except KeyboardInterrupt:
        print("\nShutting down...")
        stream_proc.terminate()
        processing_proc.terminate()
        stream_proc.join(timeout=2)
        processing_proc.join(timeout=2)


if __name__ == "__main__":
    main()
