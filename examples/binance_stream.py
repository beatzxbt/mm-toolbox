"""Comprehensive example demonstrating multi-process Binance streaming with IPC.

This example shows:
- Using WsPool for BBO and orderbook streams
- Building time-based candles on mid price
- Building standard orderbook
- IPC communication between processes
- Worker logger in stream process, master logger in processing process
"""

import asyncio
import multiprocessing
import sys
import time
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
from mm_toolbox.orderbook.standard import Orderbook, OrderbookLevel
from mm_toolbox.ringbuffer.ipc import (
    IPCRingBufferConfig,
    IPCRingBufferConsumer,
    IPCRingBufferProducer,
)
from mm_toolbox.websocket import WsConnectionConfig, WsPool, WsPoolConfig


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


# Message types for IPC communication
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

    first_update_id: int
    final_update_id: int
    bids: list[tuple[float, float]]
    asks: list[tuple[float, float]]


class StreamMessage(msgspec.Struct):
    """Wrapper for stream messages."""

    msg_type: str  # "bbo", "snapshot", "delta"
    data: dict[str, Any]


class BinanceStreamProcessor:
    """Handles WebSocket streams and forwards data via IPC."""

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

        # Initialize IPC producer for data
        self.data_producer = IPCRingBufferProducer(
            IPCRingBufferConfig(
                path=data_path,
                backlog=10000,
                num_producers=1,
                num_consumers=1,
                linger_ms=0,
            )
        )

        # Message encoder
        self.encoder = msgspec.json.Encoder()

        # WebSocket pools
        self.bbo_pool: WsPool | None = None
        self.orderbook_pool: WsPool | None = None

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
            self.data_producer.insert(self.encoder.encode(stream_msg), copy=False)
        except Exception as e:
            self.logger.error(f"Error processing BBO message: {e}")

    def _process_orderbook_message(self, msg: bytes) -> None:
        """Process orderbook depth messages."""
        try:
            decoded = msgspec.json.decode(msg, type=dict)
            if "lastUpdateId" in decoded:
                # Snapshot
                snapshot = OrderbookSnapshot(
                    last_update_id=decoded["lastUpdateId"],
                    bids=[[float(p), float(q)] for p, q in decoded["bids"]],
                    asks=[[float(p), float(q)] for p, q in decoded["asks"]],
                )
                stream_msg = StreamMessage(
                    msg_type="snapshot", data=msgspec.to_builtins(snapshot)
                )
            else:
                # Delta update
                delta = OrderbookDelta(
                    first_update_id=decoded["u"],
                    final_update_id=decoded["U"],
                    bids=[[float(p), float(q)] for p, q in decoded["b"]],
                    asks=[[float(p), float(q)] for p, q in decoded["a"]],
                )
                stream_msg = StreamMessage(
                    msg_type="delta", data=msgspec.to_builtins(delta)
                )
            self.data_producer.insert(self.encoder.encode(stream_msg), copy=False)
        except Exception as e:
            self.logger.error(f"Error processing orderbook message: {e}")

    async def _run_streams(self) -> None:
        """Run WebSocket streams."""
        self.logger.info(f"Starting streams for {self.symbol}")

        # BBO stream
        bbo_config = WsConnectionConfig.default(
            wss_url=f"wss://fstream.binance.com/ws/{self.symbol.lower()}@bookTicker"
        )
        self.bbo_pool = await WsPool.new(
            config=bbo_config,
            on_message=self._process_bbo_message,
            pool_config=WsPoolConfig.default(),
        )

        # Orderbook stream
        orderbook_config = WsConnectionConfig.default(
            wss_url=f"wss://fstream.binance.com/ws/{self.symbol.lower()}@depth@100ms"
        )
        self.orderbook_pool = await WsPool.new(
            config=orderbook_config,
            on_message=self._process_orderbook_message,
            pool_config=WsPoolConfig.default(),
        )

        async with self.bbo_pool, self.orderbook_pool:
            self.logger.info("Streams connected, processing messages...")
            try:

                async def consume_bbo():
                    async for _ in self.bbo_pool:
                        pass

                async def consume_orderbook():
                    async for _ in self.orderbook_pool:
                        pass

                await asyncio.gather(consume_bbo(), consume_orderbook())
            except KeyboardInterrupt:
                self.logger.info("Stream interrupted, shutting down...")

    def run(self) -> None:
        """Run the stream processor."""
        try:
            asyncio.run(self._run_streams())
        except KeyboardInterrupt:
            self.logger.info("Stream process interrupted")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Shutdown the stream processor."""
        if self.bbo_pool is not None:
            self.bbo_pool.close()
        if self.orderbook_pool is not None:
            self.orderbook_pool.close()
        self.data_producer.stop()
        self.logger.shutdown()


class BinanceDataProcessor:
    """Handles data processing, orderbook building, and candle generation."""

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

        # Initialize IPC consumer for data
        self.data_consumer = IPCRingBufferConsumer(
            IPCRingBufferConfig(
                path=data_path,
                backlog=10000,
                num_producers=1,
                num_consumers=1,
                linger_ms=0,
            )
        )

        # Initialize orderbook
        self.orderbook = Orderbook(
            tick_size=tick_size,
            lot_size=lot_size,
            size=500,
        )

        # Initialize time candles (1 second candles)
        self.time_candles = TimeCandles(secs_per_bucket=1.0, num_candles=100)

        # Message decoder
        self.decoder = msgspec.json.Decoder(type=StreamMessage)

        # State
        self.snapshot_received = False

    def _handle_snapshot(self, snapshot_data: OrderbookSnapshot) -> None:
        """Handle orderbook snapshot."""
        bids = [
            OrderbookLevel(price=price, size=size, norders=1, ticks=-1, lots=-1)
            for price, size in snapshot_data.bids
        ]
        asks = [
            OrderbookLevel(price=price, size=size, norders=1, ticks=-1, lots=-1)
            for price, size in snapshot_data.asks
        ]
        self.orderbook.consume_snapshot(bids=bids, asks=asks)
        self.snapshot_received = True
        self.logger.info(
            f"Orderbook snapshot received: {len(bids)} bids, {len(asks)} asks"
        )

    def _handle_delta(self, delta_data: OrderbookDelta) -> None:
        """Handle orderbook delta update."""
        if not self.snapshot_received:
            return

        bids = [
            OrderbookLevel(price=price, size=size, norders=1, ticks=-1, lots=-1)
            for price, size in delta_data.bids
        ]
        asks = [
            OrderbookLevel(price=price, size=size, norders=1, ticks=-1, lots=-1)
            for price, size in delta_data.asks
        ]
        self.orderbook.consume_deltas(bids=bids, asks=asks)

    def _handle_bbo(self, bbo_data: BBOUpdate) -> None:
        """Handle BBO update."""
        if not self.snapshot_received:
            return

        # Update BBO
        bid_level = OrderbookLevel(
            price=bbo_data.best_bid_price,
            size=bbo_data.best_bid_qty,
            norders=1,
            ticks=-1,
            lots=-1,
        )
        ask_level = OrderbookLevel(
            price=bbo_data.best_ask_price,
            size=bbo_data.best_ask_qty,
            norders=1,
            ticks=-1,
            lots=-1,
        )
        self.orderbook.consume_bbo(bid=bid_level, ask=ask_level)

        # Get mid price and log it
        mid_price = self.orderbook.get_mid_price()
        self.logger.info(f"Mid price: {mid_price:.2f}")

        # Create a trade-like object for candles (using mid price)
        current_time_ms = int(time.time() * 1000)
        trade = Trade(
            time_ms=current_time_ms,
            is_buy=True,  # Doesn't matter for mid price
            price=mid_price,
            size=0.0,  # No size for mid price
        )

        # Process trade for candles
        candle_count_before = len(self.time_candles)
        self.time_candles.process_trade(trade)
        candle_count_after = len(self.time_candles)

        # Check if a new candle was completed
        if candle_count_after > candle_count_before and candle_count_after > 1:
            completed_candle = self.time_candles[-2]
            if completed_candle.num_trades > 0:
                self.logger.info(
                    f"1s Candle: O={completed_candle.open_price:.2f} "
                    f"H={completed_candle.high_price:.2f} "
                    f"L={completed_candle.low_price:.2f} "
                    f"C={completed_candle.close_price:.2f} "
                    f"VWAP={completed_candle.vwap_price:.2f}"
                )

    def _process_message(self, msg_bytes: bytes) -> None:
        """Process a single message."""
        try:
            stream_msg = self.decoder.decode(msg_bytes)

            if stream_msg.msg_type == "snapshot":
                snapshot_data = msgspec.structs.from_dict(
                    OrderbookSnapshot, stream_msg.data
                )
                self._handle_snapshot(snapshot_data)

            elif stream_msg.msg_type == "delta":
                delta_data = msgspec.structs.from_dict(OrderbookDelta, stream_msg.data)
                self._handle_delta(delta_data)

            elif stream_msg.msg_type == "bbo":
                bbo_data = msgspec.structs.from_dict(BBOUpdate, stream_msg.data)
                self._handle_bbo(bbo_data)

        except Exception as e:
            self.logger.error(f"Error processing message: {e}")

    def run(self) -> None:
        """Run the data processor."""
        self.logger.info(f"Processing process started for {self.symbol}")
        self.logger.info("Waiting for orderbook snapshot...")

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
                    self.logger.info("Processing interrupted, shutting down...")
                    break
                except Exception as e:
                    self.logger.error(f"Error in processing loop: {e}")
                    time.sleep(0.1)

        except KeyboardInterrupt:
            self.logger.info("Processing process interrupted")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Shutdown the data processor."""
        self.data_consumer.stop()
        self.logger.shutdown()


def stream_process_entry(
    symbol: str,
    tick_size: float,
    lot_size: float,
    logger_path: str,
    data_path: str,
) -> None:
    """Entry point for stream process."""
    processor = BinanceStreamProcessor(symbol, logger_path, data_path)
    processor.run()


def processing_process_entry(
    symbol: str,
    tick_size: float,
    lot_size: float,
    logger_path: str,
    data_path: str,
) -> None:
    """Entry point for processing process."""
    processor = BinanceDataProcessor(
        symbol, tick_size, lot_size, logger_path, data_path
    )
    processor.run()


def main() -> None:
    """Main entry point."""
    symbol = "BTCUSDT"
    tick_size = 0.01
    lot_size = 0.001

    # IPC paths
    logger_path = "ipc:///tmp/binance_logger"
    data_path = "ipc:///tmp/binance_data"

    # Create processes
    stream_proc = multiprocessing.Process(
        target=stream_process_entry,
        args=(symbol, tick_size, lot_size, logger_path, data_path),
        daemon=True,
    )
    processing_proc = multiprocessing.Process(
        target=processing_process_entry,
        args=(symbol, tick_size, lot_size, logger_path, data_path),
        daemon=True,
    )

    # Start processes
    stream_proc.start()
    processing_proc.start()

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
