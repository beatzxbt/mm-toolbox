"""Comprehensive example demonstrating multi-process Binance streaming with IPC.

This example shows:
- Using WsPool for all WebSocket streams (BBO, depth, and trade)
- Building time-based candles from trade feed data
- Building standard orderbook
- IPC communication between processes
- Worker logger in stream process, master logger in processing process
"""

from __future__ import annotations

import asyncio
import multiprocessing
import sys
import time
from enum import StrEnum

import aiohttp
import msgspec

from mm_toolbox.candles import TimeCandles
from mm_toolbox.candles import Trade as CandleTrade
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
class BBOUpdate(msgspec.Struct, tag="bbo"):
    """Best bid/offer update message."""

    event_time: int
    update_id: int
    symbol: str
    best_bid_price: float
    best_bid_size: float
    best_ask_price: float
    best_ask_size: float


class OrderbookSnapshot(msgspec.Struct, tag="snapshot"):
    """Orderbook snapshot message."""

    last_update_id: int
    bids: list[tuple[float, float]]
    asks: list[tuple[float, float]]


class OrderbookDelta(msgspec.Struct, tag="delta"):
    """Orderbook delta update message."""

    first_update_id: int
    final_update_id: int
    prev_update_id: int
    bids: list[tuple[float, float]]
    asks: list[tuple[float, float]]


class Trade(msgspec.Struct, tag="trade"):
    """Trade message for IPC communication."""

    time_ms: int
    is_buy: bool
    price: float
    size: float


class StreamMessageType(StrEnum):
    ORDERBOOK_BBO = "orderbook_bbo"
    ORDERBOOK_SNAPSHOT = "orderbook_snapshot"
    ORDERBOOK_DELTA = "orderbook_delta"
    TRADES = "trades"


class StreamMessage(msgspec.Struct):
    """Wrapper for stream messages."""

    msg_type: StreamMessageType
    data: BBOUpdate | OrderbookSnapshot | OrderbookDelta | Trade


class BinanceStreamMsg(msgspec.Struct, tag_field="e"):
    """Base class for tagged Binance stream payloads."""


class BinanceBBOMsg(BinanceStreamMsg, tag="bookTicker"):
    """Book ticker payload from Binance futures streams.

    Attributes:
        s (str): Symbol.
        u (int): Order book update ID.
        b (str): Best bid price.
        B (str): Best bid quantity.
        a (str): Best ask price.
        A (str): Best ask quantity.
        T (int): Transaction time in milliseconds.
        E (int): Event time in milliseconds.
    """

    s: str
    u: int
    b: str
    B: str
    a: str
    A: str
    T: int
    E: int


class BinanceTradeMsg(BinanceStreamMsg, tag="trade"):
    """Trade payload from Binance futures streams.

    Attributes:
        E (int): Event time in milliseconds.
        T (int): Trade time in milliseconds.
        s (str): Symbol.
        t (int): Trade ID.
        p (str): Trade price.
        q (str): Trade quantity.
        X (str): Trade type (e.g., MARKET).
        m (bool): True if buyer is the market maker.
    """

    E: int
    T: int
    s: str
    t: int
    p: str
    q: str
    X: str
    m: bool


class BinanceOrderbookMsg(BinanceStreamMsg, tag="depthUpdate"):
    """Depth update payload from Binance futures streams.

    Attributes:
        E (int): Event time in milliseconds.
        T (int): Transaction time in milliseconds.
        s (str): Symbol.
        U (int): First update ID in the event.
        u (int): Final update ID in the event.
        b (list[list[str]]): Bid updates as [price, quantity] strings.
        a (list[list[str]]): Ask updates as [price, quantity] strings.
        pu (int | None): Previous update ID when present.
    """

    E: int
    T: int
    s: str
    U: int
    u: int
    b: list[list[str]]
    a: list[list[str]]
    pu: int | None = None


BinanceStreamPayload = BinanceBBOMsg | BinanceTradeMsg | BinanceOrderbookMsg


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

        # Message encoder/decoder
        self.encoder = msgspec.json.Encoder()
        self.ws_decoder = msgspec.json.Decoder(type=BinanceStreamPayload)

        # WebSocket pools
        self.bbo_pool: WsPool | None = None
        self.trade_pool: WsPool | None = None
        self.orderbook_pool: WsPool | None = None
        self._snapshot_sent = False
        self._snapshot_update_id: int | None = None
        self._pending_deltas: list[BinanceOrderbookMsg] = []
        self._pending_delta_limit = 5000
        self._orderbook_lock = asyncio.Lock()

    def _noop(self, msg: bytes) -> None:
        """Ignore callback-based messages from the WebSocket pool.

        Args:
            msg (bytes): Raw WebSocket message.

        Returns:
            None: This method does not return a value.
        """
        return

    def _decode_ws_message(self, msg: bytes) -> BinanceStreamPayload:
        """Decode a WebSocket payload into a typed message.

        Args:
            msg (bytes): Raw WebSocket message payload.

        Returns:
            BinanceStreamPayload: Decoded payload.
        """
        try:
            return self.ws_decoder.decode(msg)
        except Exception as e:
            self.logger.error(f"Failed to decode message: {e}")
            self.logger.error(f"Raw payload: {msg.decode('utf-8')}")
            raise

    def _build_orderbook_delta(self, payload: BinanceOrderbookMsg) -> OrderbookDelta:
        """Normalize a depth payload into an orderbook delta message.

        Args:
            payload (BinanceOrderbookMsg): Raw depth update payload.

        Returns:
            OrderbookDelta: Normalized delta update.
        """
        return OrderbookDelta(
            first_update_id=payload.U,
            final_update_id=payload.u,
            prev_update_id=payload.pu,
            bids=[(float(price), float(size)) for price, size in payload.b],
            asks=[(float(price), float(size)) for price, size in payload.a],
        )

    def _process_bbo_message(self, payload: BinanceBBOMsg) -> None:
        """Process a book ticker message.

        Args:
            payload (BinanceBBOMsg): Decoded book ticker payload.

        Returns:
            None: This method does not return a value.
        """
        try:
            bbo = BBOUpdate(
                event_time=payload.E,
                update_id=payload.u,
                symbol=payload.s,
                best_bid_price=float(payload.b),
                best_bid_size=float(payload.B),
                best_ask_price=float(payload.a),
                best_ask_size=float(payload.A),
            )
            stream_msg = StreamMessage(
                msg_type=StreamMessageType.ORDERBOOK_BBO, data=bbo
            )
            self.data_producer.insert(self.encoder.encode(stream_msg), copy=False)
        except Exception as e:
            self.logger.error(f"Error processing BBO message: {e}")

    def _process_trade_message(self, payload: BinanceTradeMsg) -> None:
        """Process a trade message.

        Args:
            payload (BinanceTradeMsg): Decoded trade payload.

        Returns:
            None: This method does not return a value.
        """
        try:
            trade = Trade(
                time_ms=payload.T,
                is_buy=not payload.m,
                price=float(payload.p),
                size=float(payload.q),
            )
            stream_msg = StreamMessage(msg_type=StreamMessageType.TRADES, data=trade)
            self.data_producer.insert(self.encoder.encode(stream_msg), copy=False)
        except Exception as e:
            self.logger.error(f"Error processing trade message: {e}")

    async def _fetch_snapshot(self) -> OrderbookSnapshot | None:
        """Fetch an orderbook snapshot from the REST API.

        Returns:
            OrderbookSnapshot | None: Snapshot data on success, otherwise None.
        """
        url = (
            "https://fapi.binance.com/fapi/v1/depth?"
            f"symbol={self.symbol.upper()}&limit=1000"
        )
        timeout = aiohttp.ClientTimeout(total=5)
        headers = {"User-Agent": "mm-toolbox-example"}
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers) as response:
                    response.raise_for_status()
                    payload = await response.json()
        except Exception as e:
            self.logger.error(f"Failed to fetch snapshot: {e}")
            return None

        try:
            return OrderbookSnapshot(
                last_update_id=payload["lastUpdateId"],
                bids=[(float(p), float(q)) for p, q in payload["bids"]],
                asks=[(float(p), float(q)) for p, q in payload["asks"]],
            )
        except Exception as e:
            self.logger.error(f"Failed to parse snapshot payload: {e}")
            return None

    async def _send_snapshot_and_flush(self) -> None:
        """Fetch a snapshot and flush buffered deltas without alignment checks.

        Returns:
            None: This method does not return a value.
        """
        attempts = 0
        while not self._snapshot_sent and attempts < 5:
            attempts += 1
            snapshot = await self._fetch_snapshot()
            if snapshot is None:
                await asyncio.sleep(0.5)
                continue

            async with self._orderbook_lock:
                pending = self._pending_deltas
                self._pending_deltas = []

                stream_msg = StreamMessage(
                    msg_type=StreamMessageType.ORDERBOOK_SNAPSHOT, data=snapshot
                )
                self.data_producer.insert(self.encoder.encode(stream_msg), copy=False)
                self._snapshot_sent = True

                max_update_id = snapshot.last_update_id
                flushed = 0
                if pending:
                    for payload in pending:
                        if payload.u <= snapshot.last_update_id:
                            continue
                        delta = self._build_orderbook_delta(payload)
                        delta_msg = StreamMessage(
                            msg_type=StreamMessageType.ORDERBOOK_DELTA, data=delta
                        )
                        self.data_producer.insert(
                            self.encoder.encode(delta_msg), copy=False
                        )
                        flushed += 1
                        if payload.u > max_update_id:
                            max_update_id = payload.u
                self._snapshot_update_id = max_update_id

            if pending:
                self.logger.info(
                    "Snapshot sent; flushed "
                    f"{flushed} of {len(pending)} buffered deltas"
                )
            else:
                self.logger.info("Snapshot sent; no buffered deltas to flush")
            return

        if not self._snapshot_sent:
            self.logger.error("Failed to fetch orderbook snapshot")

    async def _process_orderbook_message(self, payload: BinanceOrderbookMsg) -> None:
        """Process orderbook depth messages.

        Args:
            payload (BinanceOrderbookMsg): Decoded depth update payload.

        Returns:
            None: This method does not return a value.
        """
        async with self._orderbook_lock:
            if not self._snapshot_sent:
                if len(self._pending_deltas) >= self._pending_delta_limit:
                    self._pending_deltas.pop(0)
                self._pending_deltas.append(payload)
                return

            if (
                self._snapshot_update_id is not None
                and payload.u <= self._snapshot_update_id
            ):
                return

            delta = self._build_orderbook_delta(payload)
            stream_msg = StreamMessage(
                msg_type=StreamMessageType.ORDERBOOK_DELTA, data=delta
            )
            self.data_producer.insert(self.encoder.encode(stream_msg), copy=False)
            self._snapshot_update_id = delta.final_update_id

    async def _run_streams(self) -> None:
        """Run WebSocket streams."""
        self.logger.info(f"Starting streams for {self.symbol}")
        # Orderbook stream (buffer deltas before sending snapshot)
        orderbook_config = WsConnectionConfig.default(
            wss_url=f"wss://fstream.binance.com/ws/{self.symbol.lower()}@depth@100ms"
        )
        self.orderbook_pool = await WsPool.new(
            config=orderbook_config,
            on_message=self._noop,
            pool_config=WsPoolConfig.default(),
        )

        async with self.orderbook_pool:
            self.logger.info("Orderbook stream connected, buffering deltas...")
            try:

                async def consume_orderbook() -> None:
                    async for msg in self.orderbook_pool:
                        payload = self._decode_ws_message(msg)
                        if isinstance(payload, BinanceOrderbookMsg):
                            await self._process_orderbook_message(payload)

                orderbook_task = asyncio.create_task(consume_orderbook())
                await self._send_snapshot_and_flush()

                if not self._snapshot_sent:
                    orderbook_task.cancel()
                    try:
                        await orderbook_task
                    except asyncio.CancelledError:
                        pass
                    return

                # BBO stream
                bbo_config = WsConnectionConfig.default(
                    wss_url=(
                        "wss://fstream.binance.com/ws/"
                        f"{self.symbol.lower()}@bookTicker"
                    )
                )
                self.bbo_pool = await WsPool.new(
                    config=bbo_config,
                    on_message=self._noop,
                    pool_config=WsPoolConfig.default(),
                )

                # Trade stream
                trade_config = WsConnectionConfig.default(
                    wss_url=f"wss://fstream.binance.com/ws/{self.symbol.lower()}@trade"
                )
                self.trade_pool = await WsPool.new(
                    config=trade_config,
                    on_message=self._noop,
                    pool_config=WsPoolConfig.default(),
                )

                async with self.bbo_pool, self.trade_pool:
                    self.logger.info("Streams connected, processing messages...")

                    async def consume_bbo() -> None:
                        async for msg in self.bbo_pool:
                            payload = self._decode_ws_message(msg)
                            if isinstance(payload, BinanceBBOMsg):
                                self._process_bbo_message(payload)

                    async def consume_trades() -> None:
                        async for msg in self.trade_pool:
                            payload = self._decode_ws_message(msg)
                            if isinstance(payload, BinanceTradeMsg):
                                self._process_trade_message(payload)

                    await asyncio.gather(
                        orderbook_task,
                        consume_bbo(),
                        consume_trades(),
                        return_exceptions=True,
                    )
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
        if self.trade_pool is not None:
            self.trade_pool.close()
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

        self.orderbook = Orderbook(
            tick_size=tick_size,
            lot_size=lot_size,
            size=1000,
        )

        self.time_candles = TimeCandles(secs_per_bucket=1.0, num_candles=100)

        self.decoder = msgspec.json.Decoder(type=StreamMessage)

        # State
        self.snapshot_received = False
        self._last_logged_mid: float | None = None
        self._latest_update_id: int | None = None

    def _build_orderbook_levels(
        self, price_size_pairs: list[tuple[float, float]]
    ) -> list[OrderbookLevel]:
        """Build OrderbookLevel list from price/size tuples.

        Args:
            price_size_pairs (list[tuple[float, float]]): List of (price, size) tuples.

        Returns:
            list[OrderbookLevel]: List of OrderbookLevel objects.
        """
        return [
            OrderbookLevel.from_values(
                price=price,
                size=size,
                norders=1,
                tick_size=self.tick_size,
                lot_size=self.lot_size,
            )
            for price, size in price_size_pairs
        ]

    def _handle_snapshot(self, snapshot_data: OrderbookSnapshot) -> None:
        """Handle orderbook snapshot."""
        bids = self._build_orderbook_levels(snapshot_data.bids)
        asks = self._build_orderbook_levels(snapshot_data.asks)
        self.orderbook.consume_snapshot(asks=asks, bids=bids)
        self.snapshot_received = True
        self._latest_update_id = snapshot_data.last_update_id
        self.logger.info(
            f"Orderbook snapshot received: {len(bids)} bids, {len(asks)} asks"
        )

    def _handle_delta(self, delta_data: OrderbookDelta) -> None:
        """Handle orderbook delta update."""
        if not self.snapshot_received:
            return

        if (
            self._latest_update_id is not None
            and delta_data.final_update_id < self._latest_update_id
        ):
            return

        bids = self._build_orderbook_levels(delta_data.bids)
        asks = self._build_orderbook_levels(delta_data.asks)
        self.orderbook.consume_deltas(asks=asks, bids=bids)
        if (
            self._latest_update_id is None
            or delta_data.final_update_id > self._latest_update_id
        ):
            self._latest_update_id = delta_data.final_update_id

    def _handle_bbo(self, bbo_data: BBOUpdate) -> None:
        """Handle BBO updates.

        Args:
            bbo_data (BBOUpdate): Normalized BBO update.

        Returns:
            None: This method does not return a value.
        """
        if not self.snapshot_received:
            return

        if (
            self._latest_update_id is not None
            and bbo_data.update_id < self._latest_update_id
        ):
            return

        # Build orderbook levels for best bid and ask
        best_bid = OrderbookLevel.from_values(
            price=bbo_data.best_bid_price,
            size=bbo_data.best_bid_size,
            norders=1,
            tick_size=self.tick_size,
            lot_size=self.lot_size,
        )
        best_ask = OrderbookLevel.from_values(
            price=bbo_data.best_ask_price,
            size=bbo_data.best_ask_size,
            norders=1,
            tick_size=self.tick_size,
            lot_size=self.lot_size,
        )

        # Update orderbook with BBO
        self.orderbook.consume_bbo(ask=best_ask, bid=best_bid)

        if (
            self._latest_update_id is None
            or bbo_data.update_id > self._latest_update_id
        ):
            self._latest_update_id = bbo_data.update_id

        # Log mid price changes
        mid_price = (bbo_data.best_bid_price + bbo_data.best_ask_price) / 2.0
        if self._last_logged_mid is None or mid_price != self._last_logged_mid:
            self.logger.info(f"Mid price: {mid_price:.2f}")
            self._last_logged_mid = mid_price

    def _handle_trade(self, trade: CandleTrade) -> None:
        """Handle trade updates for candle building.

        Args:
            trade (CandleTrade): Normalized trade update.

        Returns:
            None: This method does not return a value.
        """
        if not self.snapshot_received:
            return

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

            match stream_msg.msg_type:
                case StreamMessageType.ORDERBOOK_SNAPSHOT:
                    self._handle_snapshot(stream_msg.data)
                case StreamMessageType.ORDERBOOK_DELTA:
                    self._handle_delta(stream_msg.data)
                case StreamMessageType.ORDERBOOK_BBO:
                    self._handle_bbo(stream_msg.data)
                case StreamMessageType.TRADES:
                    # Convert to CandleTrade for candle building
                    candle_trade = CandleTrade(
                        time_ms=stream_msg.data.time_ms,
                        is_buy=stream_msg.data.is_buy,
                        price=stream_msg.data.price,
                        size=stream_msg.data.size,
                    )
                    self._handle_trade(candle_trade)

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

    print(f"Starting stream and processing processes for {symbol}...")
    sys.stdout.flush()

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
    print(f"Stream process started (PID: {stream_proc.pid})")
    sys.stdout.flush()

    processing_proc.start()
    print(f"Processing process started (PID: {processing_proc.pid})")
    sys.stdout.flush()

    print("Press Ctrl+C to stop...")
    sys.stdout.flush()

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
