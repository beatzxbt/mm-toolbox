"""Binance stream processor implementation.

Handles Binance futures WebSocket streams, fetches REST snapshots, and
emits normalized BBO, trade, and orderbook messages via IPC.
"""

from __future__ import annotations

import asyncio

import aiohttp
import msgspec

from examples.stream.binance.models import (
    BinanceBBOMsg,
    BinanceOrderbookMsg,
    BinanceTradeMsg,
)
from examples.stream.core import (
    BBOUpdate,
    OrderbookLevel,
    OrderbookMsg,
    Trade,
    TradeMsg,
    Venue,
)
from examples.stream.core.base import BaseStreamProcessor
from examples.stream.core.models import MsgType
from mm_toolbox.time import time_ms
from mm_toolbox.websocket import WsConnectionConfig, WsPool, WsPoolConfig


class BinanceStreamProcessor(BaseStreamProcessor):
    """Stream processor for Binance futures."""

    def __init__(self, symbol: str, ipc_path: str, logger_path: str) -> None:
        """Initialize the Binance stream processor.

        Args:
            symbol: Venue symbol.
            ipc_path: IPC path for outgoing normalized messages.
            logger_path: IPC path for worker logs.
        """
        super().__init__(
            venue=Venue.BINANCE,
            symbol=symbol,
            ipc_path=ipc_path,
            logger_path=logger_path,
        )
        self._bbo_decoder = msgspec.json.Decoder(type=BinanceBBOMsg)
        self._trade_decoder = msgspec.json.Decoder(type=BinanceTradeMsg)
        self._orderbook_decoder = msgspec.json.Decoder(type=BinanceOrderbookMsg)

        self._snapshot_sent = False
        self._snapshot_update_id: int | None = None
        self._pending_deltas: list[BinanceOrderbookMsg] = []
        self._pending_delta_limit = 5000
        self._orderbook_lock = asyncio.Lock()

    def get_stream_url(self, msg_type: MsgType) -> str:
        """Return the Binance stream URL for a message type.

        Args:
            msg_type: Message type for the stream.

        Returns:
            str: WebSocket URL.
        """
        match msg_type:
            case MsgType.BBO:
                return f"wss://fstream.binance.com/ws/{self.symbol.lower()}@bookTicker"
            case MsgType.TRADE:
                return f"wss://fstream.binance.com/ws/{self.symbol.lower()}@trade"
            case MsgType.ORDERBOOK:
                return f"wss://fstream.binance.com/ws/{self.symbol.lower()}@depth@100ms"
            case _:
                return ""

    def parse_bbo(self, msg: bytes) -> BBOUpdate | None:
        """Parse a Binance book ticker message.

        Args:
            msg: Raw WebSocket message.

        Returns:
            BBOUpdate | None: Normalized BBO update.
        """
        payload = self._bbo_decoder.decode(msg)
        now_ms = time_ms()
        return BBOUpdate(
            venue=self.venue,
            symbol=payload.s,
            venue_time_ms=payload.E,
            local_time_ms=now_ms,
            bid_price=float(payload.b),
            bid_size=float(payload.B),
            ask_price=float(payload.a),
            ask_size=float(payload.A),
        )

    def parse_trade(self, msg: bytes) -> TradeMsg | None:
        """Parse a Binance trade message.

        Args:
            msg: Raw WebSocket message.

        Returns:
            TradeMsg | None: Normalized trade message.
        """
        payload = self._trade_decoder.decode(msg)
        now_ms = time_ms()
        price = float(payload.p)
        size = float(payload.q)
        if payload.T <= 0 or price <= 0.0 or size < 0.0:
            return None
        trade = Trade(
            time_ms=payload.T,
            is_buy=not payload.m,
            price=price,
            size=size,
        )
        return TradeMsg(
            venue=self.venue,
            symbol=payload.s,
            venue_time_ms=payload.T,
            local_time_ms=now_ms,
            trades=[trade],
        )

    def parse_orderbook(self, msg: bytes) -> OrderbookMsg | None:
        """Parse a Binance orderbook delta message.

        Args:
            msg: Raw WebSocket message.

        Returns:
            OrderbookMsg | None: Normalized orderbook update.
        """
        payload = self._orderbook_decoder.decode(msg)
        return self._build_orderbook_delta(payload)

    def _build_orderbook_delta(self, payload: BinanceOrderbookMsg) -> OrderbookMsg:
        """Normalize a depth payload into an orderbook message.

        Args:
            payload: Raw depth update payload.

        Returns:
            OrderbookMsg: Normalized orderbook update.
        """
        now_ms = time_ms()
        return OrderbookMsg(
            venue=self.venue,
            symbol=payload.s,
            venue_time_ms=payload.E,
            local_time_ms=now_ms,
            bids=self._build_levels(payload.b),
            asks=self._build_levels(payload.a),
            is_bbo=False,
            is_snapshot=False,
        )

    def _build_levels(self, raw_levels: list[list[str]]) -> list[OrderbookLevel]:
        """Build normalized orderbook levels from raw payload entries.

        Args:
            raw_levels: Raw levels as [price, size] strings.

        Returns:
            list[OrderbookLevel]: Parsed orderbook levels.
        """
        return [
            OrderbookLevel(price=float(price), size=float(size))
            for price, size in raw_levels
        ]

    async def _fetch_snapshot(self) -> tuple[OrderbookMsg, int] | None:
        """Fetch an orderbook snapshot from the Binance REST API.

        Returns:
            tuple[OrderbookMsg, int] | None: Snapshot and last update ID on success.
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
        except Exception as exc:
            self._logger.error(f"Failed to fetch snapshot: {exc}")
            return None

        try:
            now_ms = time_ms()
            last_update_id = int(payload["lastUpdateId"])
            snapshot = OrderbookMsg(
                venue=self.venue,
                symbol=self.symbol,
                venue_time_ms=now_ms,
                local_time_ms=now_ms,
                bids=[OrderbookLevel(price=float(p), size=float(q)) for p, q in payload["bids"]],
                asks=[OrderbookLevel(price=float(p), size=float(q)) for p, q in payload["asks"]],
                is_bbo=False,
                is_snapshot=True,
            )
            return snapshot, last_update_id
        except Exception as exc:
            self._logger.error(f"Failed to parse snapshot payload: {exc}")
            return None

    async def _send_snapshot_and_flush(self) -> None:
        """Fetch a snapshot and flush buffered deltas without alignment checks."""
        attempts = 0
        while not self._snapshot_sent and attempts < 5:
            attempts += 1
            snapshot_data = await self._fetch_snapshot()
            if snapshot_data is None:
                await asyncio.sleep(0.5)
                continue
            snapshot, last_update_id = snapshot_data
            async with self._orderbook_lock:
                pending = self._pending_deltas
                self._pending_deltas = []

                self._send_parsed_message(MsgType.ORDERBOOK, snapshot)
                self._snapshot_sent = True

                max_update_id = last_update_id
                flushed = 0
                if pending:
                    for payload in pending:
                        if payload.u <= last_update_id:
                            continue
                        delta = self._build_orderbook_delta(payload)
                        self._send_parsed_message(MsgType.ORDERBOOK, delta)
                        flushed += 1
                        if payload.u > max_update_id:
                            max_update_id = payload.u
                self._snapshot_update_id = max_update_id

            if pending:
                self._logger.info(
                    f"[{self.venue}] Snapshot sent; flushed "
                    f"{flushed} of {len(pending)} buffered deltas"
                )
            else:
                self._logger.info(
                    f"[{self.venue}] Snapshot sent; no buffered deltas to flush"
                )
            return

        if not self._snapshot_sent:
            self._logger.error("Failed to fetch orderbook snapshot")

    async def _process_orderbook_message(self, payload: BinanceOrderbookMsg) -> None:
        """Process orderbook depth messages.

        Args:
            payload: Decoded depth update payload.
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
            self._send_parsed_message(MsgType.ORDERBOOK, delta)
            self._snapshot_update_id = payload.u

    async def _run_stream(self, msg_type: MsgType) -> None:
        """Override orderbook stream to include snapshot bootstrap.

        Args:
            msg_type: Message type for the stream.
        """
        match msg_type:
            case MsgType.ORDERBOOK:
                pass
            case _:
                await super()._run_stream(msg_type)
                return

        orderbook_config = WsConnectionConfig.default(wss_url=self.get_stream_url(msg_type))
        self._orderbook_pool = await WsPool.new(
            config=orderbook_config,
            on_message=self._noop,
            pool_config=WsPoolConfig.default(),
        )

        async with self._orderbook_pool:
            self._logger.info("[binance] ORDERBOOK stream connected, buffering deltas")

            async def consume_orderbook() -> None:
                async for msg in self._orderbook_pool:
                    payload = self._orderbook_decoder.decode(msg)
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

            await orderbook_task
