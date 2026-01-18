"""Hyperliquid stream processor implementation.

Subscribes to l2Book and trades channels, normalizes payloads, and emits
IPC messages for downstream processing.
"""

from __future__ import annotations

from typing import Any, Iterable

import msgspec

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
from examples.stream.hyperliquid.models import HyperliquidMessage
from mm_toolbox.time import time_ms


class HyperliquidStreamProcessor(BaseStreamProcessor):
    """Stream processor for Hyperliquid markets."""

    def __init__(self, symbol: str, ipc_path: str, logger_path: str) -> None:
        """Initialize the Hyperliquid stream processor.

        Args:
            symbol: Venue symbol.
            ipc_path: IPC path for outgoing normalized messages.
            logger_path: IPC path for worker logs.
        """
        super().__init__(
            venue=Venue.HYPERLIQUID,
            symbol=symbol,
            ipc_path=ipc_path,
            logger_path=logger_path,
        )
        self._decoder = msgspec.json.Decoder(type=HyperliquidMessage)
        self._orderbook_initialized = False

    def get_stream_url(self, msg_type: MsgType) -> str:
        """Return the Hyperliquid WebSocket URL for a message type.

        Args:
            msg_type: Message type for the stream.

        Returns:
            str: WebSocket URL.
        """
        match msg_type:
            case MsgType.BBO | MsgType.TRADE | MsgType.ORDERBOOK:
                return "wss://api.hyperliquid.xyz/ws"
            case _:
                return ""

    def get_subscribe_messages(self, msg_type: MsgType) -> list[bytes]:
        """Return subscription messages for the requested stream.

        Args:
            msg_type: Message type for the stream.

        Returns:
            list[bytes]: Subscription payloads.
        """
        match msg_type:
            case MsgType.BBO | MsgType.ORDERBOOK:
                payload = {
                    "method": "subscribe",
                    "subscription": {"type": "l2Book", "coin": self.symbol},
                }
                return [msgspec.json.encode(payload)]
            case MsgType.TRADE:
                payload = {
                    "method": "subscribe",
                    "subscription": {"type": "trades", "coin": self.symbol},
                }
                return [msgspec.json.encode(payload)]
            case _:
                return []

    def parse_bbo(self, msg: bytes) -> BBOUpdate | None:
        """Parse a Hyperliquid book message into a BBO update.

        Args:
            msg: Raw WebSocket message.

        Returns:
            BBOUpdate | None: Normalized BBO update.
        """
        payload = self._decoder.decode(msg)
        data = self._normalize_data(payload.data)
        bids, asks = self._extract_levels(data)
        if not bids or not asks:
            return None
        best_bid = max(bids, key=lambda x: x.price)
        best_ask = min(asks, key=lambda x: x.price)
        venue_ts_ms = self._extract_timestamp_ms(data)
        now_ms = time_ms()
        return BBOUpdate(
            venue=self.venue,
            symbol=self.symbol,
            venue_time_ms=venue_ts_ms,
            local_time_ms=now_ms,
            bid_price=best_bid.price,
            bid_size=best_bid.size,
            ask_price=best_ask.price,
            ask_size=best_ask.size,
        )

    def parse_trade(self, msg: bytes) -> TradeMsg | None:
        """Parse a Hyperliquid trade message into normalized trades.

        Args:
            msg: Raw WebSocket message.

        Returns:
            TradeMsg | None: Normalized trade message.
        """
        payload = self._decoder.decode(msg)
        data = self._normalize_data(payload.data)
        trade_entries = self._extract_trades(data)
        if not trade_entries:
            return None
        now_ms = time_ms()
        trades: list[Trade] = []
        for entry in trade_entries:
            trade_ts = self._normalize_timestamp(entry.get("time") or entry.get("timestamp"))
            if trade_ts is None:
                trade_ts = now_ms
            side = str(entry.get("side", "buy")).lower()
            price_raw = entry.get("px") or entry.get("price")
            size_raw = entry.get("sz") or entry.get("size")
            if price_raw is None or size_raw is None:
                continue
            try:
                price = float(price_raw)
                size = float(size_raw)
            except (TypeError, ValueError):
                continue
            if trade_ts <= 0 or price <= 0.0 or size < 0.0:
                continue
            trades.append(
                Trade(
                    time_ms=trade_ts,
                    is_buy=side in {"buy", "b"},
                    price=price,
                    size=size,
                )
            )
        if not trades:
            return None
        return TradeMsg(
            venue=self.venue,
            symbol=self.symbol,
            venue_time_ms=trades[0].time_ms,
            local_time_ms=now_ms,
            trades=trades,
        )

    def parse_orderbook(self, msg: bytes) -> OrderbookMsg | None:
        """Parse a Hyperliquid orderbook message into a snapshot or delta.

        Args:
            msg: Raw WebSocket message.

        Returns:
            OrderbookMsg | None: Normalized orderbook update.
        """
        payload = self._decoder.decode(msg)
        data = self._normalize_data(payload.data)
        bids, asks = self._extract_levels(data)
        if not bids and not asks:
            return None
        venue_ts_ms = self._extract_timestamp_ms(data)
        now_ms = time_ms()
        is_snapshot = bool(data.get("isSnapshot")) or payload.type == "snapshot"
        if not self._orderbook_initialized or is_snapshot:
            self._orderbook_initialized = True
            return OrderbookMsg(
                venue=self.venue,
                symbol=self.symbol,
                venue_time_ms=venue_ts_ms,
                local_time_ms=now_ms,
                bids=bids,
                asks=asks,
                is_bbo=False,
                is_snapshot=True,
            )
        return OrderbookMsg(
            venue=self.venue,
            symbol=self.symbol,
            venue_time_ms=venue_ts_ms,
            local_time_ms=now_ms,
            bids=bids,
            asks=asks,
            is_bbo=False,
            is_snapshot=False,
        )

    def _extract_trades(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract trade entries from Hyperliquid payload data.

        Args:
            data: Raw payload data.

        Returns:
            list[dict[str, Any]]: Trade entry dicts.
        """
        if isinstance(data.get("trades"), list):
            return [entry for entry in data["trades"] if isinstance(entry, dict)]
        if isinstance(data.get("data"), list):
            return [entry for entry in data["data"] if isinstance(entry, dict)]
        if isinstance(data, dict) and all(key in data for key in ("px", "sz")):
            return [data]
        return []

    def _normalize_data(self, data: dict[str, Any] | list[Any] | None) -> dict[str, Any]:
        """Normalize payload data to a dictionary.

        Args:
            data: Raw payload data.

        Returns:
            dict[str, Any]: Normalized data dictionary.
        """
        if isinstance(data, dict):
            return data
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return {"data": data}
        return {}

    def _extract_levels(
        self, data: dict[str, Any]
    ) -> tuple[list[OrderbookLevel], list[OrderbookLevel]]:
        """Extract bid and ask levels from Hyperliquid payload data.

        Args:
            data: Raw payload data.

        Returns:
            tuple[list[OrderbookLevel], list[OrderbookLevel]]: Bids and asks.
        """
        bids: list[OrderbookLevel] = []
        asks: list[OrderbookLevel] = []

        if isinstance(data.get("bids"), list) and isinstance(data.get("asks"), list):
            bids = self._parse_levels(data["bids"])
            asks = self._parse_levels(data["asks"])
            return bids, asks

        levels = data.get("levels")
        if isinstance(levels, list) and len(levels) == 2:
            bids = self._parse_levels(levels[0])
            asks = self._parse_levels(levels[1])
            return bids, asks

        if isinstance(levels, list):
            for entry in levels:
                if not isinstance(entry, dict):
                    continue
                side = str(entry.get("side", "")).lower()
                level = self._parse_level(entry)
                if level is None:
                    continue
                if side in {"bid", "buy", "b"}:
                    bids.append(level)
                elif side in {"ask", "sell", "a"}:
                    asks.append(level)
        return bids, asks

    def _parse_levels(self, levels: Iterable[Any]) -> list[OrderbookLevel]:
        """Parse a list of levels into price/size objects.

        Args:
            levels: Iterable of raw level entries.

        Returns:
            list[OrderbookLevel]: Parsed levels.
        """
        parsed: list[OrderbookLevel] = []
        for entry in levels:
            level = self._parse_level(entry)
            if level is not None:
                parsed.append(level)
        return parsed

    def _parse_level(self, entry: Any) -> OrderbookLevel | None:
        """Parse a single level entry.

        Args:
            entry: Raw level entry.

        Returns:
            OrderbookLevel | None: Parsed orderbook level when valid.
        """
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            return OrderbookLevel(price=float(entry[0]), size=float(entry[1]))
        if isinstance(entry, dict):
            price = entry.get("px") or entry.get("price") or entry.get("p")
            size = entry.get("sz") or entry.get("size") or entry.get("s")
            if price is None or size is None:
                return None
            return OrderbookLevel(price=float(price), size=float(size))
        return None


    def _extract_timestamp_ms(self, data: dict[str, Any]) -> int:
        """Extract a millisecond timestamp from payload data.

        Args:
            data: Raw payload data.

        Returns:
            int: Timestamp in milliseconds.
        """
        for key in ("time", "ts", "timestamp"):
            value = data.get(key)
            ts = self._normalize_timestamp(value)
            if ts is not None:
                return ts
        return time_ms()

    def _normalize_timestamp(self, value: Any) -> int | None:
        """Normalize timestamps to milliseconds.

        Args:
            value: Raw timestamp value.

        Returns:
            int | None: Timestamp in milliseconds when parseable.
        """
        if value is None:
            return None
        try:
            ts = int(value)
        except (TypeError, ValueError):
            return None
        if ts > 10_000_000_000_000:
            return ts // 1000
        if ts < 10_000_000_000:
            return ts * 1000
        return ts
