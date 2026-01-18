"""Bybit stream processor implementation.

Subscribes to public linear channels and emits normalized BBO, trade,
and orderbook messages via IPC.
"""

from __future__ import annotations

from typing import Any

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
from mm_toolbox.time import time_ms


class BybitStreamProcessor(BaseStreamProcessor):
    """Stream processor for Bybit linear markets."""

    def __init__(self, symbol: str, ipc_path: str, logger_path: str) -> None:
        """Initialize the Bybit stream processor.

        Args:
            symbol: Venue symbol.
            ipc_path: IPC path for outgoing normalized messages.
            logger_path: IPC path for worker logs.
        """
        super().__init__(
            venue=Venue.BYBIT,
            symbol=symbol,
            ipc_path=ipc_path,
            logger_path=logger_path,
        )
        self._orderbook_depth: str | None = None

    def get_stream_url(self, msg_type: MsgType) -> str:
        """Return the Bybit public WebSocket URL for a message type.

        Args:
            msg_type: Message type for the stream.

        Returns:
            str: WebSocket URL.
        """
        match msg_type:
            case MsgType.BBO | MsgType.TRADE | MsgType.ORDERBOOK:
                return "wss://stream.bybit.com/v5/public/linear"
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
            case MsgType.BBO:
                payload = {"op": "subscribe", "args": [f"orderbook.1.{self.symbol}"]}
                return [msgspec.json.encode(payload)]
            case MsgType.TRADE:
                payload = {"op": "subscribe", "args": [f"publicTrade.{self.symbol}"]}
                return [msgspec.json.encode(payload)]
            case MsgType.ORDERBOOK:
                depths = ("500", "200", "50")
                return [
                    msgspec.json.encode(
                        {"op": "subscribe", "args": [f"orderbook.{depth}.{self.symbol}"]}
                    )
                    for depth in depths
                ]
            case _:
                return []

    def parse_bbo(self, msg: bytes) -> BBOUpdate | None:
        """Parse a Bybit orderbook-1 message into a BBO update.

        Args:
            msg: Raw WebSocket message.

        Returns:
            BBOUpdate | None: Normalized BBO update.
        """
        payload = self._decode_payload(msg)
        if payload is None:
            return None
        topic = self._match_topic(payload, "orderbook.1.")
        if topic is None:
            return None
        data = payload.get("data")
        if isinstance(data, list):
            data = data[0] if data else None
        if not isinstance(data, dict):
            return None
        bids = data.get("b")
        asks = data.get("a")
        if not bids or not asks:
            return None
        bid = bids[0]
        ask = asks[0]
        if not self._is_level(bid) or not self._is_level(ask):
            return None
        now_ms = time_ms()
        try:
            venue_ts = int(payload.get("ts") or data.get("ts") or 0)
        except (TypeError, ValueError):
            venue_ts = now_ms
        if venue_ts <= 0:
            venue_ts = now_ms
        symbol = data.get("s") if isinstance(data.get("s"), str) else self.symbol
        return BBOUpdate(
            venue=self.venue,
            symbol=symbol,
            venue_time_ms=venue_ts,
            local_time_ms=now_ms,
            bid_price=float(bid[0]),
            bid_size=float(bid[1]),
            ask_price=float(ask[0]),
            ask_size=float(ask[1]),
        )

    def parse_trade(self, msg: bytes) -> TradeMsg | None:
        """Parse a Bybit trade message into normalized trades.

        Args:
            msg: Raw WebSocket message.

        Returns:
            TradeMsg | None: Normalized trade message.
        """
        payload = self._decode_payload(msg)
        if payload is None:
            return None
        topic = self._match_topic(payload, "publicTrade.")
        if topic is None:
            return None
        data = payload.get("data")
        if not isinstance(data, list):
            return None
        now_ms = time_ms()
        trades: list[Trade] = []
        for entry in data:
            if not isinstance(entry, dict):
                continue
            raw_time = entry.get("T") or entry.get("t") or payload.get("ts")
            price_raw = entry.get("p")
            size_raw = entry.get("v")
            side = entry.get("S")
            if raw_time is None or price_raw is None or size_raw is None or side is None:
                continue
            try:
                trade_time = int(raw_time)
                price = float(price_raw)
                size = float(size_raw)
            except (TypeError, ValueError):
                continue
            if trade_time <= 0 or price <= 0.0 or size < 0.0:
                continue
            trades.append(
                Trade(
                    time_ms=trade_time,
                    is_buy=str(side).lower() == "buy",
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
        """Parse a Bybit orderbook message into a normalized update.

        Args:
            msg: Raw WebSocket message.

        Returns:
            OrderbookMsg | None: Normalized orderbook update.
        """
        payload = self._decode_payload(msg)
        if payload is None:
            return None
        topic = payload.get("topic")
        if not isinstance(topic, str) or not topic.startswith("orderbook."):
            return None
        depth = topic.split(".")[1] if "." in topic else ""
        if depth == "1":
            return None
        if self._orderbook_depth is None:
            self._orderbook_depth = depth
        elif depth != self._orderbook_depth:
            return None
        data = payload.get("data")
        if isinstance(data, list):
            data = data[0] if data else None
        if not isinstance(data, dict):
            return None
        bids = self._parse_levels(data.get("b") or data.get("bids"))
        asks = self._parse_levels(data.get("a") or data.get("asks"))
        if not bids and not asks:
            return None
        now_ms = time_ms()
        try:
            venue_ts = int(payload.get("ts") or data.get("ts") or 0)
        except (TypeError, ValueError):
            venue_ts = now_ms
        if venue_ts <= 0:
            venue_ts = now_ms
        symbol = data.get("s") if isinstance(data.get("s"), str) else self.symbol
        return OrderbookMsg(
            venue=self.venue,
            symbol=symbol,
            venue_time_ms=venue_ts,
            local_time_ms=now_ms,
            bids=bids,
            asks=asks,
            is_bbo=False,
            is_snapshot=str(payload.get("type", "")).lower() == "snapshot",
        )

    def _decode_payload(self, msg: bytes) -> dict[str, Any] | None:
        """Decode a raw message into a dictionary payload.

        Args:
            msg: Raw WebSocket message.

        Returns:
            dict[str, Any] | None: Parsed payload.
        """
        try:
            raw = msgspec.json.decode(msg)
        except Exception:
            return None
        if not isinstance(raw, dict):
            return None
        return raw

    def _match_topic(self, payload: dict[str, Any], prefix: str) -> str | None:
        """Return topic if it matches the requested prefix.

        Args:
            payload: Raw payload dictionary.
            prefix: Required topic prefix.

        Returns:
            str | None: Topic string when it matches.
        """
        topic = payload.get("topic")
        if isinstance(topic, str) and topic.startswith(prefix):
            return topic
        return None

    def _parse_levels(self, levels: Any) -> list[OrderbookLevel]:
        """Parse orderbook levels into normalized orderbook levels.

        Args:
            levels: Raw level list.

        Returns:
            list[OrderbookLevel]: Parsed levels.
        """
        if not isinstance(levels, list):
            return []
        parsed: list[OrderbookLevel] = []
        for entry in levels:
            if not self._is_level(entry):
                continue
            parsed.append(OrderbookLevel(price=float(entry[0]), size=float(entry[1])))
        return parsed

    def _is_level(self, entry: Any) -> bool:
        """Check if a raw entry looks like a price/size level.

        Args:
            entry: Raw level entry.

        Returns:
            bool: True if the entry resembles a level tuple.
        """
        return isinstance(entry, (list, tuple)) and len(entry) >= 2
