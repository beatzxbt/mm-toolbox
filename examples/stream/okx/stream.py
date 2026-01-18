"""OKX stream processor implementation.

Subscribes to OKX public channels and emits normalized BBO, trade, and
orderbook messages via IPC.
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


class OkxStreamProcessor(BaseStreamProcessor):
    """Stream processor for OKX swap markets."""

    def __init__(self, symbol: str, ipc_path: str, logger_path: str) -> None:
        """Initialize the OKX stream processor.

        Args:
            symbol: Venue symbol.
            ipc_path: IPC path for outgoing normalized messages.
            logger_path: IPC path for worker logs.
        """
        super().__init__(
            venue=Venue.OKX,
            symbol=symbol,
            ipc_path=ipc_path,
            logger_path=logger_path,
        )

    def get_stream_url(self, msg_type: MsgType) -> str:
        """Return the OKX public WebSocket URL for a message type.

        Args:
            msg_type: Message type for the stream.

        Returns:
            str: WebSocket URL.
        """
        match msg_type:
            case MsgType.BBO | MsgType.TRADE | MsgType.ORDERBOOK:
                return "wss://ws.okx.com:8443/ws/v5/public"
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
                payload = {
                    "op": "subscribe",
                    "args": [{"channel": "bbo-tbt", "instId": self.symbol}],
                }
                return [msgspec.json.encode(payload)]
            case MsgType.TRADE:
                payload = {
                    "op": "subscribe",
                    "args": [{"channel": "trades", "instId": self.symbol}],
                }
                return [msgspec.json.encode(payload)]
            case MsgType.ORDERBOOK:
                payload = {
                    "op": "subscribe",
                    "args": [{"channel": "books", "instId": self.symbol}],
                }
                return [msgspec.json.encode(payload)]
            case _:
                return []

    def parse_bbo(self, msg: bytes) -> BBOUpdate | None:
        """Parse an OKX BBO message.

        Args:
            msg: Raw WebSocket message.

        Returns:
            BBOUpdate | None: Normalized BBO update.
        """
        payload = self._decode_payload(msg)
        if payload is None:
            return None
        arg = self._match_channel(payload, "bbo-tbt")
        if arg is None:
            return None
        data = payload.get("data")
        if not isinstance(data, list) or not data:
            return None
        entry = data[0]
        if not isinstance(entry, dict):
            return None
        now_ms = time_ms()
        try:
            venue_ts = int(entry.get("ts") or 0)
        except (TypeError, ValueError):
            venue_ts = now_ms
        if venue_ts <= 0:
            venue_ts = now_ms
        return BBOUpdate(
            venue=self.venue,
            symbol=str(arg.get("instId", self.symbol)),
            venue_time_ms=venue_ts,
            local_time_ms=now_ms,
            bid_price=float(entry.get("bidPx", 0.0)),
            bid_size=float(entry.get("bidSz", 0.0)),
            ask_price=float(entry.get("askPx", 0.0)),
            ask_size=float(entry.get("askSz", 0.0)),
        )

    def parse_trade(self, msg: bytes) -> TradeMsg | None:
        """Parse an OKX trade message.

        Args:
            msg: Raw WebSocket message.

        Returns:
            TradeMsg | None: Normalized trade message.
        """
        payload = self._decode_payload(msg)
        if payload is None:
            return None
        arg = self._match_channel(payload, "trades")
        if arg is None:
            return None
        data = payload.get("data")
        if not isinstance(data, list) or not data:
            return None
        now_ms = time_ms()
        trades: list[Trade] = []
        for entry in data:
            if not isinstance(entry, dict):
                continue
            raw_ts = entry.get("ts")
            price_raw = entry.get("px")
            size_raw = entry.get("sz")
            side = entry.get("side")
            if raw_ts is None or price_raw is None or size_raw is None or side is None:
                continue
            try:
                trade_ts = int(raw_ts)
                price = float(price_raw)
                size = float(size_raw)
            except (TypeError, ValueError):
                continue
            if trade_ts <= 0 or price <= 0.0 or size < 0.0:
                continue
            trades.append(
                Trade(
                    time_ms=trade_ts,
                    is_buy=str(side).lower() == "buy",
                    price=price,
                    size=size,
                )
            )
        if not trades:
            return None
        return TradeMsg(
            venue=self.venue,
            symbol=str(arg.get("instId", self.symbol)),
            venue_time_ms=trades[0].time_ms,
            local_time_ms=now_ms,
            trades=trades,
        )

    def parse_orderbook(self, msg: bytes) -> OrderbookMsg | None:
        """Parse an OKX orderbook message.

        Args:
            msg: Raw WebSocket message.

        Returns:
            OrderbookMsg | None: Normalized orderbook update.
        """
        payload = self._decode_payload(msg)
        if payload is None:
            return None
        arg = self._match_channel(payload, "books")
        if arg is None:
            return None
        data = payload.get("data")
        if not isinstance(data, list) or not data:
            return None
        entry = data[0]
        if not isinstance(entry, dict):
            return None
        bids = self._parse_levels(entry.get("bids"))
        asks = self._parse_levels(entry.get("asks"))
        if not bids and not asks:
            return None
        now_ms = time_ms()
        try:
            venue_ts = int(entry.get("ts") or 0)
        except (TypeError, ValueError):
            venue_ts = now_ms
        if venue_ts <= 0:
            venue_ts = now_ms
        return OrderbookMsg(
            venue=self.venue,
            symbol=str(arg.get("instId", self.symbol)),
            venue_time_ms=venue_ts,
            local_time_ms=now_ms,
            bids=bids,
            asks=asks,
            is_bbo=False,
            is_snapshot=str(payload.get("action", "")).lower() == "snapshot",
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
        if "arg" not in raw or "data" not in raw:
            return None
        return raw

    def _match_channel(self, payload: dict[str, Any], channel: str) -> dict[str, Any] | None:
        """Return the arg dict when it matches the requested channel.

        Args:
            payload: Raw payload dictionary.
            channel: Channel name.

        Returns:
            dict[str, Any] | None: Arg dictionary when channel matches.
        """
        arg = payload.get("arg")
        if not isinstance(arg, dict):
            return None
        if arg.get("channel") != channel:
            return None
        return arg

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
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                continue
            try:
                price = float(entry[0])
                size = float(entry[1])
            except (TypeError, ValueError):
                continue
            parsed.append(OrderbookLevel(price=price, size=size))
        return parsed
