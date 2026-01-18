"""Lighter stream processor implementation.

Subscribes to order book and trade channels, normalizes payloads, and
emits IPC messages for downstream processing.
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


class LighterStreamProcessor(BaseStreamProcessor):
    """Stream processor for Lighter markets."""

    def __init__(self, symbol: str, ipc_path: str, logger_path: str) -> None:
        """Initialize the Lighter stream processor.

        Args:
            symbol: Venue symbol.
            ipc_path: IPC path for outgoing normalized messages.
            logger_path: IPC path for worker logs.
        """
        super().__init__(
            venue=Venue.LIGHTER,
            symbol=symbol,
            ipc_path=ipc_path,
            logger_path=logger_path,
        )
        self._orderbook_initialized = False
        self._orderbook_channel: str | None = None
        self._trade_channel: str | None = None
        self._resolved_market_index: str | None = None
        self._target_market_id = self._market_id_for_symbol(symbol)

    def get_stream_url(self, msg_type: MsgType) -> str:
        """Return the Lighter WebSocket URL for a message type.

        Args:
            msg_type: Message type for the stream.

        Returns:
            str: WebSocket URL.
        """
        match msg_type:
            case MsgType.ORDERBOOK | MsgType.TRADE:
                return "wss://mainnet.zklighter.elliot.ai/stream"
            case _:
                return ""

    def get_subscribe_messages(self, msg_type: MsgType) -> list[bytes]:
        """Return subscription messages for the requested stream.

        Args:
            msg_type: Message type for the stream.

        Returns:
            list[bytes]: Subscription payloads.
        """
        market_index = self._market_index()
        match msg_type:
            case MsgType.ORDERBOOK:
                channels = self._build_channels("order_book", market_index)
                payloads: list[dict[str, Any]] = [
                    {"type": "subscribe", "channel": channel} for channel in channels
                ]
                payloads.append(
                    {"type": "subscribe", "channel": "spot_market_stats/all"}
                )
                try:
                    market_index_value: int | str = int(market_index)
                except (TypeError, ValueError):
                    market_index_value = market_index
                payloads.append(
                    {
                        "type": "subscribe",
                        "channel": "order_book",
                        "market_index": market_index_value,
                    }
                )
                payloads.append(
                    {
                        "type": "subscribe",
                        "channel": "orderbook",
                        "market_index": market_index_value,
                    }
                )
                encoded: list[bytes] = []
                seen: set[bytes] = set()
                for payload in payloads:
                    data = msgspec.json.encode(payload)
                    if data in seen:
                        continue
                    seen.add(data)
                    encoded.append(data)
                return encoded
            case MsgType.TRADE:
                channels = self._build_channels("trade", market_index)
                payloads = [{"type": "subscribe", "channel": channel} for channel in channels]
                try:
                    market_index_value = int(market_index)
                except (TypeError, ValueError):
                    market_index_value = market_index
                payloads.append(
                    {
                        "type": "subscribe",
                        "channel": "trade",
                        "market_index": market_index_value,
                    }
                )
                encoded = []
                seen = set()
                for payload in payloads:
                    data = msgspec.json.encode(payload)
                    if data in seen:
                        continue
                    seen.add(data)
                    encoded.append(data)
                return encoded
            case _:
                return []

    def parse_bbo(self, msg: bytes) -> BBOUpdate | None:
        """Parse a Lighter order book update into a BBO update.

        Args:
            msg: Raw WebSocket message.

        Returns:
            BBOUpdate | None: Normalized BBO update.
        """
        payload = self._decode_payload(msg)
        if payload is None:
            return None
        book = payload.get("order_book")
        if not isinstance(book, dict):
            return None
        bids = self._parse_levels(book.get("bids"))
        asks = self._parse_levels(book.get("asks"))
        if not bids or not asks:
            return None
        best_bid = max(bids, key=lambda x: x.price)
        best_ask = min(asks, key=lambda x: x.price)
        now_ms = time_ms()
        venue_ts = self._normalize_timestamp(payload.get("timestamp")) or now_ms
        return BBOUpdate(
            venue=self.venue,
            symbol=self.symbol,
            venue_time_ms=venue_ts,
            local_time_ms=now_ms,
            bid_price=best_bid.price,
            bid_size=best_bid.size,
            ask_price=best_ask.price,
            ask_size=best_ask.size,
        )

    def parse_trade(self, msg: bytes) -> TradeMsg | None:
        """Parse a Lighter trade message into normalized trades.

        Args:
            msg: Raw WebSocket message.

        Returns:
            TradeMsg | None: Normalized trade message.
        """
        payload = self._decode_payload(msg)
        if payload is None:
            return None
        if isinstance(payload.get("spot_market_stats"), dict):
            self._ingest_market_stats(payload)
            return None
        channel = payload.get("channel")
        match channel:
            case str() as channel if (
                channel == "trade"
                or channel.startswith("trade:")
                or channel.startswith("trade/")
            ):
                normalized = self._normalize_channel(channel)
                if ":" in normalized:
                    if self._trade_channel is None:
                        self._trade_channel = normalized
                    elif normalized != self._trade_channel:
                        return None
            case str():
                return None
            case _:
                match payload.get("type"):
                    case str() as msg_type if "trade" in msg_type.lower():
                        pass
                    case str():
                        return None
                    case _:
                        pass
        trades_raw = payload.get("trades")
        if not isinstance(trades_raw, list):
            data = payload.get("data")
            if isinstance(data, dict) and isinstance(data.get("trades"), list):
                trades_raw = data.get("trades")
            elif isinstance(data, list):
                trades_raw = data
        if not isinstance(trades_raw, list):
            return None
        now_ms = time_ms()
        trades: list[Trade] = []
        for entry in trades_raw:
            if not isinstance(entry, dict):
                continue
            ts_raw = entry.get("timestamp") or entry.get("transaction_time") or entry.get(
                "ts"
            )
            price_raw = entry.get("price")
            size_raw = entry.get("size")
            side_raw = entry.get("type")
            if ts_raw is None or price_raw is None or size_raw is None or side_raw is None:
                continue
            trade_ts = self._normalize_timestamp(ts_raw) or now_ms
            try:
                price = float(price_raw)
                size = float(size_raw)
            except (TypeError, ValueError):
                continue
            if trade_ts <= 0 or price <= 0.0 or size < 0.0:
                continue
            side = str(side_raw).lower()
            trades.append(
                Trade(
                    time_ms=trade_ts,
                    is_buy=side in {"buy", "bid"},
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
        """Parse a Lighter order book message into a snapshot or delta.

        Args:
            msg: Raw WebSocket message.

        Returns:
            OrderbookMsg | None: Normalized orderbook update.
        """
        payload = self._decode_payload(msg)
        if payload is None:
            return None
        if isinstance(payload.get("spot_market_stats"), dict):
            self._ingest_market_stats(payload)
            return None
        channel = payload.get("channel")
        match channel:
            case str() as channel if (
                channel == "order_book"
                or channel.startswith("order_book:")
                or channel.startswith("order_book/")
                or channel.startswith("orderbook:")
                or channel.startswith("orderbook/")
            ):
                normalized = self._normalize_channel(channel)
                if ":" in normalized:
                    if self._orderbook_channel is None:
                        self._orderbook_channel = normalized
                    elif normalized != self._orderbook_channel:
                        return None
            case str():
                return None
            case _:
                match payload.get("type"):
                    case str() as msg_type if (
                        "order_book" in msg_type.lower() or "orderbook" in msg_type.lower()
                    ):
                        pass
                    case str():
                        return None
                    case _:
                        pass
        book = payload.get("order_book")
        if not isinstance(book, dict):
            data = payload.get("data")
            if isinstance(data, dict):
                nested = data.get("order_book")
                if isinstance(nested, dict):
                    book = nested
                else:
                    book = data
        if not isinstance(book, dict):
            return None
        bids = self._parse_levels(book.get("bids") or book.get("b"))
        asks = self._parse_levels(book.get("asks") or book.get("a"))
        if not bids and not asks:
            return None
        now_ms = time_ms()
        venue_ts = (
            self._normalize_timestamp(
                payload.get("timestamp")
                or payload.get("ts")
                or book.get("timestamp")
                or book.get("ts")
            )
            or now_ms
        )
        explicit_snapshot = False
        msg_type = payload.get("type")
        if isinstance(msg_type, str):
            msg_type_lower = msg_type.lower()
            if "snapshot" in msg_type_lower:
                explicit_snapshot = True
            elif "update" in msg_type_lower or "delta" in msg_type_lower:
                explicit_snapshot = False
        if not self._orderbook_initialized:
            if not bids or not asks:
                return None
            is_snapshot = True
        else:
            is_snapshot = explicit_snapshot
        if is_snapshot:
            self._orderbook_initialized = True
            return OrderbookMsg(
                venue=self.venue,
                symbol=self.symbol,
                venue_time_ms=venue_ts,
                local_time_ms=now_ms,
                bids=bids,
                asks=asks,
                is_bbo=False,
                is_snapshot=True,
            )
        if not self._orderbook_initialized:
            return None
        return OrderbookMsg(
            venue=self.venue,
            symbol=self.symbol,
            venue_time_ms=venue_ts,
            local_time_ms=now_ms,
            bids=bids,
            asks=asks,
            is_bbo=False,
            is_snapshot=False,
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
            match entry:
                case {"price": price_raw, "size": size_raw}:
                    pass
                case [price_raw, size_raw, *_] | (price_raw, size_raw, *_):
                    pass
                case _:
                    continue
            try:
                price = float(price_raw)
                size = float(size_raw)
            except (TypeError, ValueError):
                continue
            parsed.append(OrderbookLevel(price=price, size=size))
        return parsed

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

    def _market_index(self) -> str:
        """Return the market index for the configured symbol.

        Returns:
            str: Market index string.
        """
        if self.symbol.isdigit():
            return self.symbol
        return self.symbol if self.symbol.isdigit() else "0"

    def _build_channels(self, prefix: str, market_index: str) -> list[str]:
        """Build candidate channels for subscriptions.

        Args:
            prefix: Channel prefix (order_book or trade).
            market_index: Market index string.

        Returns:
            list[str]: Channel identifiers to subscribe to.
        """
        prefixes = {prefix}
        if prefix == "order_book":
            prefixes.add("orderbook")
        channels = [f"{name}/{market_index}" for name in prefixes]
        if not self.symbol.isdigit():
            symbol_variants = {
                self.symbol,
                self.symbol.replace("-", ""),
                self.symbol.replace("-", "/"),
            }
            for variant in symbol_variants:
                if not variant:
                    continue
                channels.extend(f"{name}/{variant}" for name in prefixes)
        return list(dict.fromkeys(channels))

    def _market_id_for_symbol(self, symbol: str) -> int | None:
        """Return the market id for a known symbol.

        Args:
            symbol: Market symbol.

        Returns:
            int | None: Market id when known.
        """
        mapping = {
            "BTC-USDC": 101,
            "BTC-USD": 101,
        }
        return mapping.get(symbol)

    def _resolve_market_index_from_stats(
        self, stats: dict[str, Any]
    ) -> str | None:
        """Resolve the market index from spot market stats.

        Args:
            stats: Spot market stats payload.

        Returns:
            str | None: Market index when resolved.
        """
        if self._target_market_id is not None:
            for index, entry in stats.items():
                if not isinstance(entry, dict):
                    continue
                market_id = entry.get("market_id")
                try:
                    market_id_value = int(market_id)
                except (TypeError, ValueError):
                    continue
                if market_id_value == self._target_market_id:
                    return str(index)
        if self.symbol.upper().startswith("BTC"):
            best_price: float | None = None
            best_index: str | None = None
            for index, entry in stats.items():
                if not isinstance(entry, dict):
                    continue
                price_raw = entry.get("mid_price") or entry.get("last_trade_price")
                try:
                    price = float(price_raw)
                except (TypeError, ValueError):
                    continue
                if best_price is None or price > best_price:
                    best_price = price
                    best_index = str(index)
            return best_index
        return None

    def _ingest_market_stats(self, payload: dict[str, Any]) -> None:
        """Update the resolved market index from spot market stats.

        Args:
            payload: Spot market stats payload.
        """
        stats = payload.get("spot_market_stats")
        if not isinstance(stats, dict):
            return
        target_index = self._resolve_market_index_from_stats(stats)
        if not target_index or target_index == self._resolved_market_index:
            return
        self._resolved_market_index = target_index
        self._orderbook_channel = f"order_book:{target_index}"
        self._trade_channel = f"trade:{target_index}"
        self._orderbook_initialized = False
        self._subscribe_market_index(target_index)

    def _subscribe_market_index(self, market_index: str) -> None:
        """Subscribe to order book and trade channels for a market index.

        Args:
            market_index: Market index string.
        """
        orderbook_msg = msgspec.json.encode(
            {"type": "subscribe", "channel": f"order_book/{market_index}"}
        )
        trade_msg = msgspec.json.encode(
            {"type": "subscribe", "channel": f"trade/{market_index}"}
        )
        if self._orderbook_pool is not None:
            try:
                self._orderbook_pool.send_data(orderbook_msg, only_fastest=False)
            except RuntimeError:
                pass
        if self._trade_pool is not None:
            try:
                self._trade_pool.send_data(trade_msg, only_fastest=False)
            except RuntimeError:
                pass

    def _normalize_channel(self, channel: str) -> str:
        """Normalize channel strings for comparisons.

        Args:
            channel: Raw channel string from the payload.

        Returns:
            str: Normalized channel identifier.
        """
        if "/" in channel and ":" not in channel:
            prefix, suffix = channel.split("/", 1)
            return f"{prefix}:{suffix}"
        return channel
