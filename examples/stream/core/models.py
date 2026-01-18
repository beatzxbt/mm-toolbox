"""Normalized message models for multi-venue streaming.

Defines standardized structures for BBO, trades, orderbook updates, and
an IPC envelope used by stream processors and consumers.
"""

from __future__ import annotations

from enum import IntEnum, StrEnum

import msgspec


class Venue(StrEnum):
    """Supported venues for the streaming example."""

    BINANCE = "binance"
    BYBIT = "bybit"
    OKX = "okx"
    HYPERLIQUID = "hyperliquid"
    LIGHTER = "lighter"


class MsgType(IntEnum):
    """Enum describing the normalized message types.

    Attributes:
        BBO: Best bid/offer update.
        TRADE: Trade update.
        ORDERBOOK: Orderbook update (snapshot or delta).
    """

    BBO = 1
    TRADE = 2
    ORDERBOOK = 3


class CoreMsg(msgspec.Struct):
    """Common fields for normalized messages.

    Attributes:
        venue: Venue name.
        symbol: Venue symbol.
        venue_time_ms: Timestamp from the venue in milliseconds.
        local_time_ms: Timestamp when received locally in milliseconds.
    """

    venue: Venue
    symbol: str
    venue_time_ms: int
    local_time_ms: int


class BBOUpdate(CoreMsg, tag="bbo"):
    """Normalized best bid/offer update.

    Attributes:
        bid_price: Best bid price.
        bid_size: Best bid size.
        ask_price: Best ask price.
        ask_size: Best ask size.
    """

    bid_price: float
    bid_size: float
    ask_price: float
    ask_size: float


class Trade(msgspec.Struct):
    """Represents a trade that occurred on the exchange."""

    time_ms: int
    price: float
    is_buy: bool
    size: float

    def __post_init__(self) -> None:
        """Validate trade values."""
        if self.time_ms <= 0:
            raise ValueError(f"Invalid time_ms; expected >0 but got {self.time_ms}")
        if self.price <= 0.0:
            raise ValueError(f"Invalid price; expected >0 but got {self.price}")
        if self.size < 0.0:
            raise ValueError(f"Invalid size; expected >=0 but got {self.size}")

    @property
    def value(self) -> float:
        """Returns the value of the trade."""
        return self.price * self.size


class TradeMsg(CoreMsg, tag=True):
    """Represents a trade that occurred on the exchange."""

    trades: list[Trade]

    def __post_init__(self) -> None:
        """Sort trades by time for consistent processing."""
        self.trades.sort(key=lambda x: x.time_ms)


class OrderbookLevel(msgspec.Struct):
    """Represents a single level in the orderbook."""

    price: float
    size: float
    num_orders: int = 1

    def __post_init__(self) -> None:
        """Validate level values."""
        if self.price <= 0.0:
            raise ValueError(f"Invalid price; expected >0 but got {self.price}")
        if self.size < 0.0:
            raise ValueError(f"Invalid size; expected >=0 but got {self.size}")
        if self.num_orders < 0:
            raise ValueError(
                f"Invalid num_orders; expected >=0 but got {self.num_orders}"
            )

    @property
    def value(self) -> float:
        """Returns the value of the orderbook level."""
        return self.price * self.size


class OrderbookMsg(CoreMsg, tag=True):
    """Represents the current state of the orderbook.

    Attributes:
        bids: Bid levels.
        asks: Ask levels.
        is_bbo: Whether this message only represents the best bid/offer.
        is_snapshot: Whether this message is a full snapshot.
    """

    bids: list[OrderbookLevel]
    asks: list[OrderbookLevel]
    is_bbo: bool
    is_snapshot: bool

    def __post_init__(self) -> None:
        """Sort levels by price for consistent processing."""
        self.bids.sort(key=lambda x: x.price)
        self.asks.sort(key=lambda x: x.price)


class StreamMessage(msgspec.Struct):
    """Envelope for normalized messages sent across IPC.

    Attributes:
        msg_type: Message type enum for routing.
        venue: Venue name.
        data: Normalized payload.
    """

    msg_type: MsgType
    venue: Venue
    data: BBOUpdate | TradeMsg | OrderbookMsg
