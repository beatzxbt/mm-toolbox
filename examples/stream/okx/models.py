"""Wire-format models for OKX public WebSocket payloads.

Used by the OKX stream processor to decode BBO, trades, and orderbook
messages into typed msgspec structures.
"""

from __future__ import annotations

import msgspec


class OkxArg(msgspec.Struct):
    """Subscription argument payload for OKX channels.

    Attributes:
        channel: Channel name.
        instId: Instrument identifier.
    """

    channel: str
    instId: str


class OkxBboEntry(msgspec.Struct):
    """Best bid/offer entry from OKX.

    Attributes:
        bidPx: Best bid price.
        bidSz: Best bid size.
        askPx: Best ask price.
        askSz: Best ask size.
        ts: Timestamp in milliseconds.
    """

    bidPx: str
    bidSz: str
    askPx: str
    askSz: str
    ts: str


class OkxBboMsg(msgspec.Struct):
    """BBO message wrapper from OKX.

    Attributes:
        arg: Subscription argument.
        data: BBO entries.
    """

    arg: OkxArg
    data: list[OkxBboEntry]


class OkxTradeEntry(msgspec.Struct):
    """Trade entry from OKX.

    Attributes:
        px: Trade price.
        sz: Trade size.
        side: Trade side (buy/sell).
        ts: Timestamp in milliseconds.
    """

    px: str
    sz: str
    side: str
    ts: str


class OkxTradeMsg(msgspec.Struct):
    """Trade message wrapper from OKX.

    Attributes:
        arg: Subscription argument.
        data: Trade entries.
    """

    arg: OkxArg
    data: list[OkxTradeEntry]


class OkxBookEntry(msgspec.Struct):
    """Orderbook entry from OKX books channel.

    Attributes:
        bids: Bid levels.
        asks: Ask levels.
        ts: Timestamp in milliseconds.
        seqId: Sequence ID.
        prevSeqId: Previous sequence ID.
    """

    bids: list[list[str]]
    asks: list[list[str]]
    ts: str
    seqId: str | None = None
    prevSeqId: str | None = None


class OkxBookMsg(msgspec.Struct):
    """Orderbook message wrapper from OKX.

    Attributes:
        arg: Subscription argument.
        action: Snapshot or update indicator.
        data: Orderbook entries.
    """

    arg: OkxArg
    action: str
    data: list[OkxBookEntry]
