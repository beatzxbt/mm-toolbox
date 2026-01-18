"""Wire-format models for Bybit v5 public WebSocket payloads.

Used by the Bybit stream processor to decode orderbook and trade topics
into typed msgspec structures.
"""

from __future__ import annotations

import msgspec


class BybitOrderbookData(msgspec.Struct):
    """Orderbook data payload for Bybit orderbook topics.

    Attributes:
        s: Symbol.
        b: Bid updates as [price, size] strings.
        a: Ask updates as [price, size] strings.
        u: Update ID when provided.
        seq: Sequence ID when provided.
    """

    s: str
    b: list[list[str]]
    a: list[list[str]]
    u: int | None = None
    seq: int | None = None


class BybitOrderbookMsg(msgspec.Struct):
    """Orderbook message wrapper from Bybit.

    Attributes:
        topic: Topic string.
        type: Message type (snapshot/delta).
        ts: Timestamp in milliseconds.
        data: Orderbook payload.
    """

    topic: str
    type: str
    ts: int
    data: BybitOrderbookData


class BybitTradeEntry(msgspec.Struct):
    """Trade entry payload from Bybit.

    Attributes:
        T: Trade time in milliseconds when present.
        t: Alternate trade time field when present.
        p: Trade price.
        v: Trade size.
        S: Side (Buy or Sell).
    """

    T: int | None = None
    t: int | None = None
    p: str = ""
    v: str = ""
    S: str = ""


class BybitTradeMsg(msgspec.Struct):
    """Trade message wrapper from Bybit.

    Attributes:
        topic: Topic string.
        type: Message type.
        ts: Timestamp in milliseconds.
        data: Trade entries.
    """

    topic: str
    type: str
    ts: int
    data: list[BybitTradeEntry]
