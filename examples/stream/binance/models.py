"""Wire-format models for Binance futures WebSocket payloads.

Used by the Binance stream processor to decode book ticker, trade, and
depth update messages via msgspec structs.
"""

from __future__ import annotations

import msgspec


class BinanceStreamMsg(msgspec.Struct, tag_field="e"):
    """Base class for tagged Binance stream payloads."""


class BinanceBBOMsg(BinanceStreamMsg, tag="bookTicker"):
    """Book ticker payload from Binance futures streams.

    Attributes:
        s: Symbol.
        u: Order book update ID.
        b: Best bid price.
        B: Best bid quantity.
        a: Best ask price.
        A: Best ask quantity.
        T: Transaction time in milliseconds.
        E: Event time in milliseconds.
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
        E: Event time in milliseconds.
        T: Trade time in milliseconds.
        s: Symbol.
        t: Trade ID.
        p: Trade price.
        q: Trade quantity.
        X: Trade type (e.g., MARKET).
        m: True if buyer is the market maker.
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
        E: Event time in milliseconds.
        T: Transaction time in milliseconds.
        s: Symbol.
        U: First update ID in the event.
        u: Final update ID in the event.
        b: Bid updates as [price, quantity] strings.
        a: Ask updates as [price, quantity] strings.
        pu: Previous update ID when present.
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
