from __future__ import annotations

from collections.abc import Iterator
from typing import Self, SupportsIndex

from msgspec import Struct

from mm_toolbox.ringbuffer.generic import GenericRingBuffer

class Trade(Struct, frozen=True):
    """Immutable trade record; numeric validity is checked on ingestion."""

    time_ms: int
    is_buy: bool
    price: float
    size: float

    def __init__(
        self, time_ms: int, is_buy: bool, price: float, size: float
    ) -> None: ...
    @property
    def value(self) -> float: ...

class Candle(Struct):
    """Mutable OHLC, volume and VWAP state, optionally retaining trades."""

    open_time_ms: int
    close_time_ms: int
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    buy_size: float
    buy_volume: float
    sell_size: float
    sell_volume: float
    vwap: float
    num_trades: int
    trades: list[Trade]

    def __init__(
        self,
        open_time_ms: int,
        close_time_ms: int,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
        buy_size: float,
        buy_volume: float,
        sell_size: float,
        sell_volume: float,
        vwap: float,
        num_trades: int,
        trades: list[Trade],
    ) -> None: ...
    def reset(self) -> None: ...
    def copy(self, include_trades: bool = True) -> Self: ...
    @classmethod
    def empty(cls) -> Self: ...

class BaseCandles:
    """Summarize individual trades into buckets of information."""

    latest_candle: Candle
    _ringbuffer: GenericRingBuffer[Candle]

    def __init__(self, num_candles: int = 1000, store_trades: bool = True) -> None: ...
    def initialize(self, trades: list[Trade]) -> None: ...
    def process_trade(self, trade: Trade) -> None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: SupportsIndex) -> Candle: ...
    def __iter__(self) -> Iterator[Candle]: ...
    def __aiter__(self) -> Self: ...
    async def __anext__(self) -> Candle: ...
