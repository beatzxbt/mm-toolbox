from __future__ import annotations

from collections.abc import Iterator

from mm_toolbox.ringbuffer.generic import GenericRingBuffer

class Trade:
    time_ms: int
    is_buy: bool
    price: float
    size: float

class Candle:
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

    def reset(self) -> None: ...
    def copy(self) -> Candle: ...
    @classmethod
    def empty(cls) -> Candle: ...

class BaseCandles:
    """Summarize individual trades into buckets of information."""

    latest_candle: Candle
    ringbuffer: GenericRingBuffer

    def __init__(self, num_candles: int = 1000) -> None: ...
    def initialize(self, trades: list[Trade]) -> None: ...
    def process_trade(self, trade: Trade) -> None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> Candle: ...
    def __iter__(self) -> Iterator[Candle]: ...
