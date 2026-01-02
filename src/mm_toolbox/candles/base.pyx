# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

import copy
import asyncio
from abc import ABC, abstractmethod
from typing import final, AsyncIterator, Self
from msgspec import Struct

from libc.stdint cimport uint64_t as u64

from mm_toolbox.ringbuffer.generic cimport GenericRingBuffer

class Trade(Struct):
    time_ms: int
    is_buy: bool
    price: float
    size: float

    @property
    def value(self) -> float:
        """The value of the trade."""
        return self.price * self.size

class Candle(Struct):
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
    vwap_price: float
    num_trades: int
    trades: list[Trade]

    def reset(self) -> None:
        """Reset the candle to an empty state."""
        self.open_time_ms = 0
        self.close_time_ms = 0
        self.open_price = 0.0
        self.high_price = 0.0
        self.low_price = 0.0
        self.close_price = 0.0
        self.buy_size = 0.0
        self.buy_volume = 0.0
        self.sell_size = 0.0
        self.sell_volume = 0.0
        self.vwap_price = 0.0
        self.num_trades = 0
        self.trades.clear()

    def copy(self) -> Self:
        """Create a copy of the candle."""
        return copy.deepcopy(self)

    @classmethod
    def empty(cls) -> Self:
        """Create an empty candle."""
        return cls(
            open_time_ms=0,
            close_time_ms=0,
            open_price=0.0,
            high_price=0.0,
            low_price=0.0,
            close_price=0.0,
            buy_size=0.0,
            buy_volume=0.0,
            sell_size=0.0,
            sell_volume=0.0,
            vwap_price=0.0,
            num_trades=0,
            trades=[]
        )


cdef class BaseCandles:
    """Summarize individual trades into buckets of information."""

    def __cinit__(self):
        """Lightweight construction; full init happens in __init__."""
        self.latest_candle = Candle.empty()
        self.candle_push_event = asyncio.Future[Candle]()

        # Reserved for VWAP, do not modify
        self.__cum_volume = 0.0
        self.__total_size = 0.0

    def __init__(self, u64 num_candles=1000):
        """Initialize ring buffer capacity and validate settings."""
        if num_candles <= 0:
            raise ValueError(f"Invalid number of candles; expected >1 but got {num_candles}")
        self.ringbuffer = GenericRingBuffer(max_capacity=num_candles)

    cdef inline double calculate_vwap(self, double price, double size) noexcept nogil:
        """Calculate the current VWAP (Volume-Weighted Average Price)."""
        self.__cum_volume += price * size
        self.__total_size += size
        if self.__total_size > 0.0:
            return self.__cum_volume / self.__total_size
        return 0.0

    cdef inline bint is_stale_trade(self, double time_ms):
        """Check if a trade is stale (older than the latest update)."""
        return time_ms < self.latest_candle.close_time_ms

    cdef inline void insert_and_reset_candle(self):
        """Insert the current candle into the ring buffer and reset attributes."""
        self.ringbuffer.insert(self.latest_candle.copy())
        
        # Set result if Future is not already done
        if not self.candle_push_event.done():
            self.candle_push_event.set_result(self.latest_candle.copy())
        
        self.candle_push_event = asyncio.Future[Candle]()
        
        self.latest_candle.reset()

        # This ensures the newest candle is always the latest one incase the 
        # ringbuffer is accessed whilst there is an open candle.
        self.ringbuffer.overwrite_latest(self.latest_candle, increment_count=False)

    cpdef void initialize(self, list[object] trades):
        """Initialize candle data from a batch of existing trades."""
        # The type must strictly be list[Trade], however Cython doesn't support
        # typed lists containing specific Python objects. Check that first and/or
        # last values are of the Trade type, and call it a day. 
        if not isinstance(trades[0], Trade) or not isinstance(trades[-1], Trade):
            raise ValueError(f"Invalid object typing in list; expected list[Trade] but got {type(trades)}")

        self.latest_candle.reset()
        self.ringbuffer.clear()

        for trade in trades:
            self.process_trade(trade)

    cpdef void process_trade(self, object trade):
        """Process a single trade tick, updating the current candle."""
        raise NotImplementedError("Subclasses must implement this method;")

    def __len__(self):
        """Number of candles currently stored."""
        return len(self.ringbuffer)

    def __getitem__(self, index: int) -> Candle:
        """Access a specific candle by index."""
        return self.ringbuffer[index]

    def __aiter__(self) -> AsyncIterator[Candle]:
        """Async iterator over the candles."""
        return self

    async def __anext__(self) -> Candle:
        """Async next candle."""
        await self.candle_push_event
        new_candle = self.candle_push_event.result()
        self.candle_push_event.set_result(None)
        return new_candle