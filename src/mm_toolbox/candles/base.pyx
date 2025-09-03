# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

import copy
import msgspec
import asyncio
from abc import ABC, abstractmethod
from typing import final, AsyncIterator

from mm_toolbox.ringbuffer.generic cimport GenericRingBuffer

class Trade(msgspec.Struct):
    time_ms: int
    is_buy: bool
    price: float
    size: float

class Candle(msgspec.Struct):
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

    def copy(self) -> "Candle":
        """Create a copy of the candle."""
        return copy.deepcopy(self)

    @classmethod
    def empty(cls) -> "Candle":
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

    def __cinit__(self, *args, **kwargs):
        """Initialize the candle aggregator."""
        # Extract num_candles from various possible argument patterns
        if len(args) == 1:
            num_candles = 1000  # Default when only one specific parameter given
        elif len(args) >= 2:
            num_candles = args[-1]  # Last argument is usually num_candles
        else:
            num_candles = kwargs.get('num_candles', 1000)
            
        if num_candles <= 0:
            raise ValueError(
                f"Invalid number of candles; expected >1 but got {num_candles}"
            )
        
        self.ringbuffer = GenericRingBuffer(max_capacity=num_candles)
        self.latest_candle = Candle.empty()
        self.candle_push_event = asyncio.Future[Candle]()

        # Reserved for VWAP, do not modify
        self.__cum_volume = 0.0
        self.__total_size = 0.0

    cdef inline double calculate_vwap(self, double price, double size):
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
        
        # Create new Future for next candle
        self.candle_push_event = asyncio.Future[Candle]()
        
        self.latest_candle.reset()

        # This ensures the newest candle is always the latest one incase the 
        # ringbuffer is accessed. If the ringbuffer is accessed, the latest candle
        # is the one that is being accessed.
        self.ringbuffer.overwrite_latest(self.latest_candle, increment_count=False)

    cpdef void initialize(self, list trades):
        """Initialize candle data from a batch of existing trades."""
        self.latest_candle.reset()
        self.ringbuffer.clear()

        for trade in trades:
            self.process_trade(trade)

    @abstractmethod
    def process_trade(self, trade: Trade):
        """Process a single trade tick, updating the current candle."""
        pass

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


def _example() -> None:
    """Example usage of BaseCandles and Candle."""
    class SimpleCandles(BaseCandles):
        def process_trade(self, trade: Trade):
            self.latest_candle.trades.append(trade)
            self.latest_candle.num_trades += 1
            self.latest_candle.close_time_ms = trade.time_ms

    trades = [
        Trade(time_ms=1, is_buy=True, price=100.0, size=1.0),
        Trade(time_ms=2, is_buy=False, price=101.0, size=2.0),
    ]
    candles = SimpleCandles(num_candles=10)
    candles.initialize(trades)
    print(f"Number of candles: {len(candles)}")
    print(f"First trade in latest candle: {candles.latest_candle.trades[0]}")

async def _example_async() -> None:
    """Example showing concurrent trade insertion and candle consumption."""
    class SimpleCandles(BaseCandles):
        def process_trade(self, trade: Trade):
            self.latest_candle.trades.append(trade)
            self.latest_candle.num_trades += 1
            self.latest_candle.close_time_ms = trade.time_ms
            self.insert_and_reset_candle()

    trades = [
        Trade(time_ms=10, is_buy=True, price=100.0, size=1.0),
        Trade(time_ms=20, is_buy=False, price=101.0, size=2.0),
        Trade(time_ms=30, is_buy=True, price=102.0, size=1.5),
    ]
    candles = SimpleCandles(num_candles=5)

    async def producer():
        for trade in trades:
            candles.process_trade(trade)
            await asyncio.sleep(0.01)

    async def consumer():
        async for candle in candles:
            if candle is None:
                break
            print(f"Consumed candle with {candle.num_trades} trades, close_time_ms={candle.close_time_ms}")
            if candle.close_time_ms == trades[-1].time_ms:
                break

    await asyncio.gather(producer(), consumer())
