"""Base candle aggregation types and shared logic.

Provides:
- Trade: immutable trade record (msgspec Struct).
- Candle: mutable candle state with OHLC, volume, VWAP, and per-trade storage.
- BaseCandles: abstract cdef base class for all candle aggregators, including
  ringbuffer management, VWAP tracking, stale-trade filtering, and async
  iteration support.
"""

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

import asyncio
from abc import ABC, abstractmethod
from typing import final, AsyncIterator, Self
from msgspec import Struct

from libc.stdint cimport uint64_t as u64

from mm_toolbox.ringbuffer.generic cimport GenericRingBuffer

class Trade(Struct, frozen=True):
    """Immutable trade record.

    Attributes:
        time_ms (int): Timestamp in milliseconds.
        is_buy (bool): True if the trade is a buy, False for a sell.
        price (float): Trade price.
        size (float): Trade size (quantity).
    """
    time_ms: int
    is_buy: bool
    price: float
    size: float

    @property
    def value(self) -> float:
        """The monetary value of the trade (price * size)."""
        return self.price * self.size

class Candle(Struct):
    """Mutable candle state capturing OHLC, volume, and trade details.

    Attributes:
        open_time_ms (int): Candle open timestamp in milliseconds.
        close_time_ms (int): Candle close timestamp in milliseconds.
        open_price (float): Opening price.
        high_price (float): Highest price seen.
        low_price (float): Lowest price seen.
        close_price (float): Closing price.
        buy_size (float): Total size of buy trades.
        buy_volume (float): Total volume of buy trades.
        sell_size (float): Total size of sell trades.
        sell_volume (float): Total volume of sell trades.
        vwap (float): Volume-weighted average price.
        num_trades (int): Number of trades in the candle.
        trades (list[Trade]): Per-trade records (if storage enabled).
    """
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
        self.vwap = 0.0
        self.num_trades = 0
        self.trades.clear()

    def copy(self, include_trades: bool = True) -> Self:
        """Create a shallow copy of the candle.

        Args:
            include_trades (bool): If True, include the trades list in the copy.

        Returns:
            Self: A new Candle instance with the same field values.
        """
        return type(self)(
            open_time_ms=self.open_time_ms,
            close_time_ms=self.close_time_ms,
            open_price=self.open_price,
            high_price=self.high_price,
            low_price=self.low_price,
            close_price=self.close_price,
            buy_size=self.buy_size,
            buy_volume=self.buy_volume,
            sell_size=self.sell_size,
            sell_volume=self.sell_volume,
            vwap=self.vwap,
            num_trades=self.num_trades,
            trades=list(self.trades) if include_trades else [],
        )

    @classmethod
    def empty(cls) -> Self:
        """Create an empty candle.

        Returns:
            Self: A new Candle with all fields zeroed and an empty trades list.
        """
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
            vwap=0.0,
            num_trades=0,
            trades=[]
        )


cdef class BaseCandles:
    """Base class for candle aggregators.

    Summarizes individual trades into Candle buckets stored in a ring buffer.
    Supports async iteration so consumers can await new closed candles.

    Attributes:
        latest_candle (Candle): The candle currently being built.
        candle_push_event (object): Async event/future list for new candles.
    """

    def __cinit__(self):
        """Lightweight C-level construction."""
        self.latest_candle = Candle.empty()
        self.candle_push_event = None

        # Reserved for VWAP, do not modify
        self.__cum_volume = 0.0
        self.__total_size = 0.0

    def __init__(self, u64 num_candles=1000, bint store_trades=True):
        """Initialize the ring buffer and trade storage setting.

        Args:
            num_candles (int): Maximum number of closed candles to retain.
            store_trades (bool): Whether to keep per-trade records in each candle.

        Raises:
            ValueError: If num_candles is not positive.
        """
        if num_candles <= 0:
            raise ValueError(f"Invalid number of candles; expected >1 but got {num_candles}")
        self._ringbuffer = GenericRingBuffer(max_capacity=num_candles)
        self._store_trades = store_trades

    cdef inline double calculate_vwap(self, double price, double size) noexcept nogil:
        """Calculate the running VWAP.

        Args:
            price (double): Trade price.
            size (double): Trade size.

        Returns:
            double: Current volume-weighted average price.
        """
        self.__cum_volume += price * size
        self.__total_size += size
        if self.__total_size > 0.0:
            return self.__cum_volume / self.__total_size
        return 0.0

    cdef inline bint is_stale_trade(self, double time_ms):
        """Check if a trade is older than the latest candle close time.

        Args:
            time_ms (double): Trade timestamp in milliseconds.

        Returns:
            bool: True if the trade is stale and should be skipped.
        """
        return time_ms < self.latest_candle.close_time_ms

    cdef inline void insert_and_reset_candle(self):
        """Close the current candle, store it, and reset state.

        Copies ``latest_candle`` into the ring buffer, resolves any pending
        async futures, and resets internal VWAP accumulators.
        """
        cdef object closed_candle = self.latest_candle.copy(
            include_trades=self._store_trades
        )
        self._ringbuffer.insert(closed_candle)

        cdef object event = self.candle_push_event
        cdef object future
        if isinstance(event, list):
            for future in event:
                if isinstance(future, asyncio.Future) and not future.done():
                    future.set_result(closed_candle)
            self.candle_push_event = closed_candle
        elif isinstance(event, asyncio.Future):
            if not event.done():
                event.set_result(closed_candle)
            self.candle_push_event = closed_candle
        else:
            self.candle_push_event = closed_candle
        
        self.latest_candle.reset()
        self.__cum_volume = 0.0
        self.__total_size = 0.0

    cpdef void initialize(self, list[object] trades):
        """Bootstrap candle state from a list of historical trades.

        Args:
            trades (list[Trade]): Ordered list of trades to replay.

        Raises:
            ValueError: If trades is empty or contains non-Trade objects.
        """
        if not trades:
            raise ValueError("Cannot initialize with an empty trade list.")
        cdef object t
        for t in trades:
            if not isinstance(t, Trade):
                raise ValueError(f"Invalid object typing in list; expected list[Trade] but got {type(trades)}")

        self.latest_candle.reset()
        self._ringbuffer.clear()
        self.__cum_volume = 0.0
        self.__total_size = 0.0

        for trade in trades:
            self.process_trade(trade)

    cpdef void process_trade(self, object trade):
        """Process a single trade tick.

        Subclasses must override this to implement their specific trigger logic.

        Args:
            trade (Trade): The trade to ingest.

        Raises:
            NotImplementedError: Always; must be overridden.
        """
        raise NotImplementedError("Subclasses must implement this method;")

    def __len__(self):
        """Return the number of closed candles stored.

        Returns:
            int: Count of candles in the ring buffer.
        """
        return len(self._ringbuffer)

    def __getitem__(self, index: int) -> Candle:
        """Access a closed candle by index.

        Args:
            index (int): Candle index (supports negative indexing).

        Returns:
            Candle: The requested candle.

        Raises:
            IndexError: If index is out of range.
        """
        candles = self._ringbuffer.unwrapped()
        if index < 0:
            index += len(candles)
        if index < 0 or index >= len(candles):
            raise IndexError(f"Index out of range; expected within (0 <> {len(candles)}) but got {index}")
        return candles[index]

    def __aiter__(self) -> AsyncIterator[Candle]:
        """Return self as an async iterator.

        Returns:
            AsyncIterator[Candle]: Iterator over closed candles.
        """
        return self

    async def __anext__(self) -> Candle:
        """Await the next closed candle.

        Supports multiple concurrent consumers.  If a candle is already
        pending it is returned immediately; otherwise a future is created
        and awaited.

        Returns:
            Candle: The next closed candle.
        """
        cdef object event = self.candle_push_event
        cdef object future
        cdef object new_candle

        if isinstance(event, Candle):
            self.candle_push_event = None
            return event

        future = asyncio.get_running_loop().create_future()
        if event is None:
            self.candle_push_event = [future]
        elif isinstance(event, list):
            event.append(future)
        elif isinstance(event, asyncio.Future):
            # Convert single pending future to a list
            self.candle_push_event = [event, future]
        else:
            self.candle_push_event = [future]

        await future
        new_candle = future.result()
        return new_candle
