"""Tick-count candle aggregator.

Creates a new candle after a fixed number of trades (ticks).
"""

from mm_toolbox.candles.base cimport BaseCandles
from libc.math cimport fmax, fmin

cdef class TickCandles(BaseCandles):
    """Candle aggregator triggered by a fixed number of trades.

    A new candle is created once ``ticks_per_bucket`` trades have been
    accumulated.

    Attributes:
        ticks_per_bucket (int): Number of trades required to close a candle.
    """
    def __init__(self, int ticks_per_bucket, int num_candles=1000, bint store_trades=True):
        """Initialize the tick-based candle aggregator.

        Args:
            ticks_per_bucket (int): Number of trades per candle (must be > 0).
            num_candles (int): Ring buffer capacity for closed candles.
            store_trades (bool): Whether to retain per-trade records.

        Raises:
            ValueError: If ticks_per_bucket is not positive.
        """
        if ticks_per_bucket <= 0:
            raise ValueError(f"Invalid ticks_per_bucket; expected >0 but got {ticks_per_bucket}")
        BaseCandles.__init__(self, num_candles, store_trades)
        self.ticks_per_bucket = ticks_per_bucket

    cpdef void process_trade(self, object trade):
        """Process a single trade tick.

        Args:
            trade (Trade): The trade to ingest.
        """
        cdef:
            double time_ms = trade.time_ms
            bint is_buy = trade.is_buy
            double price = trade.price
            double size = trade.size
            double volume = price * size

        if self.is_stale_trade(time_ms):
            return

        # Initialize a new candle if this is the first trade
        if self.latest_candle.num_trades == 0:
            self.latest_candle.open_time_ms = time_ms
            self.latest_candle.open_price = price
            self.latest_candle.high_price = price
            self.latest_candle.low_price = price

        # Update candle statistics
        self.latest_candle.high_price = fmax(self.latest_candle.high_price, price)
        self.latest_candle.low_price = fmin(self.latest_candle.low_price, price)
        self.latest_candle.close_price = price

        # Update volume based on trade direction
        if is_buy:
            self.latest_candle.buy_size += size
            self.latest_candle.buy_volume += volume
        else:
            self.latest_candle.sell_size += size
            self.latest_candle.sell_volume += volume

        self.latest_candle.vwap = self.calculate_vwap(price, size)
        if self._store_trades:
            self.latest_candle.trades.append(trade)
        self.latest_candle.num_trades += 1
        self.latest_candle.close_time_ms = time_ms

        # Check if max ticks has been reached
        if self.latest_candle.num_trades >= self.ticks_per_bucket:
            self.insert_and_reset_candle()
