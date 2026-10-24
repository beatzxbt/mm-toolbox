"""Time-interval candle aggregator.

Creates a new candle at fixed time intervals (e.g. every N seconds).
"""

from libc.math cimport fmax, fmin, isfinite, nextafter
from mm_toolbox.candles.base cimport BaseCandles

cdef class TimeCandles(BaseCandles):
    """Candle aggregator triggered by fixed time intervals.

    A new candle is created each time the interval ``secs_per_bucket``
    elapses.

    Attributes:
        millis_per_bucket (double): Interval length in milliseconds.
        next_candle_close_time (double): Timestamp when the current candle
            should close.
    """

    def __init__(self, double secs_per_bucket, int num_candles=1000, bint store_trades=True):
        """Initialize the time-based candle aggregator.

        Args:
            secs_per_bucket (double): Candle duration in seconds.
            num_candles (int): Ring buffer capacity for closed candles.
            store_trades (bool): Whether to retain per-trade records.
        """
        if not isfinite(secs_per_bucket) or secs_per_bucket <= 0.0 or not isfinite(secs_per_bucket * 1000.0) or secs_per_bucket * 1000.0 == 0.0:
            raise ValueError("secs_per_bucket must be finite and positive")
        BaseCandles.__init__(self, num_candles, store_trades)
        self.millis_per_bucket = secs_per_bucket * 1000.0
        self.next_candle_close_time = 0.0

    cdef void _reset_aggregation(self):
        """Anchor replay boundaries to its first trade instead of construction time."""
        self.next_candle_close_time = 0.0

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

        self.validate_trade(price, size)
        if self.is_stale_trade(time_ms):
            return

        if self.next_candle_close_time == 0.0:
            self.next_candle_close_time = time_ms + self.millis_per_bucket
        elif self.next_candle_close_time <= time_ms:
            if self.latest_candle.num_trades > 0:
                self.insert_and_reset_candle()
            self.next_candle_close_time += (
                (time_ms - self.next_candle_close_time) // self.millis_per_bucket + 1.0
            ) * self.millis_per_bucket
        if self.next_candle_close_time <= time_ms:
            raise ValueError("Candle interval is below timestamp precision")
        if self.latest_candle.num_trades == 0:
            self.latest_candle.open_time_ms = time_ms
            self.latest_candle.open_price = price
            self.latest_candle.high_price = price
            self.latest_candle.low_price = price

        self.latest_candle.high_price = fmax(self.latest_candle.high_price, price)
        self.latest_candle.low_price = fmin(self.latest_candle.low_price, price)
        self.latest_candle.close_price = price

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
