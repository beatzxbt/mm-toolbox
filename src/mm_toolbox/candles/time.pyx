from libc.math cimport fmax, fmin
from mm_toolbox.time.time cimport time_ms
from mm_toolbox.candles.base cimport BaseCandles

cdef class TimeCandles(BaseCandles):
    """Candle aggregator that creates new candles based on a fixed time interval."""
    
    def __init__(self, double secs_per_bucket, int num_candles=1000):
        """Initialize the time-based candle aggregator."""
        self.millis_per_bucket = secs_per_bucket * 1000.0
        self.next_candle_close_time = time_ms() + self.millis_per_bucket

    cpdef void process_trade(self, object trade):
        """Process a single trade tick, updating the current candle."""
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

        # Check if time interval has elapsed
        if self.next_candle_close_time <= time_ms:
            self.insert_and_reset_candle()
            # Calculate the next candle close time as a multiple of millis_per_bucket
            self.next_candle_close_time = (
                (time_ms // self.millis_per_bucket) + 1
            ) * self.millis_per_bucket
            self.process_trade(trade)
            return

        self.latest_candle.high_price = fmax(self.latest_candle.high_price, price)
        self.latest_candle.low_price = fmin(self.latest_candle.low_price, price)
        self.latest_candle.close_price = price

        if is_buy:
            self.latest_candle.buy_size += size
            self.latest_candle.buy_volume += volume
        else:
            self.latest_candle.sell_size += size
            self.latest_candle.sell_volume += volume

        self.latest_candle.vwap_price = self.calculate_vwap(volume, size)
        self.latest_candle.trades.append(trade)
        self.latest_candle.num_trades += 1
        self.latest_candle.close_time_ms = time_ms
