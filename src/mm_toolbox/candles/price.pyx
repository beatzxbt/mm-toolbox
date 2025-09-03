from mm_toolbox.candles.base cimport BaseCandles
from libc.math cimport fmax, fmin

cdef class PriceCandles(BaseCandles):
    """
    Candle aggregator that creates new candles based on price movement.
    
    A new candle is created when the price moves by a specified amount from the opening price.
    """
    def __init__(self, double price_bucket, int num_candles=1000):
        """Initialize the price-based candle aggregator."""
        
        self.price_bucket = price_bucket
        self.upper_price_bound = 0.0
        self.lower_price_bound = 0.0

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

        if self.latest_candle.num_trades == 0:
            self.latest_candle.open_time_ms = time_ms
            self.latest_candle.open_price = price

            self.upper_price_bound = price + self.price_bucket
            self.lower_price_bound = price - self.price_bucket

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

        if price > self.upper_price_bound or price < self.lower_price_bound:
            self.insert_and_reset_candle()
