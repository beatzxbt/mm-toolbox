from mm_toolbox.candles.base import Trade
from mm_toolbox.candles.base cimport BaseCandles
from libc.math cimport fmax, fmin

cdef class MultiCandles(BaseCandles):
    """
    Candle aggregator that creates new candles based on multiple trigger conditions.
    
    A new candle is created when any of these conditions are met:
    - Maximum duration is reached
    - Maximum number of ticks (trades) is reached
    - Maximum volume is reached
    """
    def __init__(self, double max_duration_secs, int max_ticks, double max_size, int num_candles=1000):
        """Initialize the multi-trigger candle aggregator."""
        self.max_duration_millis = max_duration_secs * 1000.0
        self.max_ticks = max_ticks
        self.max_size = max_size

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

        # Check if max duration has been exceeded
        if self.latest_candle.open_time_ms + self.max_duration_millis <= time_ms:
            self.insert_and_reset_candle()
            self.process_trade(trade)
            return

        # Update HLC statistics
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

        # Check if max ticks has been exceeded
        if self.latest_candle.num_trades >= self.max_ticks:
            self.insert_and_reset_candle()
            return

        cdef:
            double remaining_size
            double total_size = self.latest_candle.buy_size + self.latest_candle.sell_size
    
        if total_size > self.max_size:
            remaining_size = total_size - self.max_size + size

            if is_buy:
                self.latest_candle.buy_size -= remaining_size
            else:
                self.latest_candle.sell_size -= remaining_size

            self.insert_and_reset_candle()
            self.process_trade(trade)
            return
