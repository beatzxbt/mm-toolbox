from mm_toolbox.time.time cimport time_ms
from mm_toolbox.candles.base cimport BaseCandles

cdef class TimeCandles(BaseCandles):
    """
    Candle aggregator that creates new candles based on a fixed time interval.
    
    A new candle is created when the specified time interval has elapsed.
    """
    
    def __init__(self, double secs_per_bucket, int num_candles):
        """
        Initialize the time-based candle aggregator.
        
        Args:
            secs_per_bucket (double): Time interval per candle in seconds.
            num_candles (int): Maximum number of candles to store.
        """
        super().__init__(num_candles)
        self.millis_per_bucket = secs_per_bucket * 1000.0
        self.next_candle_close_time = time_ms() + self.millis_per_bucket

    cpdef void process_trade(self, double time_ms, bint is_buy, double px, double sz):
        """
        Process a single trade tick, updating the current candle.
        
        Creates a new candle when the time interval has elapsed.
        
        Args:
            time_ms (double): The timestamp of the trade in milliseconds.
            is_buy (bint): True if it's a buy trade, False if it's a sell trade.
            px (double): The trade price.
            sz (double): The trade size (volume).
        """
        if self.is_stale_trade(time_ms):
            return

        # Initialize a new candle if this is the first trade
        if self.num_trades == 0.0:
            self.open_time_ms = time_ms
            self.open_px = px
            self.high_px = px
            self.low_px = px

        # Check if time interval has elapsed
        if self.next_candle_close_time <= time_ms:
            self.insert_candle()
            # Calculate the next candle close time as a multiple of millis_per_bucket
            self.next_candle_close_time = (
                (time_ms // self.millis_per_bucket) + 1
            ) * self.millis_per_bucket
            self.process_trade(time_ms, is_buy, px, sz)
            return

        self.high_px = max(self.high_px, px)
        self.low_px = min(self.low_px, px)
        self.close_px = px

        # Update volume based on trade direction
        if is_buy:
            self.buy_sz += sz
        else:
            self.sell_sz += sz

        self.vwap_px = self.calculate_vwap(px, sz)
        self.num_trades += 1.0
        self.close_time_ms = time_ms
