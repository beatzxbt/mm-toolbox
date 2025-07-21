from mm_toolbox.candles.base cimport BaseCandles

cdef class VolumeCandles(BaseCandles):
    """
    Candle aggregator that creates new candles based on a fixed volume threshold.
    
    A new candle is created when the specified volume threshold is reached.
    """
    def __init__(self, double volume_per_bucket, int num_candles):
        """
        Initialize the volume-based candle aggregator.
        
        Args:
            volume_per_bucket (double): Volume threshold per candle.
            num_candles (int): Maximum number of candles to store.
        """
        super().__init__(num_candles)
        self.volume_per_bucket = volume_per_bucket

    cpdef void process_trade(self, double time_ms, bint is_buy, double px, double sz):
        """
        Process a single trade tick, updating the current candle.
        
        Creates a new candle when the volume threshold is reached.
        
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

        # Update candle statistics
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

        cdef:
            double remaining_volume
            double total_volume = self.buy_sz + self.sell_sz

        # Check if volume threshold has been reached
        if total_volume >= self.volume_per_bucket:
            remaining_volume = total_volume - self.volume_per_bucket

            if is_buy:
                self.buy_sz -= remaining_volume
            else:
                self.sell_sz -= remaining_volume

            self.insert_candle()
            self.process_trade(time_ms, is_buy, px, remaining_volume)
