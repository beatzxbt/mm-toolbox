import numpy as np
cimport numpy as cnp

from libc.math cimport INFINITY
from libc.stdint cimport (
    uint32_t as u32,
    int64_t as i64
)

from mm_toolbox.ringbuffer.twodim cimport RingBufferTwoDim

cdef class BaseCandles:
    """
    Aggregates trades into fixed-size candle buckets.

    Format:
      Candle:
        [0] = Open Price
        [1] = High Price
        [2] = Low Price
        [3] = Close Price
        [4] = Buy Volume
        [5] = Sell Volume
        [6] = VWAP Price
        [7] = Total Trades
        [8] = Open Time
        [9] = Close Time
    """

    def __init__(self, int num_candles=1000) -> None:
        """
        Initialize the candle aggregator.

        Args:
            num_candles (int): The maximum number of candles to store before overwriting.

        Raises:
            ValueError: If `num_candles` < 1.
        """
        if num_candles <= 0:
            raise ValueError(
                f"Invalid number of candles; expected >1 but got {num_candles}"
            )
        
        self.open_time_ms = 0.0
        self.open_px = 0.0
        self.high_px = <double>(-INFINITY)
        self.low_px = <double>INFINITY
        self.close_px = 0.0
        self.close_time_ms = 0.0
        self.buy_sz = 0.0
        self.sell_sz = 0.0
        self.vwap_px = 0.0
        self.num_trades = 0.0
        
        self._cum_volume = 0.0
        self._total_sz = 0.0
        self._ringbuffer = RingBufferTwoDim(
            capacity=<u32>num_candles,
            sub_array_len=<u32>10
        )

    cdef inline double calculate_vwap(self, double px, double sz):
        """
        Calculate the current VWAP (Volume-Weighted Average Price).

        Args:
            px (float): The trade price.
            sz (float): The trade size (volume).

        Returns:
            float: The updated VWAP after incorporating this trade.
        """
        self._cum_volume += px * sz
        self._total_sz += sz
        if self._total_sz > 0.0:
            return self._cum_volume / self._total_sz
        return 0.0

    cdef void insert_candle(self):
        """
        Insert the current candle into the ring buffer and reset attributes.
        """
        # This can be faster with unsafe writes, but that can be 
        # optimized in a later minor release.
        self._ringbuffer.append(self.get_current())

        # Reset all candle attributes
        self.open_time_ms = 0.0
        self.open_px = 0.0
        self.high_px = <double>(-INFINITY)
        self.low_px = <double>INFINITY
        self.close_px = 0.0
        self.close_time_ms = 0.0
        self.buy_sz = 0.0
        self.sell_sz = 0.0
        self.vwap_px = 0.0
        self.num_trades = 0.0

        self._cum_volume = 0.0
        self._total_sz = 0.0

    cdef inline bint is_stale_trade(self, double time_ms):
        """
        Check if a trade is stale (older than the latest update).
        """
        return time_ms < self.close_time_ms
        
    cpdef void initialize(self, cnp.ndarray trades):
        """
        Initialize candle data from a batch of existing trades.

        Args:
            trades (np.ndarray): A 2D float array of trades, 
                where each row is [time, is_buy (0/1), price, size].
        """
        cdef:
            double  time_ms, px, sz
            bint    is_buy
            i64     len_trades = trades.shape[0]
            i64     dim_trades = trades.shape[1]
            i64     i

        if len_trades <= 0:
            raise ValueError("Cannot initialize from empty trades array")

        if dim_trades != 4:
            raise ValueError("Trades arrays must have 4 columns; [time, is_buy, px, sz]")

        # Reset all candle attributes incase this is called when a 
        # candle is currently being constructed.
        self.open_time_ms = 0.0
        self.open_px = 0.0
        self.high_px = <double>(-INFINITY)
        self.low_px = <double>INFINITY
        self.close_px = 0.0
        self.close_time_ms = 0.0
        self.buy_sz = 0.0
        self.sell_sz = 0.0
        self.vwap_px = 0.0
        self.num_trades = 0.0
        self._cum_volume = 0.0
        self._total_sz = 0.0

        for i in range(len_trades):
            time_ms = trades[i, 0]
            is_buy = trades[i, 1] == 1.0
            px = trades[i, 2]
            sz = trades[i, 3]
            self.process_trade(time_ms, is_buy, px, sz)

    cpdef void process_trade(self, double time_ms, bint is_buy, double px, double sz):
        """
        Process a single trade tick, updating the current candle.

        Args:
            time_ms (double): The trade time in milliseconds.
            is_buy (bint): Whether the trade is a buy.
            px (double): The trade price.
            sz (double): The trade size.

        Note:
            This method is meant to be overridden by subclasses for specialized behavior.
        """
        pass

    cpdef cnp.ndarray get_current(self):
        """
        Return the current candle as a NumPy array.

        Returns:
            np.ndarray: A 1D array of length 10 representing the current candle.
        """
        return np.array([
            self.open_time_ms,
            self.open_px,
            self.high_px,
            self.low_px,
            self.close_px,
            self.close_time_ms,
            self.buy_sz,
            self.sell_sz,
            self.vwap_px,
            self.num_trades,    
        ], dtype=np.float64)

    cpdef cnp.ndarray get_all(self):
        """
        Return the ring buffer's candle data plus the current candle if it's open.

        Returns:
            np.ndarray: A 2D array of candle rows. 
        """
        if self.open_time_ms != 0.0:
            return np.vstack((self._ringbuffer.unwrapped(), self.get_current()))
        else:
            return self._ringbuffer.unwrapped()

    def __len__(self):
        """
        Number of candles currently stored.

        Returns:
            int: The count of valid candles in the ring buffer.
        """
        return len(self._ringbuffer)

    def __getitem__(self, int index):
        """
        Access a specific candle by index.

        Args:
            index (int): The candle index in logical order (0 is oldest).

        Returns:
            np.ndarray: A 1D array of length 10 representing the candle.

        Raises:
            IndexError: If the index is out of range.
        """
        return self._ringbuffer[index]

    def __iter__(self):
        """
        Iterate over all candles in the ring buffer in logical order.

        Yields:
            np.ndarray: Each candle as a 1D array of length 10.
        """
        return iter(self._ringbuffer)
