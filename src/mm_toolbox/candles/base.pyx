import numpy as np
cimport numpy as np
from libc.math cimport INFINITY
from libc.stdint cimport uint32_t

from mm_toolbox.ringbuffer.twodim cimport RingBufferTwoDim

cdef class BaseCandles:
    """
    A class to aggregate trades into pre-defined fixed buckets.

    Format
    ------
    Candle[]:
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

    def __init__(self, uint32_t num_candles) -> None:
        if num_candles < 1:
            raise ValueError(f"Invalid number of candles; expected >1 but got {num_candles}.")
            
        self.open_price = 0.0
        self.high_price = <double>(-INFINITY)
        self.low_price = <double>INFINITY
        self.close_price = 0.0
        self.buy_volume = 0.0
        self.sell_volume = 0.0
        self.vwap_price = 0.0
        self.total_trades = 0.0
        self.open_time = 0.0
        self.close_time = 0.0

        self._cum_price_volume = 0.0
        self._total_volume = 0.0
        self._ringbuffer = RingBufferTwoDim(num_candles, 10)

    cdef inline double calculate_vwap(self, double px, double sz):
        self._cum_price_volume += px * sz
        self._total_volume += sz
        return self._cum_price_volume / self._total_volume

    cdef void insert_candle(self):
        """
        Inserts the completed candle into the ring buffer and resets the current candle.
        """
        self._ringbuffer.append(self.current_candle())
        
        # Reset all attributes.
        self.open_price = 0.0
        self.high_price = <double>(-INFINITY)
        self.low_price = <double>INFINITY
        self.close_price = 0.0
        self.buy_volume = 0.0
        self.sell_volume = 0.0
        self.vwap_price = 0.0
        self.total_trades = 0.0
        self.open_time = 0.0
        self.close_time = 0.0

        self._cum_price_volume = 0.0
        self._total_volume = 0.0

    cpdef void process_trade(self, double time, bint is_buy, double px, double sz):
        """
        Processes a single trade tick and updates the current candle data.
        This method should be overridden by subclasses.
        """
        pass

    cpdef np.ndarray unwrapped(self):
        """
        Returns the aggregated candle data as a NumPy array.
        """
        if self.open_time != 0.0:
            return np.vstack((
                self._ringbuffer.unwrapped(),
                self.current_candle()
            ))
        else:
            return self._ringbuffer.unwrapped()

    cpdef void initialize(self, np.ndarray trades):
        """
        Initializes the candle data with a batch of trades.
        """
        cdef:
            double time, px, sz
            bint is_buy
            uint32_t i, n = trades.shape[0]

        for i in range(n):
            time = trades[i, 0]
            is_buy = trades[i, 1] != 0.0
            px = trades[i, 2]
            sz = trades[i, 3]
            self.process_trade(time, is_buy, px, sz)

    cpdef np.ndarray durations(self):
        cdef np.ndarray candles = self.unwrapped()
        return candles[:, 9] - candles[:, 8]

    cpdef np.ndarray imbalances(self):
        cdef np.ndarray candles = self.unwrapped()
        return candles[:, 4] / candles[:, 5]

    cpdef np.ndarray current_candle(self):
        return np.array([
            self.open_price,
            self.high_price,
            self.low_price,
            self.close_price,
            self.buy_volume,
            self.sell_volume,
            self.vwap_price,
            self.total_trades,
            self.open_time,
            self.close_time,
        ], dtype=np.float64)

    cpdef np.ndarray open_prices(self):
        if not self._ringbuffer.is_empty():
            return self.unwrapped()[:, 0]
        else:
            return np.array([], dtype=np.float64)

    cpdef np.ndarray high_prices(self):
        if not self._ringbuffer.is_empty():
            return self.unwrapped()[:, 1]
        else:
            return np.array([], dtype=np.float64)

    cpdef np.ndarray low_prices(self):
        if not self._ringbuffer.is_empty():
            return self.unwrapped()[:, 2]
        else:
            return np.array([], dtype=np.float64)

    cpdef np.ndarray close_prices(self):
        if not self._ringbuffer.is_empty():
            return self.unwrapped()[:, 3]
        else:
            return np.array([], dtype=np.float64)

    cpdef np.ndarray buy_volumes(self):
        if not self._ringbuffer.is_empty():
            return self.unwrapped()[:, 4]
        else:
            return np.array([], dtype=np.float64)

    cpdef np.ndarray sell_volumes(self):
        if not self._ringbuffer.is_empty():
            return self.unwrapped()[:, 5]
        else:
            return np.array([], dtype=np.float64)

    cpdef np.ndarray vwap_prices(self):
        if not self._ringbuffer.is_empty():
            return self.unwrapped()[:, 6]
        else:
            return np.array([], dtype=np.float64)

    cpdef np.ndarray all_trades(self):
        if not self._ringbuffer.is_empty():
            return self.unwrapped()[:, 7]
        else:
            return np.array([], dtype=np.float64)

    def __len__(self):
        return len(self._ringbuffer)

    def __getitem__(self, int index):
        return self._ringbuffer[index]

    def __iter__(self):
        return iter(self.unwrapped())
