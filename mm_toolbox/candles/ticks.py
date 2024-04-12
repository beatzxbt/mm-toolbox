import numpy as np
from numba import njit
from numba.types import bool_, int64, float64, Array
from numba.experimental import jitclass

from mm_toolbox.ringbuffer.ringbuffer import RingBufferF64

spec = [
    ('bucket_size', int64),
    ('_time_', RingBufferF64.class_type.instance_type),
    ('_open_', RingBufferF64.class_type.instance_type),
    ('_high_', RingBufferF64.class_type.instance_type),
    ('_low_', RingBufferF64.class_type.instance_type),
    ('_close_', RingBufferF64.class_type.instance_type),
    ('_size_', RingBufferF64.class_type.instance_type),
    ('_current_time_', float64),
    ('_current_open_', float64),
    ('_current_high_', float64),
    ('_current_low_', float64),
    ('_current_close_', float64),
    ('_current_size_', float64),
    ('_current_trade_count_', int64),
]


@jitclass(spec)
class TickCandles:
    def __init__(self, num_candles: int, bucket_size: int) -> None:
        self.bucket_size = bucket_size

        self._time_ = RingBufferF64(num_candles)
        self._open_ = RingBufferF64(num_candles)
        self._high_ = RingBufferF64(num_candles)
        self._low_ = RingBufferF64(num_candles)
        self._close_ = RingBufferF64(num_candles)
        self._size_ = RingBufferF64(num_candles)

        self._current_time_ = 0.0
        self._current_open_ = 0.0
        self._current_high_ = 0.0
        self._current_low_ = 0.0
        self._current_close_ = 0.0
        self._current_size_ = 0.0
        self._current_trade_count_ = 0

    def __getitem__(self, item: int) -> Array:
        return np.array([
            self._time_[item],
            self._open_[item],
            self._high_[item],
            self._low_[item],
            self._close_[item],
            self._size_[item],
        ]).T
    
    def _reset_current_candle_(self) -> None:
        self._current_time_ = 0.0
        self._current_open_ = 0.0
        self._current_high_ = 0.0
        self._current_low_ = 0.0
        self._current_close_ = 0.0
        self._current_size_ = 0.0
        self._current_trade_count_ = 0
    
    def _process_tick_(self, time: float, price: float, size: float) -> None:
        if self._current_trade_count_ == 0:
            self._current_time_ = time
            self._current_open_ = price

        self._current_high_ = max(self._current_high_, price)
        self._current_low_ = min(self._current_low_, price)
        self._current_size_ += size

        if self._current_trade_count_ >= self.bucket_size:
            self._current_close_ = price
            self._time_.appendright(self._current_time_)
            self._open_.appendright(self._current_open_)
            self._high_.appendright(self._current_high_)
            self._low_.appendright(self._current_low_)
            self._close_.appendright(self._current_close_)
            self._size_.appendright(self._current_size_)
            self._reset_current_candle_()

        self._current_trade_count_ += 1

    def initialize(self, trades: Array) -> Array:
        """ Trades in format F64Array[[Time, Side, Price, Size]]"""
        for trade in trades:
            self._process_tick_(trade[0], trade[2], trade[3])

    def update(self, time: float, price: float, size: float) -> None:
        self._process_tick_(time, price, size)