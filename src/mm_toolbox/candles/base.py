import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Iterator, Union

from mm_toolbox.ringbuffer import RingBufferMultiDim


class BaseCandles(ABC):
    """
    A class to aggregate trades into pre-defined fixed buckets. 
    
    Format
    ---------
    Candle[]: 
        [0] = Open Price
        [1] = High Price
        [2] = Low Price
        [3] = Close Price
        [4] = Buy Volume
        [5] = Sell Volume
        [6] = VWAP Price
        [7] = Total Trades
        [8] = Open Timestamp
        [9] = Close Timestamp

    Parameters
    ----------
    num_candles : int
        The number of candles to maintain in the ring buffer
    """
    def __init__(self, num_candles: int) -> None:
        self.num_candles = num_candles

        self.open_price = 0.0
        self.high_price = -np.inf  
        self.low_price = np.inf  
        self.close_price = 0.0
        self.buy_volume = 0.0
        self.sell_volume = 0.0
        self.vwap_price = 0.0
        self.total_trades = 0.0
        self.open_timestamp = 0.0
        self.close_timestamp = 0.0

        self._cum_price_volume_ = 0.0
        self._total_volume_ = 0.0

        self.ringbuffer = RingBufferMultiDim(
            shape=(num_candles, 10),
            dtype=np.float64
        )

    def as_array(self) -> np.ndarray[np.ndarray]:
        """
        Returns the aggregated candle data as a NumPy array.

        Returns
        -------
        np.ndarray[np.ndarray]
            The array of aggregated candle data.
        """
        if not self.ringbuffer.is_empty:
            current_candle = np.array([[
                self.open_price,
                self.high_price,
                self.low_price,
                self.close_price,
                self.buy_volume,
                self.sell_volume,
                self.vwap_price,
                self.total_trades,
                self.open_timestamp,
                self.close_timestamp
            ]])

            return np.concatenate((
                self.ringbuffer.as_array(),
                current_candle
            ))
        
        else:
            return np.array([[]], dtype=np.float64)
    
    def reset_current_candle(self) -> None:
        """
        Resets the current candle data to its initial state.
        """
        self.open_price = 0.0
        self.high_price = -np.inf  
        self.low_price = np.inf  
        self.close_price = 0.0
        self.buy_volume = 0.0
        self.sell_volume = 0.0
        self.vwap_price = 0.0
        self.total_trades = 0.0
        self.open_timestamp = 0.0
        self.close_timestamp = 0.0
        self.total_volume = 0.0

        self._cum_price_volume_ = 0.0
        self._total_volume_ = 0.0
    
    def insert_candle(
        self, 
        open_price: float, 
        high_price: float, 
        low_price: float, 
        close_price: float, 
        buy_volume: float, 
        sell_volume: float,
        vwap_price: float,
        total_trades: float,
        open_timestamp: float,
        close_timestamp: float
    ) -> None:
        """
        Inserts a completed candle into the ring buffer and resets the current candle.

        Parameters
        ----------
        open_price : float
            The open price of the candle.

        high_price : float
            The high price of the candle.

        low_price : float
            The low price of the candle.

        close_price : float
            The close price of the candle.

        buy_volume : float
            The buy volume of the candle.

        sell_volume : float
            The sell volume of the candle.

        vwap_price : float
            The volume-weighted average price of the candle.

        total_trades : float
            The total number of trades in the candle.

        open_timestamp : float
            The open timestamp of the candle.

        close_timestamp : float
            The close timestamp of the candle.
        """
        current_candle = np.array([
            open_price,
            high_price,
            low_price,
            close_price,
            buy_volume,
            sell_volume,
            vwap_price,
            total_trades,
            open_timestamp,
            close_timestamp
        ])
        self.ringbuffer.appendright(current_candle)
        self.reset_current_candle()

    @abstractmethod
    def process_trade(self, timestamp: float, side: bool, price: float, size: float) -> None:
        """
        Processes a single trade tick and updates the current candle data.

        Parameters
        ----------
        timestamp : float
            The timestamp of the trade.

        side : bool
            Whether the trade is a buy (True) or sell (False).

        price : float
            The price at which the trade occurred.

        size : float
            The size (volume) of the trade.
        """
        pass
    
    def initialize(self, trades: np.ndarray) -> None:
        """
        Initializes the candle data with a batch of trades.

        Parameters
        ----------
        trades : np.ndarray
            A 2D NumPy array of trades in the format [[Timestamp, Side, Price, Size]].
        """
        for trade in trades:
            self.process_trade(*trade)

    def update(self, timestamp: float, side: bool, price: float, size: float) -> None:
        """
        Updates the candle data with a new trade.

        Parameters
        ----------
        timestamp : float
            The timestamp of the trade.

        side : bool
            Whether the trade is a buy (True) or sell (False).

        price : float
            The price at which the trade occurred.

        size : float
            The size (volume) of the trade.
        """
        self.process_trade(timestamp, side, price, size)

    def calculate_vwap(self, price: float, size: float) -> float:
        self._cum_price_volume_ += price * size
        self._total_volume_ += size
        return self._cum_price_volume_ / self._total_volume_
    
    @property
    def durations(self) -> np.ndarray:
        candles = self.as_array()
        return candles[:, 9] - candles[:, 8]
    
    @property
    def imbalances(self) -> np.ndarray:
        candles = self.as_array()
        return candles[:, 4] / candles[:, 5]
    
    @property
    def current_candle(self) -> np.ndarray:
        return np.array([
            self.open_price,
            self.high_price,
            self.low_price,
            self.close_price,
            self.buy_volume,
            self.sell_volume,
            self.vwap_price,
            self.total_trades,
            self.open_timestamp,
            self.close_timestamp
        ])
    
    def __getitem__(self, index: Union[int, Tuple]) -> np.ndarray:
        return self.ringbuffer.as_array()[index]
    
    def __len__(self) -> int:
        return len(self.ringbuffer)
    
    def __iter__(self) -> Iterator[np.ndarray]:
        """
        Return an iterator over the candles.

        Returns
        -------
        iterator
            An iterator over the candles.
        """
        return iter(self.as_array())
    
