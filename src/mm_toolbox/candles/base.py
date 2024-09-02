import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Iterator, Union

from mm_toolbox.ringbuffer import RingBufferTwoDimFloat


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

        self._cum_price_volume = 0.0
        self._total_volume = 0.0

        self.ringbuffer = RingBufferTwoDimFloat(
            capacity=self.num_candles, sub_array_len=10
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
            if self.open_timestamp != 0.0:
                return np.concatenate(
                    (
                        self.ringbuffer.as_array(),
                        np.array(
                            [
                                [
                                    self.open_price,
                                    self.high_price,
                                    self.low_price,
                                    self.close_price,
                                    self.buy_volume,
                                    self.sell_volume,
                                    self.vwap_price,
                                    self.total_trades,
                                    self.open_timestamp,
                                    self.close_timestamp,
                                ]
                            ]
                        ),
                    )
                )

            else:
                return self.ringbuffer.as_array()

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

        self._cum_price_volume = 0.0
        self._total_volume = 0.0

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
        close_timestamp: float,
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
        current_candle = np.array(
            [
                open_price,
                high_price,
                low_price,
                close_price,
                buy_volume,
                sell_volume,
                vwap_price,
                total_trades,
                open_timestamp,
                close_timestamp,
            ]
        )
        self.ringbuffer.append(current_candle)
        self.reset_current_candle()

    @abstractmethod
    def process_trade(
        self, timestamp: float, side: bool, price: float, size: float
    ) -> None:
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
        Updates the candle data with a new trade. Checks if the update is
        as a result of stale data being recieved or not.

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
        if timestamp >= self.open_timestamp:
            self.process_trade(timestamp, side, price, size)

    def calculate_vwap(self, price: float, size: float) -> float:
        self._cum_price_volume += price * size
        self._total_volume += size
        return self._cum_price_volume / self._total_volume

    def durations(self) -> np.ndarray[float]:
        candles = self.as_array()
        return candles[:, 9] - candles[:, 8]

    def imbalances(self) -> np.ndarray[float]:
        candles = self.as_array()
        return candles[:, 4] / candles[:, 5]

    def average_true_range(self) -> np.ndarray[float]:
        """
        Calculate the true range of a trading price bar.

        The true range is the greatest of the following:
        - The difference between the current high and the current low,
        - The absolute difference between the current high and the previous close,
        - The absolute difference between the current low and the previous close.

        Returns
        -------
        np.ndarray[float]
            The true range of price in the candles.
        """
        candles = self.as_array()

        if len(candles) < 2:
            return np.array([], dtype=np.float64)

        high_low_diff = candles[:, 1] - candles[:, 2]
        high_prev_close_diff = np.abs(candles[1:, 1] - candles[:-1, 3])
        low_prev_close_diff = np.abs(candles[1:, 2] - candles[:-1, 3])

        # True Range is the maximum of the three calculated differences
        true_range = np.maximum.reduce(
            [high_low_diff[1:], high_prev_close_diff, low_prev_close_diff]
        )

        return true_range

    def rsi(self, period: int = 14) -> np.ndarray[float]:
        """
        Calculate the Relative Strength Index (RSI) for the given period.

        Parameters
        ----------
        period : int
            The period over which to calculate the RSI (default is 14).

        Returns
        -------
        np.ndarray
            The RSI values for each candle.
        """
        close_prices = self.close_prices

        if period >= close_prices.size:
            raise RuntimeWarning(
                f"Not enough candles for period {period}, calculating partial RSI only."
            )

        delta = np.diff(close_prices, 1)
        gain = np.where(delta > 0.0, delta, 0.0)
        loss = np.where(delta < 0.0, -delta, 0.0)

        avg_gain = np.zeros_like(close_prices)
        avg_loss = np.zeros_like(close_prices)

        for i in range(1, period):
            # Partial RSI is just previous price's SMA
            avg_gain[i] = gain[:i].mean()
            avg_loss[i] = loss[:i].mean()

        for i in range(period, len(close_prices) - 1):
            avg_gain[i] = (avg_gain[i - 1] * (period - 1.0) + gain[i]) / period
            avg_loss[i] = (avg_loss[i - 1] * (period - 1.0) + loss[i]) / period

        rs = np.divide(avg_gain[1:], avg_loss[1:], where=avg_loss[1:] != 0)
        rsi = 100.0 - (100.0 / (1.0 + rs))

        return rsi

    def bollinger_bands(
        self, period: int = 20, num_std_dev: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate Bollinger Bands (BB) for the given period.

        Parameters
        ----------
        period : int
            The period over which to calculate the Bollinger Bands (default is 20).

        num_std_dev : float
            The number of standard deviations to use for the bands (default is 2.0).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            The lower band, middle band (SMA), and upper band.
        """
        close_prices = self.close_prices

        sma = np.convolve(close_prices, np.ones(period) / period, mode="valid")
        rolling_std = np.zeros_like(sma)

        for i in range(sma.size):
            rolling_std[i] = np.std(close_prices[i : i + period])

        upper_band = sma + num_std_dev * rolling_std
        lower_band = sma - num_std_dev * rolling_std

        return lower_band, sma, upper_band

    @property
    def current_candle(self) -> np.ndarray[float]:
        return np.array(
            [
                self.open_price,
                self.high_price,
                self.low_price,
                self.close_price,
                self.buy_volume,
                self.sell_volume,
                self.vwap_price,
                self.total_trades,
                self.open_timestamp,
                self.close_timestamp,
            ]
        )

    @property
    def open_prices(self) -> np.ndarray[float]:
        if not self.ringbuffer.is_empty:
            return self.as_array()[:, 0]
        else:
            return np.array([[]], dtype=np.float64)

    @property
    def high_prices(self) -> np.ndarray[float]:
        if not self.ringbuffer.is_empty:
            return self.as_array()[:, 1]
        else:
            return np.array([[]], dtype=np.float64)

    @property
    def low_prices(self) -> np.ndarray[float]:
        if not self.ringbuffer.is_empty:
            return self.as_array()[:, 2]
        else:
            return np.array([[]], dtype=np.float64)

    @property
    def close_prices(self) -> np.ndarray[float]:
        if not self.ringbuffer.is_empty:
            return self.as_array()[:, 3]
        else:
            return np.array([[]], dtype=np.float64)

    @property
    def buy_volumes(self) -> np.ndarray[float]:
        if not self.ringbuffer.is_empty:
            return self.as_array()[:, 4]
        else:
            return np.array([[]], dtype=np.float64)

    @property
    def sell_volumes(self) -> np.ndarray[float]:
        if not self.ringbuffer.is_empty:
            return self.as_array()[:, 5]
        else:
            return np.array([[]], dtype=np.float64)

    @property
    def vwap_prices(self) -> np.ndarray[float]:
        if not self.ringbuffer.is_empty:
            return self.as_array()[:, 6]
        else:
            return np.array([[]], dtype=np.float64)

    @property
    def all_trades(self) -> np.ndarray[float]:
        if not self.ringbuffer.is_empty:
            return self.as_array()[:, 7]
        else:
            return np.array([[]], dtype=np.float64)

    def __getitem__(self, index: Union[int, Tuple]) -> np.ndarray:
        return self.as_array()[index]

    def __len__(self) -> int:
        return len(self.ringbuffer)

    def __iter__(self) -> Iterator[np.ndarray]:
        return iter(self.as_array())
