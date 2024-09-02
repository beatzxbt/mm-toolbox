import numpy as np
from numba.types import float64
from numba.experimental.jitclass import jitclass


@jitclass
class Round:
    """
    A class for performing rounding operations on prices and sizes according to specified tick and lot sizes.

    Parameters
    ----------
    tick_size : float
        The step size for rounding prices.

    lot_size : float
        The step size for rounding sizes.
    """

    tick_size: float64
    lot_size: float64

    _inverse_tick_size: float64
    _inverse_lot_size: float64
    _tick_rounding_decimals: float64
    _lot_rounding_decimals: float64

    def __init__(self, tick_size: float, lot_size: float) -> None:
        self.tick_size = tick_size
        self.lot_size = lot_size

        self._inverse_tick_size = 1.0 / self.tick_size
        self._inverse_lot_size = 1.0 / self.lot_size
        self._tick_rounding_decimals = float64(np.ceil(-np.log10(self.tick_size)))
        self._lot_rounding_decimals = float64(np.ceil(-np.log10(self.lot_size)))

    def bid(self, price: float) -> float:
        """
        Rounds the given price down to the nearest tick size multiple.

        Parameters
        ----------
        price : float
            The price to be rounded.

        Returns
        -------
        float
            The rounded price.
        """
        return round(
            self.tick_size * np.floor(price * self._inverse_tick_size),
            self._tick_rounding_decimals,
        )

    def ask(self, price: float) -> float:
        """
        Rounds the given price up to the nearest tick size multiple.

        Parameters
        ----------
        price : float
            The price to be rounded.

        Returns
        -------
        float
            The rounded price.
        """
        return round(
            self.tick_size * np.ceil(price * self._inverse_tick_size),
            self._tick_rounding_decimals,
        )

    def size(self, size: float) -> float:
        """
        Rounds the given size down to the nearest lot size multiple.

        Parameters
        ----------
        size : float
            The size to be rounded.

        Returns
        -------
        float
            The rounded size.
        """
        return round(
            self.lot_size * np.floor(size * self._inverse_lot_size),
            self._lot_rounding_decimals,
        )

    def bids(self, prices: np.ndarray[float]) -> np.ndarray[float]:
        """
        Rounds an array of prices down to the nearest tick size multiple.

        Parameters
        ----------
        prices : np.ndarray[float]
            The array of prices to be rounded.

        Returns
        -------
        np.ndarray[float]
            The array of rounded prices.
        """
        return np.round(
            self.tick_size * np.floor(prices * self._inverse_tick_size),
            self._tick_rounding_decimals,
        )

    def asks(self, prices: np.ndarray[float]) -> np.ndarray[float]:
        """
        Rounds an array of prices up to the nearest tick size multiple.

        Parameters
        ----------
        prices : np.ndarray[float]
            The array of prices to be rounded.

        Returns
        -------
        np.ndarray[float]
            The array of rounded prices.
        """
        return np.round(
            self.tick_size * np.ceil(prices * self._inverse_tick_size),
            self._tick_rounding_decimals,
        )

    def sizes(self, sizes: np.ndarray[float]) -> np.ndarray[float]:
        """
        Rounds an array of sizes down to the nearest lot size multiple.

        Parameters
        ----------
        sizes : np.ndarray[float]
            The array of sizes to be rounded.

        Returns
        -------
        np.ndarray[float]
            The array of rounded sizes.
        """
        return np.round(
            self.lot_size * np.ceil(sizes * self._inverse_lot_size),
            self._lot_rounding_decimals,
        )