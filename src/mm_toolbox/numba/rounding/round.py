import numpy as np
from numba.types import int64, float64
from numba.experimental.jitclass import jitclass


@jitclass
class Round:
    """
    A class for performing rounding operations on prices and sizes according to specified tick and lot sizes.
    """

    tick_size: float64
    lot_size: float64

    _inverse_tick_size: float64
    _inverse_lot_size: float64
    _tick_rounding_decimals: int64
    _lot_rounding_decimals: int64

    def __init__(self, tick_size: float, lot_size: float) -> None:
        if tick_size <= 0.0:
            raise ValueError("Invalid tick_size; must be greater than 0.")

        if lot_size <= 0.0:
            raise ValueError("Invalid lot_size; must be greater than 0.")

        self.tick_size = tick_size
        self.lot_size = lot_size

        self._inverse_tick_size = 1.0 / self.tick_size
        self._inverse_lot_size = 1.0 / self.lot_size
        self._tick_rounding_decimals = int64(np.ceil(-np.log10(self.tick_size)))
        self._lot_rounding_decimals = int64(np.ceil(-np.log10(self.lot_size)))

    def bid(self, px: float) -> float:
        """
        Rounds the given price down to the nearest tick size multiple.

        Parameters
        ----------
        px : float
            The price to be rounded.

        Returns
        -------
        float
            The rounded price.
        """
        return round(
            self.tick_size * np.floor(px * self._inverse_tick_size),
            self._tick_rounding_decimals,
        )

    def ask(self, px: float) -> float:
        """
        Rounds the given price up to the nearest tick size multiple.

        Parameters
        ----------
        px : float
            The price to be rounded.

        Returns
        -------
        float
            The rounded price.
        """
        return round(
            self.tick_size * np.ceil(px * self._inverse_tick_size),
            self._tick_rounding_decimals,
        )

    def size(self, sz: float) -> float:
        """
        Rounds the given size down to the nearest lot size multiple.

        Parameters
        ----------
        sz : float
            The size to be rounded.

        Returns
        -------
        float
            The rounded size.
        """
        return round(
            self.lot_size * np.floor(sz * self._inverse_lot_size),
            self._lot_rounding_decimals,
        )

    def bids(self, pxs: np.ndarray) -> np.ndarray:
        """
        Rounds an array of prices down to the nearest tick size multiple.

        Parameters
        ----------
        pxs : np.ndarray
            The array of prices to be rounded.

        Returns
        -------
        np.ndarray
            The array of rounded prices.
        """
        return np.round(
            self.tick_size * np.floor(pxs * self._inverse_tick_size),
            self._tick_rounding_decimals,
        )

    def asks(self, pxs: np.ndarray) -> np.ndarray:
        """
        Rounds an array of prices up to the nearest tick size multiple.

        Parameters
        ----------
        pxs : np.ndarray
            The array of prices to be rounded.

        Returns
        -------
        np.ndarray
            The array of rounded prices.
        """
        return np.round(
            self.tick_size * np.ceil(pxs * self._inverse_tick_size),
            self._tick_rounding_decimals,
        )

    def sizes(self, szs: np.ndarray) -> np.ndarray:
        """
        Rounds an array of sizes down to the nearest lot size multiple.

        Parameters
        ----------
        szs : np.ndarray
            The array of sizes to be rounded.

        Returns
        -------
        np.ndarray
            The array of rounded sizes.
        """
        return np.round(
            self.lot_size * np.ceil(szs * self._inverse_lot_size),
            self._lot_rounding_decimals,
        )
