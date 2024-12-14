import numpy as np
cimport numpy as np
from libc.math cimport ceil, floor, log10, pow, round

cdef class Round:
    """
    A class for performing rounding operations on prices and sizes according to specified tick and lot sizes.
    """

    def __init__(self, double tick_size, double lot_size):
        if tick_size <= 0.0:
            raise ValueError("tick_size must be greater than 0.")

        if lot_size <= 0.0:
            raise ValueError("lot_size must be greater than 0.")

        self.tick_size = tick_size
        self.lot_size = lot_size

        self._inverse_tick_size = 1.0 / self.tick_size
        self._inverse_lot_size = 1.0 / self.lot_size
        self._tick_rounding_factor = pow(10.0, ceil(-log10(self.tick_size)))
        self._lot_rounding_factor = pow(10.0, ceil(-log10(self.lot_size)))

    cdef inline double _round_to(self, double num, double factor):
        """
        Rounds a value to the specified number of decimal places.
        """
        return round(num * factor) / factor

    cpdef double bid(self, double px):
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
        cdef double value = self.tick_size * floor(px * self._inverse_tick_size)
        return self._round_to(value, self._tick_rounding_factor)

    cpdef double ask(self, double px):
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
        cdef double value = self.tick_size * ceil(px * self._inverse_tick_size)
        return self._round_to(value, self._tick_rounding_factor)

    cpdef double size(self, double sz):
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
        cdef double value = self.lot_size * floor(sz * self._inverse_lot_size)
        return self._round_to(value, self._lot_rounding_factor)

    cpdef np.ndarray bids(self, np.ndarray pxs):
        """
        Rounds an array of prices down to the nearest tick size multiple.

        Parameters
        ----------
        prices : array_like
            The array of prices to be rounded.

        Returns
        -------
        np.ndarray
            The array of rounded prices.
        """
        cdef:
            double[:] pxs_view = pxs
            int n = pxs_view.shape[0]
            np.ndarray result = np.empty(n, dtype=np.double)
            double[:] result_view = result
            int i

        for i in range(n):
            result_view[i] = self._round_to(
                self.tick_size * floor(pxs_view[i] * self._inverse_tick_size),
                self._tick_rounding_factor
            )
        return result

    cpdef np.ndarray asks(self, np.ndarray pxs):
        """
        Rounds an array of prices up to the nearest tick size multiple.

        Parameters
        ----------
        prices : array_like
            The array of prices to be rounded.

        Returns
        -------
        np.ndarray
            The array of rounded prices.
        """
        cdef:
            double[:] pxs_view = pxs
            int n = pxs_view.shape[0]
            np.ndarray result = np.empty(n, dtype=np.double)
            double[:] result_view = result
            int i

        for i in range(n):
            result_view[i] = self._round_to(
                self.tick_size * ceil(pxs_view[i] * self._inverse_tick_size),
                self._tick_rounding_factor
            )
        return result

    cpdef np.ndarray sizes(self, np.ndarray szs):
        """
        Rounds an array of sizes down to the nearest lot size multiple.

        Parameters
        ----------
        sizes : array_like
            The array of sizes to be rounded.

        Returns
        -------
        np.ndarray
            The array of rounded sizes.
        """
        cdef:
            double[:] szs_view = szs
            int n = szs_view.shape[0]
            np.ndarray result = np.empty(n, dtype=np.double)
            double[:] result_view = result
            int i

        for i in range(n):
            result_view[i] = self._round_to(
                self.lot_size * floor(szs_view[i] * self._inverse_lot_size),
                self._lot_rounding_factor
            )
        return result
