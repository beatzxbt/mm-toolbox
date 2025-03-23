import numpy as np
cimport numpy as cnp
from libc.math cimport ceil, floor, log10, pow, round

cdef class Round:
    """
    Provides rounding operations on prices and sizes according to specified tick and lot sizes.
    """

    def __cinit__(self, double tick_sz, double lot_sz):
        """
        Initialize the Round class with given tick and lot sizes.

        Args:
            tick_sz (float): The minimum price increment.
            lot_sz (float): The minimum size increment.

        Raises:
            ValueError: If either tick_sz or lot_sz is <= 0.
        """
        if tick_sz <= 0.0:
            raise ValueError("Invalid tick_sz; must be greater than 0")
        if lot_sz <= 0.0:
            raise ValueError("Invalid lot_sz; must be greater than 0")

        self.tick_sz = tick_sz
        self.lot_sz = lot_sz

        # Precompute frequently used values
        self._inverse_tick_sz = 1.0 / self.tick_sz
        self._inverse_lot_sz = 1.0 / self.lot_sz
        self._tick_rounding_factor = pow(10.0, ceil(-log10(self.tick_sz)))
        self._lot_rounding_factor = pow(10.0, ceil(-log10(self.lot_sz)))

    cdef inline double _round_to(self, double num, double factor) nogil:
        """
        Round a value to a specific decimal precision.

        Args:
            num (float): The value to round.
            factor (float): The rounding factor (10^decimals).

        Returns:
            float: The rounded value.
        """
        return round(num * factor) / factor

    cpdef double bid(self, double px):
        """
        Round a price down to the nearest tick size multiple.

        Args:
            px (float): The price to be rounded down.

        Returns:
            float: The rounded (bid) price.
        """
        cdef double value = self.tick_sz * floor(px * self._inverse_tick_sz)
        return self._round_to(value, self._tick_rounding_factor)

    cpdef double ask(self, double px):
        """
        Round a price up to the nearest tick size multiple.

        Args:
            px (float): The price to be rounded up.

        Returns:
            float: The rounded (ask) price.
        """
        cdef double value = self.tick_sz * ceil(px * self._inverse_tick_sz)
        return self._round_to(value, self._tick_rounding_factor)

    cpdef double sz(self, double sz):
        """
        Round a size down to the nearest lot size multiple.

        Args:
            sz (float): The size to be rounded down.

        Returns:
            float: The rounded size.
        """
        cdef double value = self.lot_sz * floor(sz * self._inverse_lot_sz)
        return self._round_to(value, self._lot_rounding_factor)

    cpdef cnp.ndarray bids(self, cnp.ndarray pxs):
        """
        Round an array of prices down to the nearest tick size multiple (bids).

        Args:
            pxs (np.ndarray): An array of prices to be rounded.

        Returns:
            np.ndarray: A new array containing rounded bid prices.
        """
        cdef:
            double[:] pxs_view = pxs
            Py_ssize_t n = pxs_view.shape[0]
            cnp.ndarray result = np.empty(n, dtype=np.double)
            double[:] result_view = result
            Py_ssize_t i
            
            # Local copies of instance variables for faster access in the loop
            double tick_sz = self.tick_sz
            double inverse_tick_sz = self._inverse_tick_sz
            double tick_rounding_factor = self._tick_rounding_factor

        for i in range(n):
            result_view[i] = self._round_to(
                tick_sz * floor(pxs_view[i] * inverse_tick_sz),
                tick_rounding_factor
            )
        return result

    cpdef cnp.ndarray asks(self, cnp.ndarray pxs):
        """
        Round an array of prices up to the nearest tick size multiple (asks).

        Args:
            pxs (np.ndarray): An array of prices to be rounded.

        Returns:
            np.ndarray: A new array containing rounded ask prices.
        """
        cdef:
            double[:] pxs_view = pxs
            Py_ssize_t n = pxs_view.shape[0]
            cnp.ndarray result = np.empty(n, dtype=np.double)
            double[:] result_view = result
            Py_ssize_t i

            # Local copies of instance variables for faster access in the loop
            double tick_sz = self.tick_sz
            double inverse_tick_sz = self._inverse_tick_sz
            double tick_rounding_factor = self._tick_rounding_factor

        for i in range(n):
            result_view[i] = self._round_to(
                tick_sz * ceil(pxs_view[i] * inverse_tick_sz),
                tick_rounding_factor
            )
        return result

    cpdef cnp.ndarray szs(self, cnp.ndarray szs):
        """
        Round an array of sizes down to the nearest lot size multiple.

        Args:
            szs (np.ndarray): An array of sizes to be rounded.

        Returns:
            np.ndarray: A new array containing rounded sizes.
        """
        cdef:
            double[:] szs_view = szs
            Py_ssize_t n = szs_view.shape[0]
            cnp.ndarray result = np.empty(n, dtype=np.double)
            double[:] result_view = result
            Py_ssize_t i

            # Local copies of instance variables for faster access in the loop
            double lot_sz = self.lot_sz
            double inverse_lot_sz = self._inverse_lot_sz
            double lot_rounding_factor = self._lot_rounding_factor

        for i in range(n):
            result_view[i] = self._round_to(
                lot_sz * floor(szs_view[i] * inverse_lot_sz),
                lot_rounding_factor
            )
        return result
