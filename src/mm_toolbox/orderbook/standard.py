import numpy as np
from numba.types import int32, float64, bool_
from numba.experimental import jitclass
from typing import Dict, Union

from src.mm_toolbox.numba import nbisin, nbroll


@jitclass
class Orderbook:
    """
    An orderbook class, maintaining separate arrays for bid and
    ask orders with functionality to initialize, update, and sort
    the orders.
    """

    size: int32
    _warmed_up_: bool_
    _seq_id_: int32
    _asks_: float64[:, :]
    _bids_: float64[:, :]

    def __init__(self, size: int) -> None:
        self.size: int = size
        self._warmed_up_: bool = False
        self._seq_id_: int = 0
        self._asks_: np.ndarray = np.zeros((self.size, 2), dtype=float64)
        self._bids_: np.ndarray = np.zeros((self.size, 2), dtype=float64)

    def recordable(self) -> Dict[str, Union[int, np.ndarray]]:
        """
        Unwraps the internal structures into widely-used Python structures
        for easy recordability (databases, logging, debugging etc).

        Returns
        -------
        Dict
            A dict containing the current state of the orderbook.
        """
        return {
            # "seq_id": self._seq_id_, # TODO: Find a way to output this
            "asks": self._asks_,
            "bids": self._bids_,
        }

    def _reset_(self) -> None:
        """
        Sets all attribute values back to default.
        """
        self._warmed_up_ = False
        self._seq_id_ = 0
        self._asks_.fill(0.0)
        self._bids_.fill(0.0)

    def _sort_bids_(self, bids: np.ndarray) -> None:
        """
        Removes entries with matching prices in update, regardless of size, and then 
        adds non-zero quantity data from update to the book.

        Sorts the bid orders in descending order of price.

        If the best bid is higher than any asks, remove those asks by:
         - Filling the to-be removed arrays with zeros.
         - Rolling it to the back of the orderbook
        """
        removed_old_prices = self._bids_[~nbisin(self._bids_[:, 0], bids[:, 0])]
        new_full_bids = np.vstack((
            removed_old_prices[removed_old_prices[:, 1] != 0.0], 
            bids[bids[:, 1] != 0.0]
        ))

        self._bids_ = new_full_bids[new_full_bids[:, 0].argsort()][::-1][: self.size]

        if self._bids_[0, 0] >= self._asks_[0, 0]:
            overlapping_asks = self._asks_[
                self._asks_[:, 0] <= self._bids_[0, 0]
            ].shape[0]
            self._asks_[:overlapping_asks].fill(0.0)
            self._asks_[:, :] = nbroll(self._asks_, -overlapping_asks, 0)

    def _sort_asks_(self, asks: np.ndarray) -> None:
        """
        Removes entries with matching prices in update, regardless of size, and then 
        adds non-zero quantity data from update to the book.

        Sorts the ask orders in ascending order of price.

        If the best ask is lower than any bids, remove those bids by:
         - Filling the to-be removed arrays with zeros.
         - Rolling it to the back of the orderbook
        """
        removed_old_prices = self._asks_[~nbisin(self._asks_[:, 0], asks[:, 0])]
        new_full_asks = np.vstack((
            removed_old_prices[removed_old_prices[:, 1] != 0.0], 
            asks[asks[:, 1] != 0.0]
        ))
        
        self._asks_ = new_full_asks[new_full_asks[:, 0].argsort()][: self.size]

        if self._asks_[0, 0] <= self._bids_[0, 0]:
            overlapping_bids = self._bids_[
                self._bids_[:, 0] >= self._asks_[0, 0]
            ].shape[0]
            self._bids_[:overlapping_bids].fill(0.0)
            self._bids_[:, :] = nbroll(self._bids_, -overlapping_bids, 0)

    def refresh(self, asks: np.ndarray, bids: np.ndarray, new_seq_id: int) -> None:
        """
        Refreshes the order book with given *complete* ask and bid data and sorts the book.

        Parameters
        ----------
        asks : np.ndarray
            Initial ask orders data, formatted as [[price, size], ...].

        bids : np.ndarray
            Initial bid orders data, formatted as [[price, size], ...].
        """
        assert (
            asks.ndim == 2
            and bids.ndim == 2
            and bids.shape[0] == self.size
            and asks.shape[0] == self.size
        ), "Both arrays must be shape(orderbook.size, 2)"

        self._reset_()

        # Prefer to broadcast onto internal arrays, not overwrite.
        # We also assume that they come in without any overlapping bids/asks,
        # skipping that check and previous array stacking for perf.  
        self._asks_[:, :] = asks[asks[:, 0].argsort()]
        self._bids_[:, :] = bids[bids[:, 0].argsort()[::-1]]

        self._seq_id_ = new_seq_id
        self._warmed_up_ = True

    def update_bids(self, bids: np.ndarray, new_seq_id: int) -> None:
        """
        Updates the current bids with new data.

        Parameters
        ----------
        bids : np.ndarray
            New bid orders data, formatted as [[price, size], ...].
        """
        assert bids.shape[0] > 0 and bids.ndim == 2 and self._warmed_up_ == True

        if new_seq_id > self._seq_id_:
            self._seq_id_ = new_seq_id
            self._sort_bids_(bids)

    def update_asks(self, asks: np.ndarray, new_seq_id: int) -> None:
        """
        Updates the current asks with new data.

        Parameters
        ----------
        asks : np.ndarray
            New ask orders data, formatted as [[price, size], ...].
        """
        assert asks.shape[0] > 0 and asks.ndim == 2 and self._warmed_up_ == True

        if new_seq_id > self._seq_id_:
            self._seq_id_ = new_seq_id
            self._sort_asks_(asks)

    def update_full(self, asks: np.ndarray, bids: np.ndarray, new_seq_id: int) -> None:
        """
        Updates the order book with new ask and bid data.

        Parameters
        ----------
        asks : np.ndarray
            New ask orders data, formatted as [[price, size], ...].

        bids : np.ndarray
            New bid orders data, formatted as [[price, size], ...].
        """
        assert bids.size > 0 and bids.ndim == 2 and asks.size > 0 and asks.ndim == 2 and self._warmed_up_ == True

        if new_seq_id > self._seq_id_:
            self._seq_id_ = new_seq_id
            self._sort_bids_(bids)
            self._sort_asks_(asks)

    def get_vamp(self, depth: float) -> float:
        """
        Calculates the volume-weighted average market price (VAMP) up to a specified depth for both bids and asks.

        Parameters
        ----------
        depth : float
            The depth (in terms of volume) up to which the VAMP is calculated.

        Returns
        -------
        float
            The VAMP, representing an average price weighted by order sizes up to the specified depth.
        """
        if self._warmed_up_:
            # Avoid div 0 error by ensuring orderbook is warm.
            bid_size_weighted_sum = 0.0
            ask_size_weighted_sum = 0.0
            bid_cum_size = 0.0
            ask_cum_size = 0.0

            # Calculate size-weighted sum for bids
            for price, size in self._bids_:
                if bid_cum_size + size > depth:
                    remaining_size = depth - bid_cum_size
                    bid_size_weighted_sum += price * remaining_size
                    bid_cum_size += remaining_size
                    break

                bid_size_weighted_sum += price * size
                bid_cum_size += size

                if bid_cum_size >= depth:
                    break

            # Calculate size-weighted sum for asks
            for price, size in self._asks_:
                if ask_cum_size + size > depth:
                    remaining_size = depth - ask_cum_size
                    ask_size_weighted_sum += price * remaining_size
                    ask_cum_size += remaining_size
                    break

                ask_size_weighted_sum += price * size
                ask_cum_size += size

                if ask_cum_size >= depth:
                    break

            total_size = bid_cum_size + ask_cum_size

            return (bid_size_weighted_sum + ask_size_weighted_sum) / total_size

        return 0.0

    def get_slippage(self, book: np.ndarray, size: float) -> float:
        """
        Calculates the slippage cost for a hypothetical order of a given size, based on either the bid or ask side of the book.

        Parameters
        ----------
        book : np.ndarray
            The order book data for the side (bids or asks) being considered.

        size : float
            The size of the hypothetical order for which slippage is being calculated.

        Returns
        -------
        float
            The slippage cost, defined as the volume-weighted average deviation from the mid price for the given order size.
        """
        if self._warmed_up_:
            # Avoid div 0 error by ensuring orderbook is warm.
            mid_price = self.mid_price
            cum_size = 0.0
            slippage = 0.0

            for level in range(book.shape[0]):
                cum_size += book[level, 1]
                slippage += np.abs(mid_price - book[level, 0]) * book[level, 1]

                if cum_size >= size:
                    slippage /= cum_size
                    break

            return slippage if slippage <= mid_price else mid_price

        return 0.0

    @property
    def bids(self) -> np.ndarray[float]:
        return self._bids_

    @property
    def asks(self) -> np.ndarray[float]:
        return self._asks_

    @property
    def seq_id(self) -> int:
        return self._seq_id_

    @property
    def is_empty(self) -> bool:
        return np.all(self._bids_ == 0.0) and np.all(self._asks_ == 0.0)

    @property
    def mid_price(self) -> float:
        """
        Calculates the mid price of the order book based on the best bid and ask prices.

        Returns
        -------
        float
            The mid price, which is the average of the best bid and best ask prices.
        """
        if self._warmed_up_:
            # Avoid div 0 error by ensuring orderbook is warm.
            return (self._bids_[0, 0] + self._asks_[0, 0]) / 2.0
        else:
            return 0.0

    @property
    def wmid_price(self) -> float:
        """
        Calculates the weighted mid price of the order book, considering the volume imbalance
        between the best bid and best ask.

        Returns
        -------
        float
            The weighted mid price, which accounts for the volume imbalance at the top of the book.
        """
        if self._warmed_up_:
            # Avoid div 0 error by ensuring orderbook is warm.
            bid_price, bid_size = self._bids_[0]
            ask_price, ask_size = self._asks_[0]
            imb = bid_size / (bid_size + ask_size)
            return (bid_price * imb) + (ask_price * (1.0 - imb))
        else:
            return 0.0

    @property
    def bid_ask_spread(self) -> float:
        """
        Calculates the current spread of the order book.

        Returns
        -------
        float
            The spread, defined as the difference between the best ask and the best bid prices.
        """
        return self._asks_[0, 0] - self._bids_[0, 0]

    def __eq__(self, orderbook: "Orderbook") -> bool:
        assert isinstance(orderbook, Orderbook)
        return (
            orderbook._seq_id_ == self._seq_id_
            and np.array_equal(orderbook._bids_, self._bids_)
            and np.array_equal(orderbook._asks_, self._asks_)
        )

    def __len__(self) -> int:
        return min(
            self._bids_[self._bids_[:, 0] != 0.0].shape[0],
            self._asks_[self._asks_[:, 0] != 0.0].shape[0],
        )

    def __str__(self) -> str:
        return (
            f"Orderbook(size={self.size}, "
            f"seq_id={self.seq_id}, "
            f"bids={self._bids_}, "
            f"asks={self._asks_}"
        )
