import numpy as np
from numba.types import int32, float64, bool_
from numba.experimental import jitclass
from typing import Tuple

from mm_toolbox.numba import nbisin, nbroll


@jitclass
class Orderbook:
    """
    An orderbook class, maintaining separate arrays for bid and
    ask orders with functionality to initialize, update, and sort
    the orders.
    """

    size: int32

    _warmed_up: bool_
    _seq_id: int32
    _asks: float64[:, :]
    _bids: float64[:, :]

    def __init__(self, size: int) -> None:
        self.size: int = size

        self._warmed_up: bool = False
        self._seq_id: int = 0
        self._asks: np.ndarray = np.zeros((size, 2), dtype=float64)
        self._bids: np.ndarray = np.zeros((size, 2), dtype=float64)

    def recordable(self) -> Tuple[int, np.ndarray, np.ndarray]:
        """
        Unwraps the internal structures into widely-used Python structures
        for easy recordability (databases, logging, debugging etc).

        Returns
        -------
        Dict
            A dict containing the current state of the orderbook.
        """
        return (self._seq_id, self._asks, self._bids)

    def _sort_bids(self, bids: np.ndarray) -> None:
        """
        Removes entries with matching prices in update, regardless of size, and then
        adds non-zero quantity data from update to the book.

        Sorts the bid orders in descending order of price.

        If the best bid is higher than any asks, remove those asks by:
         - Filling the to-be removed arrays with zeros.
         - Rolling it to the back of the orderbook.
        """
        removed_old_prices = self._bids[~nbisin(self._bids[:, 0], bids[:, 0])]
        new_full_bids = np.vstack(
            (
                removed_old_prices[
                    removed_old_prices[:, 1] != 0.0  # Re-remove zeros incase of overlap
                ],
                bids[bids[:, 1] != 0.0],
            )
        )

        self._bids = new_full_bids[new_full_bids[:, 0].argsort()][::-1][: self.size]

        # Remove overlapping asks.
        if self._bids[0, 0] >= self._asks[0, 0]:
            overlapping_asks = self._asks[self._asks[:, 0] <= self._bids[0, 0]].shape[0]
            self._asks[:overlapping_asks].fill(0.0)
            self._asks[:, :] = nbroll(self._asks, -overlapping_asks, 0)

    def _sort_asks(self, asks: np.ndarray) -> None:
        """
        Removes entries with matching prices in update, regardless of size, and then
        adds non-zero quantity data from update to the book.

        Sorts the ask orders in ascending order of price.

        If the best ask is lower than any bids, remove those bids by:
         - Filling the to-be removed arrays with zeros.
         - Rolling it to the back of the orderbook.
        """
        removed_old_prices = self._asks[~nbisin(self._asks[:, 0], asks[:, 0])]
        new_full_asks = np.vstack(
            (
                removed_old_prices[
                    removed_old_prices[:, 1] != 0.0  # Re-remove zeros incase of overlap
                ],
                asks[asks[:, 1] != 0.0],
            )
        )

        self._asks = new_full_asks[new_full_asks[:, 0].argsort()][: self.size]

        # Remove overlapping bids.
        if self._asks[0, 0] <= self._bids[0, 0]:
            overlapping_bids = self._bids[self._bids[:, 0] >= self._asks[0, 0]].shape[0]
            self._bids[:overlapping_bids].fill(0.0)
            self._bids[:, :] = nbroll(self._bids, -overlapping_bids, 0)

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
        ), (
            f"Both input arrays must be 2D and at least of size {self.size}, "
            f"but got size {bids.shape[0]} and {asks.shape[0]}"
        )

        # Reset attributes and internal arrays.
        self._warmed_up = False
        self._seq_id = 0
        self._asks.fill(0.0)
        self._bids.fill(0.0)

        # Prefer to broadcast onto internal arrays, not overwrite.
        # We also assume that they come in without any overlapping bids/asks,
        # skipping that check and previous array stacking for perf.
        self._asks[:, :] = asks[asks[:, 0].argsort()]
        self._bids[:, :] = bids[bids[:, 0].argsort()[::-1]]

        self._seq_id = new_seq_id
        self._warmed_up = True

    def update_bbo(
        self,
        bid_price: float,
        bid_size: float,
        ask_price: float,
        ask_size: float,
        new_seq_id: int,
    ) -> None:
        """
        Updates the current orderbook with new best bid ask data.
        """
        assert (
            self._warmed_up
        ), "Orderbook must be warmed up (initialized) before attempting to update."

        if new_seq_id <= self._seq_id:
            return

        self._seq_id = new_seq_id

        best_bid_price = self._bids[0, 0]
        best_ask_price = self._asks[0, 0]

        # Matching bid price, update size.
        if bid_price == best_bid_price:
            if bid_size == 0.0:
                # Most feeds don't send size 0 through the BBA
                # feed, but just incase, this case is included.
                # It is inefficient due to the realloc, but wont
                # be optimized further.
                self._bids = self._bids[1:]
            else:
                self._bids[0, 1] = bid_size

        # Higher bid price, insert ontop then solve ask overlaps.
        elif bid_price > best_bid_price:
            self._bids = nbroll(self._bids, 1, 0)
            self._bids[0, 0] = bid_price
            self._bids[0, 1] = bid_size

            # Remove overlapping asks (identical to self._sort_bids())
            if self._bids[0, 0] >= self._asks[0, 0]:
                overlapping_asks = self._asks[
                    self._asks[:, 0] <= self._bids[0, 0]
                ].shape[0]
                self._asks[:overlapping_asks].fill(0.0)
                self._asks[:, :] = nbroll(self._asks, -overlapping_asks, 0)

        # Matching ask price, update size.
        if ask_price == best_ask_price:
            if ask_size == 0.0:
                # Most feeds don't send size 0 through the BBA
                # feed, but just incase, this case is included.
                # It is inefficient due to the realloc, but wont
                # be optimized further.
                self._asks = self._asks[1:]
            else:
                self._asks[0, 1] = ask_size

        # Lower ask price, insert ontop then solve bid overlaps.
        elif ask_price < best_ask_price:
            self._asks = nbroll(self._asks, 1, 0)
            self._asks[0, 0] = ask_price
            self._asks[0, 1] = ask_size

            # Remove overlapping bids (identical to self._sort_asks()).
            if self._asks[0, 0] <= self._bids[0, 0]:
                overlapping_bids = self._bids[
                    self._bids[:, 0] >= self._asks[0, 0]
                ].shape[0]
                self._bids[:overlapping_bids].fill(0.0)
                self._bids[:, :] = nbroll(self._bids, -overlapping_bids, 0)

    def update_bids(self, bids: np.ndarray, new_seq_id: int) -> None:
        """
        Updates the current bids with new data.

        Parameters
        ----------
        bids : np.ndarray
            New bid orders data, formatted as [[price, size], ...].
        """
        assert (
            self._warmed_up
        ), "Orderbook must be warmed up (initialized) before attempting to update."
        assert bids.shape[0] > 0 and bids.ndim == 2, (
            f"Input array must be 2D and at least of size 1, "
            f"but got size {bids.shape[0]}"
        )

        if new_seq_id > self._seq_id:
            self._seq_id = new_seq_id
            self._sort_bids(bids)

    def update_asks(self, asks: np.ndarray, new_seq_id: int) -> None:
        """
        Updates the current asks with new data.

        Parameters
        ----------
        asks : np.ndarray
            New ask orders data, formatted as [[price, size], ...].
        """
        assert (
            self._warmed_up
        ), "Orderbook must be warmed up (initialized) before attempting to update."
        assert asks.ndim == 2 and asks.shape[0] > 0, (
            f"Input array must be 2D and at least of size 1, "
            f"but got size {asks.shape[0]}"
        )

        if new_seq_id > self._seq_id:
            self._seq_id = new_seq_id
            self._sort_asks(asks)

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
        assert (
            self._warmed_up
        ), "Orderbook must be warmed up (initialized) before attempting to update."
        assert bids.size > 0 and bids.ndim == 2 and asks.size > 0 and asks.ndim == 2, (
            f"Both input arrays must be 2D and at least of size 1, "
            f"but got size {bids.shape[0]} and {asks.shape[0]}"
        )

        if new_seq_id > self._seq_id:
            self._seq_id = new_seq_id
            self._sort_bids(bids)
            self._sort_asks(asks)

    def get_vamp(self, dollar_depth: float) -> float:
        """
        Calculates the volume-weighted average market price (VAMP) up to a specified depth for both bids and asks.

        Parameters
        ----------
        dollar_depth : float
            The depth (in terms of dollars) up to which the VAMP is calculated.

        Returns
        -------
        float
            The VAMP, representing an average price weighted by order sizes up to the specified depth.
        """
        assert self._warmed_up, "Orderbook is not warmed up."

        bid_size_weighted_sum = 0.0
        ask_size_weighted_sum = 0.0
        bid_cum_size = 0.0
        ask_cum_size = 0.0

        # Calculate size-weighted sum for bids
        for price, size in self._bids:
            if bid_cum_size + size > dollar_depth:
                remaining_size = dollar_depth - bid_cum_size
                bid_size_weighted_sum += price * remaining_size
                bid_cum_size += remaining_size
                break

            bid_size_weighted_sum += price * size
            bid_cum_size += size

            if bid_cum_size >= dollar_depth:
                break

        # Calculate size-weighted sum for asks
        for price, size in self._asks:
            if ask_cum_size + size > dollar_depth:
                remaining_size = dollar_depth - ask_cum_size
                ask_size_weighted_sum += price * remaining_size
                ask_cum_size += remaining_size
                break

            ask_size_weighted_sum += price * size
            ask_cum_size += size

            if ask_cum_size >= dollar_depth:
                break

        total_size = bid_cum_size + ask_cum_size

        return (bid_size_weighted_sum + ask_size_weighted_sum) / total_size

    def get_imbalance(self, depth_bps: float) -> float:
        """
        Calculates the size imbalance at a given depth.

        Parameters
        ----------
        depth_bps : float
            The depth of imbalance desired in each orderbook, in basis points.

        Returns
        -------
        float
            The imbalance, given by the ratio between bid/ask volume.
        """
        assert self._warmed_up, "Orderbook is not warmed up."

        depth_decimal = depth_bps / 10_000.0
        mid_price = self.mid_price
        max_bid = mid_price - (mid_price * depth_decimal)
        max_ask = mid_price + (mid_price * depth_decimal)
        bid_cum_size = self._bids[self._bids[:, 0] >= max_bid][:, 1].sum()
        ask_cum_size = self._asks[self._asks[:, 0] <= max_ask][:, 1].sum()

        return bid_cum_size / ask_cum_size

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
        assert self._warmed_up, "Orderbook is not warmed up."

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

    @property
    def bids(self) -> np.ndarray:
        return self._bids

    @property
    def asks(self) -> np.ndarray:
        return self._asks

    @property
    def seq_id(self) -> int:
        return self._seq_id

    @property
    def is_empty(self) -> bool:
        return np.all(self._bids == 0.0) and np.all(self._asks == 0.0)

    @property
    def mid_price(self) -> float:
        """
        Calculates the mid price of the order book based on the best bid and ask prices.

        Returns
        -------
        float
            The mid price, which is the average of the best bid and best ask prices.
        """
        assert self._warmed_up, "Orderbook is not warmed up."
        return (self._bids[0, 0] + self._asks[0, 0]) / 2.0

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
        assert self._warmed_up, "Orderbook is not warmed up."

        bid_price, bid_size = self._bids[0]
        ask_price, ask_size = self._asks[0]
        imb = bid_size / (bid_size + ask_size)
        return (bid_price * imb) + (ask_price * (1.0 - imb))

    @property
    def bid_ask_spread(self) -> float:
        """
        Calculates the current spread of the order book.

        Returns
        -------
        float
            The spread, defined as the difference between the best ask and the best bid prices.
        """
        assert self._warmed_up, "Orderbook is not warmed up."
        return self._asks[0, 0] - self._bids[0, 0]

    def __eq__(self, orderbook: "Orderbook") -> bool:
        assert isinstance(orderbook, Orderbook)
        return (
            orderbook._seq_id == self._seq_id
            and np.array_equal(orderbook._bids, self._bids)
            and np.array_equal(orderbook._asks, self._asks)
        )

    def __len__(self) -> int:
        return min(
            self._bids[self._bids[:, 0] != 0.0].shape[0],
            self._asks[self._asks[:, 0] != 0.0].shape[0],
        )

    def __str__(self) -> str:
        return (
            f"Orderbook(size={self.size}, "
            f"seq_id={self.seq_id}, "
            f"bids={self._bids}, "
            f"asks={self._asks}"
        )
