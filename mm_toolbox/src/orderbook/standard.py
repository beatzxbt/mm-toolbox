import numpy as np
from numba import njit
from numba.types import int32, float64, bool_
from numba.experimental import jitclass
from typing import Dict, Union

@njit(["bool_[:](float64[:], float64[:])"], inline="always")
def isin(a: np.ndarray[float], b: np.ndarray[float]) -> np.ndarray[bool]:
    out_len = a.size
    out = np.empty(out_len, dtype=bool_)
    b_set = set(b)

    for i in range(out_len):
        out[i] = a[i] in b_set

    return out

@jitclass
class Orderbook:
    """
    An orderbook class, maintaining separate arrays for bid and
    ask orders with functionality to initialize, update, and sort
    the orders.

    Attributes
    ----------
    size : int
        The maximum number of bid/ask pairs the order book will hold.

    asks : np.ndarray
        Array to store ask orders, each with price and quantity.

    bids : np.ndarray
        Array to store bid orders, each with price and quantity.
    """

    size: int32
    seq_id: int32
    asks: float64[:, :]
    bids: float64[:, :]

    def __init__(self, size: int) -> None:
        """
        Constructs all the necessary attributes for the orderbook object.

        Parameters
        ----------
        size : int
            Size of the order book (number of orders to store).
        """
        self.size: int = size
        self._seq_id_: int = 0
        self._asks_: np.ndarray = np.zeros((self.size, 2), dtype=float64)
        self._bids_: np.ndarray = np.zeros((self.size, 2), dtype=float64)
    
    def reset(self) -> None:
        """
        Sets all attribute values back to 0 
        """
        self._seq_id_ = 0
        self._asks_.fill(0.0)
        self._bids_.fill(0.0)

    def recordable(self) -> Dict[str, Union[float, np.ndarray]]:
        """
        Unwraps the internal structures into widely-used Python structures
        for easy recordability (databases, logging, debugging etc). 

        Returns
        -------
        Dict
            A dict containing the current state of the orderbook.
        """
        return {
            "seq_id": np.float64(self._seq_id_),
            "asks": self._asks_.astype(np.float64),
            "bids": self._bids_.astype(np.float64)
        }

    def sort_bids(self) -> None:
        """
        Sorts the bid orders in descending order of price and updates the best bid.
        """
        self._bids_ = self._bids_[self._bids_[:, 0].argsort()][::-1][: self.size]

    def sort_asks(self) -> None:
        """
        Sorts the ask orders in ascending order of price and updates the best ask.
        """
        self._asks_ = self._asks_[self._asks_[:, 0].argsort()][: self.size]

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
        self.reset()

        self._seq_id_ = new_seq_id
        
        max_asks_idx = min(asks.shape[0], self.size)
        max_bids_idx = min(bids.shape[0], self.size)

        self._asks_[:max_asks_idx, :] = asks[:max_asks_idx, :]
        self._bids_[:max_bids_idx, :] = bids[:max_bids_idx, :]
        self.sort_bids()
        self.sort_asks()

    def update_bids(self, bids: np.ndarray, new_seq_id: int) -> None:
        """
        Updates the current bids with new data. Removes entries with matching
        prices in update, regardless of size, and then adds non-zero quantity
        data from update to the book.

        Parameters
        ----------
        bids : np.ndarray
            New bid orders data, formatted as [[price, size], ...].
        """
        if bids.size > 0 and new_seq_id > self._seq_id_:
            self._seq_id_ = new_seq_id
            self._bids_ = self._bids_[~isin(self._bids_[:, 0], bids[:, 0])]
            self._bids_ = np.vstack((self._bids_, bids[bids[:, 1] != 0]))
            self.sort_bids()

    def update_asks(self, asks: np.ndarray, new_seq_id: int) -> None:
        """
        Updates the current asks with new data. Removes entries with matching
        prices in update, regardless of size, and then adds non-zero quantity
        data from update to the book.

        Parameters
        ----------
        asks : np.ndarray
            New ask orders data, formatted as [[price, size], ...].
        """
        if asks.size > 0 and new_seq_id > self._seq_id_:
            self._seq_id_ = new_seq_id
            self._asks_ = self._asks_[~isin(self._asks_[:, 0], asks[:, 0])]
            self._asks_ = np.vstack((self._asks_, asks[asks[:, 1] != 0]))
            self.sort_asks()

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
        self.update_asks(asks, new_seq_id)
        self.update_bids(bids, new_seq_id)

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
        bid_size_weighted_sum = 0.0
        ask_size_weighted_sum = 0.0
        bid_cum_size = 0.0
        ask_cum_size = 0.0

        # Calculate size-weighted sum for bids
        for price, size in self._bids_[::-1]:
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

        if total_size == 0.0:
            return 0.0

        return (bid_size_weighted_sum + ask_size_weighted_sum) / total_size

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
        return self._bids_
    
    @property
    def asks(self) -> np.ndarray:
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
        return (self._bids_[-1, 0] + self._asks_[0, 0]) / 2.0
    
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
        bid_price, bid_size = self._bids_[-1]
        ask_price, ask_size = self._asks_[0]
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
        return self._bids_[-1, 0] - self._asks_[0, 0]
