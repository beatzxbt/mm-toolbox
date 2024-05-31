import numpy as np
from numba import njit
from numba.types import int32, float32, bool_, Array
from numba.experimental import jitclass
from typing import Tuple, Union


@njit(["int32(int32[:, :], int32)"], fastmath=True)
def linear_search(arr: Array, price: int) -> int:
    """
    Performs a linear search on the order book array to find the level that matches the specified price.

    Parameters
    ----------
    arr : Array
        The bid or ask array where each row represents a price level, typically used for ask arrays.

    price : int
        The price level to search for in the order book.

    Returns
    -------
    int
        The index of the matching price level, or n+1 if the price is not found (indicative of an error condition).
    """
    n = arr.shape[0]
    for i in range(0, n):
        if arr[i, 0] == price:
            return i

    return n + 1

@njit(["int32(int32[:, :], int32)"], fastmath=True)
def linear_search_reversed(arr: Array, price: int) -> int:
    """
    Performs a linear search on the order book array starting from the end to find the level that matches the specified price.

    Parameters
    ----------
    arr : Array
        The bid or ask array where each row represents a price level, typically used for bid arrays.

    price : int
        The price level to search for in the order book.

    Returns
    -------
    int
        The index of the matching price level, or n+1 if the price is not found (indicative of an error condition).
    """
    n = arr.shape[0]
    for i in range(n - 1, 0, -1):
        if arr[i, 0] == price:
            return i

    return n + 1

@njit(["void(int32[:, :], int32, int32)"], error_model="numpy", fastmath=True)
def process_single_l2_bid(bids: Array, price: int, size: int) -> None:
    """
    Processes a single level 2 update for bids.

    Steps:
    1. Identify the best and worst price levels in the current bids.
    2. Determine the type of update based on the incoming price:
       a. If the incoming price matches the best bid price, update or remove the level if the size is zero.
       b. If the incoming price is less than the worst bid price, ignore the update.
       c. If the incoming price is between the worst and the best prices, find the exact level to update the size.
       d. If the incoming price is better than the best price, adjust the order book to accommodate new price levels.

    Parameters
    ----------
    bids : Array
        The bids array where each row represents a price level.

    price : int
        The price at which the update is happening.

    size : int
        The new size for the price level, which can be zero to indicate removal of the price level.

    Returns
    -------
    None
    """
    best_price = bids[-1, 0]
    worst_price = bids[0, 0]

    # Check (a)
    if price == best_price:
        if size == 0:
            bids[1:] = bids[:-1]
            bids[0, 0] = worst_price - 1
            bids[0, 1] = 0
        else:
            bids[-1, 1] = size

        return None

    # Check (b)
    elif price < worst_price:
        return None
    
    # Check (c)
    elif price < best_price:
        idx = linear_search_reversed(bids, price)
        bids[idx, 1] = size
        return None
    
    # Check (d)
    elif price > best_price:
        num_new_levels = price - best_price
        bids[:-num_new_levels] = bids[num_new_levels:]
        start_idx = bids.shape[0] - num_new_levels
        for i in range(1, num_new_levels + 1):
            idx = start_idx + i
            bids[idx, 0] = best_price + i
            bids[idx, 1] = 0
        bids[-1, 1] = size 
        return None

@njit(["void(int32[:, :], int32, int32)"], error_model="numpy", fastmath=True)
def process_single_l2_ask(asks: Array, price: int, size: int) -> None:
    """
    Processes a single level 2 update for asks.

    Steps:
    1. Identify the best and worst price levels in the current asks.
    2. Based on the incoming price, determine the appropriate action:
       a. If the incoming price matches the best ask price, update or remove the level if the size is zero.
       b. If the incoming price is less than the worst ask price, ignore the update.
       c. If the incoming price is between the worst and the best prices, find the exact level to update the size.
       d. If the incoming price is better than the best price, adjust the order book to accommodate new price levels.

    Parameters
    ----------
    asks : Array
        The asks array where each row represents a price level.

    price : int
        The price at which the update is happening.

    size : int
        The new size for the price level, which can be zero to indicate removal of the price level.

    Returns
    -------
    None
    """
    worst_price = asks[-1, 0]
    best_price = asks[0, 0]

    # Step (a)
    if price == best_price:
        if size == 0:
            asks[:-1] = asks[1:]
            asks[-1, 0] = worst_price + 1
            asks[-1, 1] = 0
        else:
            asks[0, 1] = size

        return None

    # Step (b)
    elif price > worst_price:
        return None
    
    # Step (c)
    elif price > best_price:
        idx = linear_search(asks, price)
        asks[idx, 1] = size
        return None
    
    # Step (d)
    elif price < best_price:
        num_new_levels = best_price - price
        asks[num_new_levels:] = asks[:-num_new_levels]
        for i in range(1, num_new_levels + 1):
            idx = num_new_levels - i
            asks[idx, 0] = best_price - i
            asks[idx, 1] = 0
        asks[0, 1] = size 
        return None
    
@njit(["void(int32[:, :], int32[:, :], bool_, int32, int32)"], error_model="numpy", fastmath=True)
def process_single_trade(asks: Array, bids: Array, isBuy: bool, price: int, size: int):
    """
    Process a single trade, updating the order book for either an ask or a bid based on the trade direction.

    Steps:
    1. Determine if the trade is a buy or a sell:
       a. For a buy trade, adjust the best ask level:
          i. If the trade size equals or exceeds the best ask size, remove the ask level.
          ii. Otherwise, decrement the best ask size by the trade size.
       b. For a sell trade, adjust the best bid level similarly.

    Parameters
    ----------
    asks : Array
        The current ask levels.

    bids : Array
        The current bid levels.

    isBuy : bool
        True if the trade is a buy, False if a sell.

    price : int
        The price at which the trade occurred.

    size : int
        The size of the trade.

    Returns
    -------
    None
    """
    if isBuy:
        best_ask_price, best_ask_size = asks[0]
        
        # Step (1a)
        if price == best_ask_price:
            if size >= best_ask_size:
                new_worst_price = asks[-1, 0] + 1
                asks[:-1] = asks[1:]
                asks[-1, 0] = new_worst_price
                asks[-1, 1] = 0
            else:
                asks[0, 1] -= size

    else:
        best_bid_price, best_bid_size = bids[-1]
        
        # Step (1b)
        if price == best_bid_price:
            if size >= best_bid_size:
                new_worst_price = bids[0, 0] - 1
                bids[1:] = bids[:-1]
                bids[0, 0] = new_worst_price
                bids[0, 1] = 0
            else:
                bids[-1, 1] -= size
          

@jitclass
class Orderbook:
    tick_size: float32
    lot_size: float32
    num_levels: int32
    asks: int32[:, :]
    bids: int32[:, :]
    last_updated_timestamp: int32
    warmed_up: bool_

    def __init__(self, tick_size: float, lot_size: float, num_levels: int) -> None:
        self.tick_size = tick_size
        self.lot_size = lot_size
        self.num_levels = num_levels
        self.asks = np.zeros((self.num_levels, 2), dtype=int32)
        self.bids = np.zeros((self.num_levels, 2), dtype=int32)
        self.last_updated_timestamp = 0
        self.warmed_up = False

    def reset(self) -> None:
        self.asks = np.zeros_like(self.asks)
        self.bids = np.zeros_like(self.bids)
        self.last_updated_timestamp = 0
        self.warmed_up = False

    def display_internal(self, levels: int) -> None:
        """
        Displays the top X bid/ask levels of the order book.

        Parameters
        ----------
        levels : int
            The number of levels to display from both the ask and bid sides.
        """
        if levels > self.num_levels:
            print(f"Too many levels! Max levels are: {self.num_levels}.")
            return None

        last_asks = self.asks[::-1][-levels:]
        adjusted_asks_prices = last_asks[:, 0] # * float(self.tick_size)
        adjusted_asks_sizes = last_asks[:, 1] # * float(self.lot_size)

        first_bids = self.bids[::-1][:levels]
        adjusted_bids_prices = first_bids[:, 0] # * float(self.tick_size)
        adjusted_bids_sizes = first_bids[:, 1] # * float(self.lot_size)

        ask_str = "Asks: |" + "\n      |".join([
            f"Price: {price}, Size: {size}"
            for price, size in zip(adjusted_asks_prices, adjusted_asks_sizes)
        ])

        bid_str = "Bids: |" + "\n      |".join([
            f"Price: {price}, Size: {size}"
            for price, size in zip(adjusted_bids_prices, adjusted_bids_sizes)
        ])

        return print(f"{ask_str}\n{'-' * 40}\n{bid_str}")

    def normalize(self, num: float, step: float) -> int:
        """
        Normalizes a number by dividing it by a step value and rounding to the nearest integer.

        Parameters
        ----------
        num : float
            The number to be normalized.

        step : float
            The step size used for normalization.

        Returns
        -------
        int
            The normalized integer value.
        """
        return round(num/step)
    
    def denormalize(self, num: Union[int, Array], step: Union[float, Array]) -> Union[float, Array]:
        """
        Denormalizes a number or an array of numbers by multiplying with a step value or an array of step values.

        Parameters
        ----------
        num : int or Array
            The integer or array of integers to be denormalized.
            
        step : float or Array
            The step size or array of step sizes used for denormalization.

        Returns
        -------
        float or Array
            The denormalized float value or array of float values.
        """
        return num * step

    def denormalize_book(self, orderbook: Array) -> Array:
        """
        Denormalizes the order book array where the first column represents prices and the second column represents sizes.

        Parameters
        ----------
        orderbook : Array
            The (x, 2) sized array representing the order book with the first column as prices and the second column as sizes.

        Returns
        -------
        Array
            The denormalized order book.
        """
        denormalize_book = orderbook.copy()
        denormalize_book[:, 0] = self.denormalize(denormalize_book[:, 0], self.tick_size)
        denormalize_book[:, 1] = self.denormalize(denormalize_book[:, 1], self.lot_size)
        return denormalize_book

    def warmup_asks(self, best_ask_price: float, best_ask_size: float) -> None:
        """
        Initializes the ask side of the order book with a linear spread of prices starting
        from the best ask price. The first level is set to the provided best ask size, with
        other levels initialized to zero.

        Parameters
        ----------
        best_ask_price : float
            The lowest asking price to start the order book levels.

        best_ask_size : float
            The size (quantity) available at the best ask price.

        Returns
        -------
        None
        """
        start = self.normalize(best_ask_price, self.tick_size)
        stop = start + self.num_levels
        self.asks[:, 0] = np.arange(start, stop, 1)
        self.asks[0, 1] = self.normalize(best_ask_size, self.lot_size)
    
    def warmup_bids(self, best_bid_price: float, best_bid_size: float) -> None:
        """
        Initializes the bid side of the order book with a linear spread of prices starting
        from the best bid price downwards. The last level is set to the provided best bid size,
        with other levels initialized to zero.

        Parameters
        ----------
        best_bid_price : float
            The highest bid price to start the order book levels.
            
        best_bid_size : float
            The size (quantity) available at the best bid price.

        Returns
        -------
        None
        """
        start = self.normalize(best_bid_price, self.tick_size)
        stop = start - self.num_levels
        self.bids[:, 0] = np.arange(start, stop, -1)[::-1]
        self.bids[-1, 1] = self.normalize(best_bid_size, self.lot_size)

    def ingest_l2_update(self, timestamp: int, asks: Array, bids: Array) -> None:
        """
        Processes Level 2 market data updates by updating the asks and bids in the order book.

        Parameters
        ----------
        timestamp : int
            The timestamp of the update, used to ensure updates are processed in order.

        asks : Array
            An array of new ask prices and sizes.

        bids : Array
            An array of new bid prices and sizes.
        """
        if not self.warmed_up:
            best_ask = asks[asks[:, 0] == np.min(asks[:, 0])][0]
            best_bid = bids[bids[:, 0] == np.max(bids[:, 0])][0]
            self.warmup_asks(best_ask[0], best_ask[1])
            self.warmup_bids(best_bid[0], best_bid[1])
            self.last_updated_timestamp = timestamp
            self.warmed_up = True
            return None
        
        if timestamp < self.last_updated_timestamp:
            return None
        
        self.last_updated_timestamp = timestamp

        try:
            for price, size in bids:
                process_single_l2_bid(
                    bids=self.bids, 
                    price=self.normalize(price, self.tick_size), 
                    size=self.normalize(size, self.lot_size), 
                )
                # print(f"Updated bid: {(price, size)}...")

            for price, size in asks:
                process_single_l2_ask(
                    asks=self.asks, 
                    price=self.normalize(price, self.tick_size), 
                    size=self.normalize(size, self.lot_size), 
                )
                # print(f"Updated ask: {(price, size)}...")

        except Exception as e:
            print(f"\nError: {e}")
            print(f"Attempted to update {(price, size)}...")
            print(f"Current BBA :{(self.get_best_bid(), self.get_best_ask())}...")
            self.display_internal(self.num_levels)
            raise e

    def ingest_trade_update(self, timestamp: int, isBuy: bool, price: float, size: float) -> None:
        """
        Processes trade updates for the order book.

        Parameters
        ----------
        timestamp : int
            The timestamp of the trade update.

        isBuy : bool
            True if the trade is a buy; False if it is a sell.

        price : float
            The price at which the trade occurred.

        size : float
            The size of the trade.
        """
        if not self.warmed_up:
            return None
        
        if timestamp < self.last_updated_timestamp:
            return None
        
        self.last_updated_timestamp = timestamp

        try:
            process_single_trade(
                asks=self.asks, 
                bids=self.bids, 
                isBuy=isBuy,
                price=self.normalize(price, self.tick_size),
                size=self.normalize(size, self.lot_size)
            )
            # print(f"Updated trade: {(price, size)}...")

        except Exception as e:
            print(f"\nError! Find below why you are a moron...\n")
            print(f"Attempted to update {(price, size)}...")
            print(f"Current BBA :{(self.get_best_bid(), self.get_best_ask())}...")
            self.display_internal(self.num_levels)
            raise e

    def get_best_bid(self) -> Tuple[float, float]:
        """
        Retrieves the best bid price and size from the order book.

        Returns
        -------
        Tuple[float, float]
            A tuple containing the best bid price and size, both denormalized.
        """
        return (
            self.denormalize(self.bids[-1, 0], self.tick_size), 
            self.denormalize(self.bids[-1, 1], self.lot_size)
        )
    
    def get_best_ask(self) -> Tuple[float, float]:
        """
        Retrieves the best ask price and size from the order book.

        Returns
        -------
        Tuple[float, float]
            A tuple containing the best ask price and size, both denormalized.
        """
        return (
            self.denormalize(self.asks[0, 0], self.tick_size), 
            self.denormalize(self.asks[0, 1], self.lot_size)
        )

    def get_mid(self) -> float:
        """
        Calculates the mid price of the order book based on the best bid and ask prices.

        Returns
        -------
        float
            The mid price, which is the average of the best bid and best ask prices.
        """
        mid = (self.bids[-1, 0] + self.asks[0, 0]) / 2
        return self.denormalize(mid, self.tick_size)

    def get_wmid(self) -> float:
        """
        Calculates the weighted mid price of the order book, considering the volume imbalance 
        between the best bid and best ask.

        Returns
        -------
        float
            The weighted mid price, which accounts for the volume imbalance at the top of the book.
        """
        imb = self.bids[-1, 1] / (self.bids[-1, 1] + self.asks[0, 1])
        wmid = self.bids[-1, 0] * imb + self.asks[0, 0] * (1 - imb)
        return self.denormalize(wmid, self.tick_size)
    
    def get_vamp(self, dollar_depth: float) -> float:
        """
        Calculates the volume-weighted average market price (VAMP) up to a specified dollar depth for both bids and asks.

        Parameters
        ----------
        dollar_depth : float
            The dollar depth (in terms of total order value) up to which the VAMP is calculated.

        Returns
        -------
        float
            The VAMP, representing an average price weighted by order sizes up to the specified dollar depth.
        """
        bid_dollar_weighted_sum = 0
        ask_dollar_weighted_sum = 0
        bid_dollar_cum = 0
        ask_dollar_cum = 0
        
        # Indexed backwards for best -> worst order
        for price, size in self.denormalize_book(self.bids[::-1]): 
            order_value = price * size
            if bid_dollar_cum + order_value > dollar_depth:
                remaining_value = dollar_depth - bid_dollar_cum
                bid_dollar_weighted_sum += remaining_value
                bid_dollar_cum += remaining_value
                break

            bid_dollar_weighted_sum += order_value
            bid_dollar_cum += order_value
            
            if bid_dollar_cum >= dollar_depth:
                break
        
        for price, size in self.denormalize_book(self.asks):
            order_value = price * size
            if ask_dollar_cum + order_value > dollar_depth:
                remaining_value = dollar_depth - ask_dollar_cum
                ask_dollar_weighted_sum += remaining_value
                ask_dollar_cum += remaining_value
                break

            ask_dollar_weighted_sum += order_value
            ask_dollar_cum += order_value

            if ask_dollar_cum >= dollar_depth:
                break

        total_dollar = bid_dollar_cum + ask_dollar_cum

        if total_dollar == 0:
            return 0.0
        
        return (bid_dollar_weighted_sum + ask_dollar_weighted_sum) / total_dollar

    def get_spread(self) -> float:
        """
        Calculates the current spread of the order book.

        Returns
        -------
        float
            The spread, defined as the difference between the best ask and the best bid prices.
        """
        return self.denormalize(self.asks[0, 0] - self.bids[-1, 0], self.tick_size)
    
    def get_slippage(self, book: Array, dollar_depth: float) -> float:
        """
        Calculates the slippage cost for a hypothetical order of a given dollar depth, based on either the bid or ask side of the book.

        Parameters
        ----------
        book : Array
            The order book data for the side (bids or asks) being considered.

        dollar_depth : float
            The dollar depth of the hypothetical order for which slippage is being calculated.

        Returns
        -------
        float
            The slippage cost, defined as the volume-weighted average deviation from the mid price for the given order depth in dollars.
        """
        mid = self.get_mid()
        dollar_cum = 0.0
        slippage = 0.0
        denormalize_book = self.denormalize_book(book)

        for price, size in denormalize_book:
            order_value = price * size
            dollar_cum += order_value
            slippage += np.abs(mid - price) * order_value

            if dollar_cum >= dollar_depth:
                slippage /= dollar_cum
                return slippage

        # If the full depth wasn't reached (i.e., the book is not deep enough), return NaN
        return np.nan