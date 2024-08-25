# class HFTOrderbook:
#     raise NotImplementedError(f"Still working on this one!")

# import numpy as np
# from typing import Tuple
# from numba import njit
# from numba.types import uint32, float64, bool_
# from numba.experimental import jitclass

# from src.mm_toolbox.rounding import Round

# @njit(["uint32(uint32[:, :], uint32, uint32, uint32)"], inline="always")
# def linear_search(arr: np.ndarray, price: int, max_len: int, start: int = 0) -> int:
#     """
#     Performs a linear search on the order book array to find the level that matches the specified price,
#     starting from a specified index.

#     Parameters
#     ----------
#     arr : np.ndarray
#         The bid or ask array where each row represents a price level.

#     price : uint32
#         The price level to search for in the order book.

#     max_len : uint32
#         The maximum length of the array to search within.

#     start : uint32, optional
#         The index to start the search from. Default is 0.

#     Returns
#     -------
#     int
#         The index of the matching price level, or max_len if the price is not found (indicative of an error condition).
#     """
#     for i in range(start, max_len):
#         if arr[i, 0] == price:
#             return i

#     return max_len

# @njit(["uint32(uint32[:, :], uint32, uint32, uint32)"], inline="always")
# def binary_search(arr: np.ndarray, price: int, max_len: int, start: int = 0) -> int:
#     """
#     Performs a binary search on the order book array to find the level that matches the specified price,
#     starting from a specified index.

#     Parameters
#     ----------
#     arr : np.ndarray
#         The bid or ask array where each row represents a price level, typically used for ask arrays.

#     price : uint32
#         The price level to search for in the order book.

#     max_len : uint32
#         The length of the array to search through.

#     start : uint32, optional
#         The index to start the search from. Default is 0.

#     Returns
#     -------
#     int
#         The index of the matching price level, or n+1 if the price is not found (indicative of an error condition).
#     """
#     low = start
#     high = max_len - 1

#     while low <= high:
#         mid = (low + high) // 2
#         mid_price = arr[mid, 0]

#         if mid_price < price:
#             low = mid + 1
#         elif mid_price > price:
#             high = mid - 1
#         else:
#             return mid

#     return max_len

# @njit(["void(uint32[:, :], uint32[:, :], uint32[:, :])"], error_model="numpy", fastmath=True)
# def process_full_l2_bids(bids: np.ndarray, asks: np.ndarray, updates: np.ndarray) -> None:
#     """
#     Processes a full level 2 update for bids and adjusts asks if necessary.

#     Steps:
#     1. Identify the best and worst price levels in the current bids.
#     2. For each update in the updates array:
#        a. If the incoming price matches the best bid price, update or remove the level if the size is zero.
#        b. If the incoming price is greater than the best bid price, adjust the order book to accommodate new price levels.
#           Additionally, if the new bid price overlaps with asks, roll the asks to the right.
#        c. If the incoming price is between the worst and best prices, find the exact level to update the size.
#        d. If the incoming price is lower than the worst bid price, ignore the update.
#     3. Update the best and worst prices after each iteration.

#     Parameters
#     ----------
#     bids : np.ndarray
#         The bids array where each row represents a price level.

#     updates : np.ndarray
#         The updates array where each row represents a [price, size] pair for the update.

#     asks : np.ndarray
#         The asks array where each row represents a price level. Adjusted if new bid price overlaps.

#     Returns
#     -------
#     None
#     """

#     bid_len = bids.shape[0]
#     best_price = bids[-1, 0]
#     worst_price = bids[0, 0]
#     start_idx = bid_len - 1

#     # Hard assumption that its ordered low -> high
#     for update in updates[::-1]:
#         price, size = update

#         # Check (a)
#         if price == best_price:
#             if size == 0:
#                 bids[1:] = bids[:-1]
#                 bids[0, 0] = worst_price - 1
#                 bids[0, 1] = 0
#             else:
#                 bids[-1, 1] = size

#         # Check (b)
#         elif price > best_price:
#             num_new_levels = price - best_price
#             bids[:-num_new_levels] = bids[num_new_levels:]
#             start_idx = bid_len - num_new_levels

#             for i in range(1, num_new_levels + 1):
#                 idx = start_idx + i
#                 bids[idx, 0] = best_price + i
#                 bids[idx, 1] = 0

#             bids[-1, 1] = size

#             # If price overlaps with any asks, roll asks to the right
#             best_ask_price = asks[0, 0]

#             if price > best_ask_price:
#                 num_new_levels = price - best_ask_price
#                 asks[:-num_new_levels] = asks[num_new_levels:]
#                 start_idx = asks.shape[0] - num_new_levels

#                 for i in range(1, num_new_levels + 1):
#                     idx = start_idx + i
#                     asks[idx, 0] = asks[idx-1, 0] + 1
#                     asks[idx, 1] = 0

#         # Check (c)
#         elif price >= worst_price:
#             start_idx = linear_search_reversed(bids, price, start_idx)
#             bids[start_idx, 1] = size

#         # Check (d)
#         else:
#             continue

#         # Update best and worst prices
#         best_price = bids[-1, 0]
#         worst_price = bids[0, 0]

# @njit(["void(uint32[:, :], uint32[:, :], uint32[:, :])"], error_model="numpy", fastmath=True)
# def process_full_l2_asks(bids: np.ndarray, asks: np.ndarray, updates: np.ndarray) -> None:
#     """
#     Processes a full level 2 update for asks.

#     Steps:
#     1. Identify the best and worst price levels in the current asks.
#     2. For each update in the updates array:
#        a. If the incoming price matches the best ask price, update or remove the level if the size is zero.
#        b. If the incoming price is greater than the worst ask price, ignore the update.
#        c. If the incoming price is between the worst and the best prices, find the exact level to update the size.
#        d. If the incoming price is better than the best price, adjust the order book to accommodate new price levels.
#     3. Update the best and worst prices after each iteration.

#     Parameters
#     ----------
#     bids : np.ndarray
#         The bids array where each row represents a price level.

#     asks : np.ndarray
#         The asks array where each row represents a price level.

#     updates : np.ndarray
#         The updates array where each row represents a [price, size] pair for the update.

#     Returns
#     -------
#     None
#     """
#     best_price = asks[0, 0]
#     worst_price = asks[-1, 0]
#     start_idx = 0

#     for update in updates:
#         price, size = update

#         # Step (a)
#         if price == best_price:
#             if size == 0:
#                 asks[:-1] = asks[1:]
#                 asks[-1, 0] = worst_price + 1
#                 asks[-1, 1] = 0
#             else:
#                 asks[0, 1] = size

#         # Step (b)
#         elif price < best_price:
#             num_new_levels = best_price - price
#             asks[num_new_levels:] = asks[:-num_new_levels]
#             start_idx = num_new_levels

#             for i in range(1, num_new_levels + 1):
#                 idx = num_new_levels - i
#                 asks[idx, 0] = best_price - i
#                 asks[idx, 1] = 0

#             asks[0, 1] = size

#             # If price overlaps with any bids, roll bids to the left
#             best_bid_price = bids[0, 0]

#             if price < best_bid_price:
#                 num_new_levels = best_bid_price - price
#                 bids[num_new_levels:] = bids[:-num_new_levels]

#                 for i in range(1, num_new_levels + 1):
#                     idx = num_new_levels - i
#                     bids[idx, 0] = bids[idx+1, 0] - 1
#                     bids[idx, 1] = 0

#         # Step (c)
#         elif price < worst_price:
#             start_idx = linear_search(asks, price, start_idx)
#             asks[start_idx, 1] = size

#         # Step (d)
#         else:
#             continue

#         # Update best and worst prices
#         best_price = asks[0, 0]
#         worst_price = asks[-1, 0]

# @njit(["void(uint32[:, :], uint32[:, :], uint32, uint32, uint32, uint32)"], error_model="numpy", fastmath=True)
# def process_bbo_l2(bids: np.ndarray, asks: np.ndarray, updated_bid_price: int, updated_bid_size: int, updated_ask_price: int, updated_ask_size: int) -> None:
#     """
#     Processes a BBO (Best Bid and Offer) level 2 update.

#     Steps
#     -----
#     1. Process the incoming bid update:
#        a. If the incoming bid price matches the best bid price:
#           - If the size is zero, remove the level and shift the order book.
#           - Otherwise, update the size at the best bid level.
#        b. If the incoming bid price is higher than the worst bid price, ignore the update.
#        c. If the incoming bid price is between the worst and the best prices, find the exact level to update the size.
#        d. If the incoming bid price is better than the best price, adjust the order book to accommodate new price levels.

#     2. Process the incoming ask update:
#        a. If the incoming ask price matches the best ask price:
#           - If the size is zero, remove the level and shift the order book.
#           - Otherwise, update the size at the best ask level.
#        b. If the incoming ask price is lower than the worst ask price, ignore the update.
#        c. If the incoming ask price is between the worst and the best prices, find the exact level to update the size.
#        d. If the incoming ask price is better than the best price, adjust the order book to accommodate new price levels.

#     Parameters
#     ----------
#     bids : np.ndarray
#         The bids array where each row represents a price level.

#     asks : np.ndarray
#         The asks array where each row represents a price level.

#     updated_bid_price : int
#         The updated price for the bid.

#     updated_bid_size : int
#         The updated size for the bid.

#     updated_ask_price : int
#         The updated price for the ask.

#     updated_ask_size : int
#         The updated size for the ask.

#     Returns
#     -------
#     None
#     """
#     best_bid_price = bids[-1, 0]
#     best_ask_price = asks[0, 0]

#     sides_updated: int = 0

#     if best_bid_price == updated_bid_price:
#         if updated_bid_size == 0:
#             bids[1:] = bids[:-1]
#             bids[0, 0] = bids[1, 0] - 1
#             bids[0, 1] = 0
#         else:
#             bids[-1, 1] = updated_bid_size

#         sides_updated += 1

#     if best_ask_price == updated_ask_price:
#         if updated_ask_size == 0:
#             asks[:-1] = asks[1:]
#             asks[-1, 0] = asks[-2, 0] + 1
#             asks[-1, 1] = 0
#         else:
#             asks[-1, 1] = updated_ask_size

#         sides_updated += 1

#     if sides_updated == 2:
#         return None

#     bid_price_greater = best_bid_price > updated_bid_price
#     ask_price_greater = best_ask_price > updated_ask_price
#     bid_price_lower = best_bid_price < updated_bid_price
#     ask_price_lower = best_ask_price < updated_ask_price

#     if bid_price_greater:
#         num_new_levels = best_price - updated_price
#         asks[num_new_levels:] = asks[:-num_new_levels]

#         for i in range(1, num_new_levels + 1):
#             idx = num_new_levels - i
#             asks[idx, 0] = best_price - i
#             asks[idx, 1] = 0

#         asks[0, 1] = updated_size

#         # If price overlaps with any bids, roll bids to the left
#         best_bid_price = bids[0, 0]

#         if updated_price < best_bid_price:
#             num_new_levels = best_bid_price - updated_price
#             bids[num_new_levels:] = bids[:-num_new_levels]

#             for i in range(1, num_new_levels + 1):
#                 idx = num_new_levels - i
#                 bids[idx, 0] = bids[idx+1, 0] - 1
#                 bids[idx, 1] = 0


# @jitclass
# class HFTOrderbook:
#     tick_size: float64
#     lot_size: float64
#     num_levels: uint32
#     asks: uint32[:, :]
#     bids: uint32[:, :]
#     warmed_up: bool_
#     round: Round.class_type.instance_type

#     def __init__(self, tick_size: float, lot_size: float, num_levels: int=2500) -> None:
#         """
#         Initializes the Orderbook with given tick size, lot size, and number of levels.

#         Parameters
#         ----------
#         tick_size : float
#             The tick size for price normalization.

#         lot_size : float
#             The lot size for size normalization.

#         num_levels : int
#             The number of price levels to maintain in the order book.
#         """
#         self.tick_size = tick_size
#         self.lot_size = lot_size
#         self.num_levels = num_levels
#         self.asks = np.zeros((self.num_levels, 2), dtype=uint32)
#         self.bids = np.zeros((self.num_levels, 2), dtype=uint32)
#         self.warmed_up = False

#         self.round = Round(self.tick_size, self.lot_size)

#     def reset(self) -> None:
#         """
#         Resets the order book by clearing the bids and asks and setting warmed_up to False.

#         Returns
#         -------
#         None
#         """
#         self.asks.fill(0)
#         self.bids.fill(0)
#         self.warmed_up = False

#     def normalize(self, num: float, step: float) -> int:
#         """
#         Normalizes a number by dividing it by a step value and rounding to the nearest integer.

#         Parameters
#         ----------
#         num : float
#             The number to be normalized.

#         step : float
#             The step size used for normalization.

#         Returns
#         -------
#         int
#             The normalized integer value.
#         """
#         return round(num / step)

#     def normalize_book(self, orderbook: np.ndarray) -> np.ndarray:
#         """
#         Normalizes the order book array by normalizing the prices and sizes.

#         Parameters
#         ----------
#         orderbook : np.ndarray
#             The (x, 2) sized array representing the order book with the first column as prices and the second column as sizes.

#         Returns
#         -------
#         Array
#             The normalized order book.
#         """
#         orderbook[:, 0] = np.rint(orderbook[:, 0] / self.tick_size)
#         orderbook[:, 1] = np.rint(orderbook[:, 1] / self.lot_size)
#         return orderbook.astype(uint32)

#     def denormalize(self, num: int, step: float) -> float:
#         """
#         Denormalizes a number by multiplying it with a step value.

#         Parameters
#         ----------
#         num : int
#             The integer to be denormalized.

#         step : float
#             The step size used for denormalization.

#         Returns
#         -------
#         float
#             The denormalized float value.
#         """
#         return num * step

#     def denormalize_book(self, orderbook: np.ndarray) -> np.ndarray:
#         """
#         Denormalizes the order book array where the first column represents prices and the second column represents sizes.

#         Parameters
#         ----------
#         orderbook : np.ndarray
#             The (x, 2) sized array representing the order book with the first column as prices and the second column as sizes.

#         Returns
#         -------
#         Array
#             The denormalized order book.
#         """
#         denormalized_book = np.zeros_like(orderbook, dtype=float64)
#         denormalized_book[:, 0] = self.denormalize(orderbook[:, 0], self.tick_size)
#         denormalized_book[:, 1] = self.denormalize(orderbook[:, 1], self.lot_size)
#         return denormalized_book

#     def warmup(self, new_bids: np.ndarray, new_asks: np.ndarray) -> None:
#         """
#         Initializes the order book with initial bid and ask updates.

#         Steps
#         -----
#         1. Identify the best bid and best ask prices and sizes from the updates.
#         - The best bid is the bid with the highest price.
#         - The best ask is the ask with the lowest price.

#         2. Normalize the best bid and best ask prices using the tick size.
#         - Normalization involves dividing the prices by the tick size and rounding to the nearest integer.

#         3. Initialize the bid side of the order book.
#         - Set the starting bid price as the normalized best bid price.
#         - Set the stopping bid price as the starting bid price minus the number of levels.
#         - Create a range of bid prices from the starting bid price to the stopping bid price, in reverse order.
#         - Assign these prices to the first column of the bids array.
#         - Set the size of the best bid level (last level) to the normalized best bid size.

#         4. Initialize the ask side of the order book.
#         - Set the starting ask price as the normalized best ask price.
#         - Set the stopping ask price as the starting ask price plus the number of levels.
#         - Create a range of ask prices from the starting ask price to the stopping ask price.
#         - Assign these prices to the first column of the asks array.
#         - Set the size of the best ask level (first level) to the normalized best ask size.

#         5. Mark the order book as warmed up by setting the warmed_up attribute to True.

#         Parameters
#         ----------
#         new_bids : np.ndarray
#             An array of initial bid prices and sizes.

#         new_asks : np.ndarray
#             An array of initial ask prices and sizes.

#         Returns
#         -------
#         None
#         """
#         best_bid = new_bids[new_bids[:, 0] == np.max(new_bids[:, 0])][0]
#         best_ask = new_asks[new_asks[:, 0] == np.min(new_asks[:, 0])][0]

#         bid_start = self.normalize(best_bid[0], self.tick_size)
#         bid_stop = bid_start - self.num_levels
#         self.bids[:, 0] = np.arange(bid_start, bid_stop, -1)[::-1]
#         self.bids[-1, 1] = self.normalize(best_bid[1], self.lot_size)

#         ask_start = self.normalize(best_ask[0], self.tick_size)
#         ask_stop = ask_start + self.num_levels
#         self.asks[:, 0] = np.arange(ask_start, ask_stop, 1)
#         self.asks[0, 1] = self.normalize(best_ask[1], self.lot_size)

#         self.warmed_up = True

#     def ingest_bbo_update(self, bid_price: float, bid_size: float, ask_price: float, ask_size: float) -> None:
#         if not self.warmed_up:
#             raise Exception("Orderbook is not warmed up!")

#         process_bbo_l2(
#             bids=self.bids,
#             asks=self.asks,
#             updated_bid_price=self.normalize(bid_price),
#             updated_bid_size=self.normalize(bid_size),
#             updated_ask_price=self.normalize(ask_price),
#             updated_ask_size=self.normalize(ask_size)
#         )

#     def ingest_l2_bid_update(self, update_bids: np.ndarray) -> None:
#         """
#         Processes Level 2 market data bid updates by updating the bids in the order book.

#         Parameters
#         ----------
#         update_bids : np.ndarray
#             An array of new bid prices and sizes.

#         Returns
#         -------
#         None
#         """
#         if not self.warmed_up:
#             raise Exception("Orderbook is not warmed up!")

#         process_full_l2_bids(
#             bids=self.bids,
#             asks=self.asks,
#             updates=self.normalize_book(update_bids)
#         )

#     def ingest_l2_ask_update(self, update_asks: np.ndarray) -> None:
#         """
#         Processes Level 2 market data ask updates by updating the asks in the order book.

#         Parameters
#         ----------
#         update_asks : np.ndarray
#             An array of new ask prices and sizes.

#         Returns
#         -------
#         None
#         """
#         if not self.warmed_up:
#             raise Exception("Orderbook is not warmed up!")

#         process_full_l2_asks(
#             bids=self.bids,
#             asks=self.asks,
#             updates=self.normalize_book(update_asks)
#         )

#     def get_bids(self) -> np.ndarray:
#         return self.denormalize_book(self.bids)

#     def get_asks(self) -> np.ndarray:
#         return self.denormalize_book(self.asks)

#     def get_best_bid(self) -> Tuple[float, float]:
#         """
#         Retrieves the best bid price and size from the order book.

#         Returns
#         -------
#         Tuple[float, float]
#             A tuple containing the best bid price and size, both denormalized.
#         """
#         return (
#             self.round.bid(self.denormalize(self.bids[-1, 0], self.tick_size)),
#             self.round.size(self.denormalize(self.bids[-1, 1], self.lot_size))
#         )

#     def get_best_ask(self) -> Tuple[float, float]:
#         """
#         Retrieves the best ask price and size from the order book.

#         Returns
#         -------
#         Tuple[float, float]
#             A tuple containing the best ask price and size, both denormalized.
#         """
#         return (
#             self.denormalize(self.asks[0, 0], self.tick_size),
#             self.denormalize(self.asks[0, 1], self.lot_size)
#         )

#     def get_mid_price(self) -> float:
#         """
#         Calculates the mid price of the order book based on the best bid and ask prices.

#         Returns
#         -------
#         float
#             The mid price, which is the average of the best bid and best ask prices.
#         """
#         mid_price = (self.bids[-1, 0] + self.asks[0, 0]) / 2.0
#         return self.denormalize(mid_price, self.tick_size)

#     def get_wmid_price(self) -> float:
#         """
#         Calculates the weighted mid price of the order book, considering the volume imbalance
#         between the best bid and best ask.

#         Returns
#         -------
#         float
#             The weighted mid price, which accounts for the volume imbalance at the top of the book.
#         """
#         imb = self.bids[-1, 1] / (self.bids[-1, 1] + self.asks[0, 1])
#         wmid_price = self.bids[-1, 0] * imb + self.asks[0, 0] * (1.0 - imb)
#         return self.denormalize(wmid_price, self.tick_size)

#     def get_vamp(self, dollar_depth: float) -> float:
#         """
#         Calculates the volume-weighted average market price (VAMP) up to a specified dollar depth for both bids and asks.

#         Parameters
#         ----------
#         dollar_depth : float
#             The dollar depth (in terms of total order value) up to which the VAMP is calculated.

#         Returns
#         -------
#         float
#             The VAMP, representing an average price weighted by order sizes up to the specified dollar depth.
#         """
#         bid_dollar_weighted_sum = 0.0
#         ask_dollar_weighted_sum = 0.0
#         bid_dollar_cum = 0.0
#         ask_dollar_cum = 0.0

#         # Indexed backwards for best -> worst order
#         for price, size in self.denormalize_book(self.bids[::-1]):
#             order_value = price * size
#             if bid_dollar_cum + order_value > dollar_depth:
#                 remaining_value = dollar_depth - bid_dollar_cum
#                 bid_dollar_weighted_sum += remaining_value
#                 bid_dollar_cum += remaining_value
#                 break

#             bid_dollar_weighted_sum += order_value
#             bid_dollar_cum += order_value

#             if bid_dollar_cum >= dollar_depth:
#                 break

#         for price, size in self.denormalize_book(self.asks):
#             order_value = price * size
#             if ask_dollar_cum + order_value > dollar_depth:
#                 remaining_value = dollar_depth - ask_dollar_cum
#                 ask_dollar_weighted_sum += remaining_value
#                 ask_dollar_cum += remaining_value
#                 break

#             ask_dollar_weighted_sum += order_value
#             ask_dollar_cum += order_value

#             if ask_dollar_cum >= dollar_depth:
#                 break

#         total_dollar = bid_dollar_cum + ask_dollar_cum

#         if total_dollar == 0:
#             return 0.0

#         return (bid_dollar_weighted_sum + ask_dollar_weighted_sum) / total_dollar

#     def get_spread(self) -> float:
#         """
#         Calculates the current spread of the order book.

#         Returns
#         -------
#         float
#             The spread, defined as the difference between the best ask and the best bid prices.
#         """
#         return self.denormalize(self.asks[0, 0] - self.bids[-1, 0], self.tick_size)

#     def get_slippage(self, book: np.ndarray, dollar_depth: float) -> float:
#         """
#         Calculates the slippage cost for a hypothetical order of a given dollar depth, based on either the bid or ask side of the book.

#         Parameters
#         ----------
#         book : np.ndarray
#             The order book data for the side (bids or asks) being considered.

#         dollar_depth : float
#             The dollar depth of the hypothetical order for which slippage is being calculated.

#         Returns
#         -------
#         float
#             The slippage cost, defined as the volume-weighted average deviation from the mid price for the given order depth in dollars.
#         """
#         mid_price = self.get_mid_price()
#         dollar_cum = 0.0
#         slippage = 0.0
#         denormalize_book = self.denormalize_book(book)

#         for price, size in denormalize_book:
#             order_value = price * size
#             dollar_cum += order_value
#             slippage += np.abs(mid_price - price) * order_value

#             if dollar_cum >= dollar_depth:
#                 slippage /= dollar_cum
#                 return slippage

#         # If the full depth wasn't reached (i.e., the book is not deep enough), return NaN
#         return np.nan
