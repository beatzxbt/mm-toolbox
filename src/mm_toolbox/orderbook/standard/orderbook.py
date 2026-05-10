from bisect import bisect_left, bisect_right, insort
from collections.abc import Iterator

from .level import OrderbookLevel, price_from_ticks, price_to_ticks_fast


class Orderbook:
    """An orderbook class with functionality to initialize, update, and access
    best bid/ask information.
    """

    def __init__(
        self,
        tick_size: float,
        lot_size: float,
        size: int = 500,
        initial_bids: list[OrderbookLevel] | None = None,
        initial_asks: list[OrderbookLevel] | None = None,
        trust_input_precision: bool = False,
    ) -> None:
        """Initialize orderbook with tick size, lot size, and optional
        initial levels.

        Args:
            tick_size: Minimum price increment.
            lot_size: Minimum size increment.
            size: Minimum snapshot depth required by consume_snapshot.
            initial_bids: Optional initial bid levels.
            initial_asks: Optional initial ask levels.
            trust_input_precision: If True, reuse existing level ticks/lots
                when present instead of recomputing from price/size.
        """
        if tick_size <= 0.0:
            raise ValueError(f"Invalid tick_size; expected >0 but got {tick_size}")
        if lot_size <= 0.0:
            raise ValueError(f"Invalid lot_size; expected >0 but got {lot_size}")
        if size <= 0:
            raise ValueError(f"Invalid size; expected >0 but got {size}")

        self._tick_size = tick_size
        self._lot_size = lot_size
        self._inv_tick_size = 1.0 / tick_size
        self._inv_lot_size = 1.0 / lot_size
        self._size = size
        self._trust_input_precision = trust_input_precision

        # {ticks: OrderbookLevel}
        self._asks: dict[int, OrderbookLevel] = {}
        self._bids: dict[int, OrderbookLevel] = {}

        self._sorted_ask_ticks: list[int] = []
        self._sorted_bid_ticks: list[int] = []

        # Cached BBO values to avoid property overhead.
        self._best_ask_ticks = 0
        self._best_bid_ticks = 0

        self._is_initialized = False
        self._is_populated = False

        if initial_bids is not None and initial_asks is not None:
            self.consume_snapshot(asks=initial_asks, bids=initial_bids)

    def is_initialized(self) -> bool:
        """Check whether the orderbook has received at least one update.

        Returns:
            bool: True after the first snapshot or delta is consumed.

        """
        return self._is_initialized

    def is_populated(self) -> bool:
        """Check whether both bid and ask sides currently have levels.

        Returns:
            bool: True when both sides are non-empty.

        """
        return self._is_populated

    def _ensure_initialized(self) -> None:
        """Guard method that raises if the orderbook has never been updated.

        Raises:
            ValueError: If the orderbook has not received any data yet.

        """
        if not self._is_initialized:
            raise ValueError("Orderbook is not populated.")

    def _ensure_bbo_available(self) -> None:
        """Guard method that raises if either side is empty.

        Raises:
            ValueError: If bids or asks are currently unavailable.

        """
        self._ensure_initialized()
        if not self._is_populated:
            raise ValueError("Orderbook side unavailable.")

    def _refresh_population_state(self) -> None:
        """Recalculate whether both sides are non-empty."""
        self._is_populated = (
            len(self._sorted_bid_ticks) > 0 and len(self._sorted_ask_ticks) > 0
        )

    def _update_bbo_cache(self) -> None:
        """Refresh cached best-bid/best-ask tick values from sorted lists."""
        if self._sorted_ask_ticks:
            self._best_ask_ticks = self._sorted_ask_ticks[0]
        if self._sorted_bid_ticks:
            self._best_bid_ticks = self._sorted_bid_ticks[-1]

    def reset(self) -> None:
        """Clear all levels and reset state to empty."""
        self._asks.clear()
        self._bids.clear()
        self._sorted_ask_ticks.clear()
        self._sorted_bid_ticks.clear()
        self._best_ask_ticks = 0
        self._best_bid_ticks = 0
        self._is_initialized = False
        self._is_populated = False

    def _ensure_level_precision(self, level: OrderbookLevel) -> None:
        """Compute ticks and lots for a level when not already present.

        Args:
            level (OrderbookLevel): Level to populate in-place.

        """
        if self._trust_input_precision and level.ticks >= 0 and level.lots >= 0:
            return
        level.ticks = int(level.price * self._inv_tick_size)
        level.lots = int(level.size * self._inv_lot_size)

    def _consume_side_deltas(
        self,
        levels: list[OrderbookLevel],
        side_levels: dict[int, OrderbookLevel],
        sorted_ticks: list[int],
    ) -> None:
        """Apply a list of delta levels to one side of the book.

        Args:
            levels (list[OrderbookLevel]): Incoming delta levels.
            side_levels (dict[int, OrderbookLevel]): Existing side dictionary.
            sorted_ticks (list[int]): Sorted tick list for the side.

        """

        for level in levels:
            self._ensure_level_precision(level)
            ticks = level.ticks

            if level.lots == 0:
                if ticks in side_levels:
                    side_levels.pop(ticks, None)
                    idx = bisect_left(sorted_ticks, ticks)
                    if idx < len(sorted_ticks) and sorted_ticks[idx] == ticks:
                        sorted_ticks.pop(idx)
            else:
                is_new = ticks not in side_levels
                side_levels[ticks] = level
                if is_new:
                    idx = bisect_left(sorted_ticks, ticks)
                    sorted_ticks.insert(idx, ticks)

    def _prune_better_bids(self, bid_ticks: int) -> None:
        """Remove bids that are priced better than the authoritative best bid.

        Args:
            bid_ticks (int): Authoritative best-bid tick value.

        """
        cutoff = bisect_right(self._sorted_bid_ticks, bid_ticks)
        if cutoff >= len(self._sorted_bid_ticks):
            return
        for tick in self._sorted_bid_ticks[cutoff:]:
            self._bids.pop(tick, None)
        del self._sorted_bid_ticks[cutoff:]

    def _prune_better_asks(self, ask_ticks: int) -> None:
        """Remove asks that are priced better than the authoritative best ask.

        Args:
            ask_ticks (int): Authoritative best-ask tick value.

        """
        cutoff = bisect_left(self._sorted_ask_ticks, ask_ticks)
        if cutoff <= 0:
            return
        for tick in self._sorted_ask_ticks[:cutoff]:
            self._asks.pop(tick, None)
        del self._sorted_ask_ticks[:cutoff]

    def consume_snapshot(
        self,
        asks: list[OrderbookLevel],
        bids: list[OrderbookLevel],
    ) -> None:
        """Replace the entire book with a full snapshot.

        Args:
            asks (list[OrderbookLevel]): Complete ask side.
            bids (list[OrderbookLevel]): Complete bid side.

        Raises:
            ValueError: If either side has fewer levels than ``size``.

        """
        if len(asks) < self._size:
            raise ValueError(
                f"Invalid asks with snapshot; expected >= {self._size} levels but got {len(asks)}"
            )
        if len(bids) < self._size:
            raise ValueError(
                f"Invalid bids with snapshot; expected >= {self._size} levels but got {len(bids)}"
            )

        self.reset()

        for ask in asks:
            self._ensure_level_precision(ask)
            self._asks[ask.ticks] = ask
        self._sorted_ask_ticks[:] = sorted(self._asks)

        for bid in bids:
            self._ensure_level_precision(bid)
            self._bids[bid.ticks] = bid
        self._sorted_bid_ticks[:] = sorted(self._bids)

        self._is_initialized = True
        self._is_populated = (
            len(self._sorted_bid_ticks) > 0 and len(self._sorted_ask_ticks) > 0
        )
        self._update_bbo_cache()

    def consume_deltas(
        self,
        asks: list[OrderbookLevel],
        bids: list[OrderbookLevel],
    ) -> None:
        """Apply incremental updates to the existing book.

        Args:
            asks (list[OrderbookLevel]): Ask delta levels.
            bids (list[OrderbookLevel]): Bid delta levels.

        """
        if asks:
            self._consume_side_deltas(asks, self._asks, self._sorted_ask_ticks)
        if bids:
            self._consume_side_deltas(bids, self._bids, self._sorted_bid_ticks)
        if self._is_initialized:
            self._refresh_population_state()
        if self._is_populated:
            self._update_bbo_cache()

    def consume_bbo(
        self,
        ask: OrderbookLevel,
        bid: OrderbookLevel,
    ) -> None:
        """Update the best bid and offer.

        BBO updates usually don't supply size=0 updates for signalling level
        deletion, therefore, we ignore it and directly update the best bid/ask for now,
        fixing any issues arising with the orderbook by assuming this source of truth.

        Args:
            ask: Ask level.
            bid: Bid level.
        """
        bids = self._bids
        asks = self._asks
        sorted_bid_ticks = self._sorted_bid_ticks
        sorted_ask_ticks = self._sorted_ask_ticks

        self._ensure_level_precision(bid)
        self._ensure_level_precision(ask)

        bid_ticks, bid_lots = bid.ticks, bid.lots
        ask_ticks, ask_lots = ask.ticks, ask.lots

        if bid_lots == 0:
            if sorted_bid_ticks:
                best_bid_ticks = self._best_bid_ticks
                bids.pop(best_bid_ticks, None)
                sorted_bid_ticks.pop()
        else:
            self._prune_better_bids(bid_ticks)
            if bid_ticks not in bids:
                insort(sorted_bid_ticks, bid_ticks)
            bids[bid_ticks] = bid

        if ask_lots == 0:
            if sorted_ask_ticks:
                best_ask_ticks = self._best_ask_ticks
                asks.pop(best_ask_ticks, None)
                sorted_ask_ticks.pop(0)
        else:
            self._prune_better_asks(ask_ticks)
            if ask_ticks not in asks:
                insort(sorted_ask_ticks, ask_ticks)
            asks[ask_ticks] = ask

        self._is_initialized = True
        self._refresh_population_state()
        self._update_bbo_cache()

    def get_asks(self, depth: int | None = None) -> list[OrderbookLevel]:
        """Return ask levels sorted by price ascending.

        Args:
            depth (int, optional): Maximum number of levels to return.
                Returns all levels when ``None``.

        Returns:
            list[OrderbookLevel]: Ask levels from best to worst.

        Raises:
            ValueError: If the orderbook has not been initialized.

        """
        self._ensure_initialized()
        asks = self._asks
        sorted_ask_ticks = self._sorted_ask_ticks
        if depth is None:
            return [asks[tick] for tick in sorted_ask_ticks]
        if depth <= 0:
            return []
        result = []
        for i, tick in enumerate(sorted_ask_ticks):
            if i >= depth:
                break
            result.append(asks[tick])
        return result

    def get_bids(self, depth: int | None = None) -> list[OrderbookLevel]:
        """Return bid levels sorted by price descending.

        Args:
            depth (int, optional): Maximum number of levels to return.
                Returns all levels when ``None``.

        Returns:
            list[OrderbookLevel]: Bid levels from best to worst.

        Raises:
            ValueError: If the orderbook has not been initialized.

        """
        self._ensure_initialized()
        bids = self._bids
        sorted_bid_ticks = self._sorted_bid_ticks
        if depth is None:
            return [bids[tick] for tick in reversed(sorted_bid_ticks)]
        if depth <= 0:
            return []
        result = []
        n = len(sorted_bid_ticks)
        for i in range(n - 1, n - 1 - depth, -1):
            if i < 0:
                break
            result.append(bids[sorted_bid_ticks[i]])
        return result

    def iter_asks(self, depth: int | None = None) -> Iterator[OrderbookLevel]:
        """Yield ask levels from best to worst price.

        Args:
            depth (int, optional): Maximum number of levels to yield.
                Yields all levels when ``None``.

        Yields:
            OrderbookLevel: Next ask level in ascending price order.

        Raises:
            ValueError: If the orderbook has not been initialized.

        """
        self._ensure_initialized()
        if depth is not None and depth <= 0:
            return
        asks = self._asks
        sorted_ask_ticks = self._sorted_ask_ticks
        if depth is None:
            for tick in sorted_ask_ticks:
                yield asks[tick]
        else:
            for i, tick in enumerate(sorted_ask_ticks):
                if i >= depth:
                    break
                yield asks[tick]

    def iter_bids(self, depth: int | None = None) -> Iterator[OrderbookLevel]:
        """Yield bid levels from best to worst price.

        Args:
            depth (int, optional): Maximum number of levels to yield.
                Yields all levels when ``None``.

        Yields:
            OrderbookLevel: Next bid level in descending price order.

        Raises:
            ValueError: If the orderbook has not been initialized.

        """
        self._ensure_initialized()
        if depth is not None and depth <= 0:
            return
        bids = self._bids
        sorted_bid_ticks = self._sorted_bid_ticks
        n = len(sorted_bid_ticks)
        if depth is None:
            for i in range(n - 1, -1, -1):
                yield bids[sorted_bid_ticks[i]]
        else:
            for i in range(n - 1, n - 1 - depth, -1):
                if i < 0:
                    break
                yield bids[sorted_bid_ticks[i]]

    def get_bbo(self) -> tuple[OrderbookLevel, OrderbookLevel]:
        """Return the best bid and best ask.

        Returns:
            tuple[OrderbookLevel, OrderbookLevel]: ``(best_bid, best_ask)``.

        Raises:
            ValueError: If the orderbook is uninitialized or one side is empty.

        """
        self._ensure_bbo_available()
        return self._bids[self._best_bid_ticks], self._asks[self._best_ask_ticks]

    def get_bbo_spread(self) -> float:
        """Return the bid-ask spread in price terms.

        Returns:
            float: Difference between best ask and best bid prices.

        Raises:
            ValueError: If the orderbook is uninitialized or one side is empty.

        """
        self._ensure_bbo_available()
        spread_ticks = self._best_ask_ticks - self._best_bid_ticks
        tick_size = self._tick_size
        return spread_ticks * tick_size

    def get_mid_price(self) -> float:
        """Return the simple mid price.

        Returns:
            float: Arithmetic midpoint of best bid and best ask.

        Raises:
            ValueError: If the orderbook is uninitialized or one side is empty.

        """
        self._ensure_bbo_available()
        mid_ticks = (self._best_ask_ticks + self._best_bid_ticks) // 2
        tick_size = self._tick_size
        return mid_ticks * tick_size

    def get_wmid_price(self) -> float:
        """Return the lot-weighted mid price.

        Weights the midpoint by the relative lot sizes at best bid and ask.

        Returns:
            float: Weighted mid price.

        Raises:
            ValueError: If the orderbook is uninitialized or one side is empty.

        """
        self._ensure_bbo_available()
        bids = self._bids
        asks = self._asks
        best_bid_ticks = self._best_bid_ticks
        best_ask_ticks = self._best_ask_ticks
        best_bid_lots = bids[best_bid_ticks].lots
        best_ask_lots = asks[best_ask_ticks].lots

        total_lots = best_bid_lots + best_ask_lots
        if total_lots == 0:
            return 0.0
        wmid_ticks = (
            best_bid_ticks * best_bid_lots + best_ask_ticks * best_ask_lots
        ) // total_lots
        return price_from_ticks(wmid_ticks, self._tick_size)

    def get_volume_weighted_mid_price(
        self, size: float, is_base_currency: bool = True
    ) -> float:
        """Return the mid of the prices required to buy and sell a given size.

        Walks the book to determine the average execution price on each
        side for the requested quantity, then returns their midpoint.

        Args:
            size (float): Target size to evaluate.
            is_base_currency (bool): If True, ``size`` is in base currency.
                Otherwise it is treated as quote notional and converted using
                the mid price. Defaults to True.

        Returns:
            float: Midpoint of the buy and sell execution prices, or
            ``float('inf')`` if the book is too shallow.

        Raises:
            ValueError: If the orderbook is uninitialized or one side is empty.

        """
        self._ensure_bbo_available()

        asks = self._asks
        bids = self._bids
        sorted_ask_ticks = self._sorted_ask_ticks
        sorted_bid_ticks = self._sorted_bid_ticks
        best_ask_ticks = self._best_ask_ticks
        best_bid_ticks = self._best_bid_ticks
        tick_size = self._tick_size

        if size == 0.0:
            mid_ticks = (best_ask_ticks + best_bid_ticks) // 2
            return mid_ticks * tick_size

        mid_ticks = (best_ask_ticks + best_bid_ticks) // 2
        mid_price = mid_ticks * tick_size
        if not is_base_currency:
            size = size / mid_price

        cum_bid_size = 0.0
        buy_price = None
        for tick in sorted_ask_ticks:
            level = asks[tick]
            if cum_bid_size + level.size >= size:
                buy_price = level.price
                break
            cum_bid_size += level.size
        if buy_price is None:
            return float("inf")

        cum_ask_size = 0.0
        sell_price = None
        for tick in reversed(sorted_bid_ticks):
            level = bids[tick]
            if cum_ask_size + level.size >= size:
                sell_price = level.price
                break
            cum_ask_size += level.size
        if sell_price is None:
            return float("inf")

        return (buy_price + sell_price) / 2.0

    def get_price_impact(
        self, size: float, is_buy: bool, is_base_currency: bool = True
    ) -> float:
        """Return the price impact for executing a theoretical size.

        Walks the relevant side of the book until the full size is filled
        and reports the distance from the touch price to the final fill price.

        Args:
            size (float): Size to execute. Must be positive.
            is_buy (bool): If True, impact is computed on the ask side.
                Otherwise on the bid side.
            is_base_currency (bool): If True, ``size`` is in base currency.
                Otherwise it is treated as quote notional. Defaults to True.

        Returns:
            float: Absolute price impact, or ``float('inf')`` if the book is
            too shallow to fill the size.

        Raises:
            ValueError: If the orderbook is uninitialized or one side is empty.

        """
        self._ensure_bbo_available()
        if size <= 0.0:
            return 0.0

        asks = self._asks
        bids = self._bids
        sorted_ask_ticks = self._sorted_ask_ticks
        sorted_bid_ticks = self._sorted_bid_ticks
        best_ask_ticks = self._best_ask_ticks
        best_bid_ticks = self._best_bid_ticks

        if is_buy:
            touch_anchor_price = asks[best_ask_ticks].price
        else:
            touch_anchor_price = bids[best_bid_ticks].price

        if not is_base_currency:
            size = size / touch_anchor_price

        remaining_size = size
        last_touched_price = touch_anchor_price

        if is_buy:
            for tick in sorted_ask_ticks:
                level = asks[tick]
                consumed_size = min(remaining_size, level.size)
                remaining_size -= consumed_size
                if consumed_size > 0.0:
                    last_touched_price = level.price
                if remaining_size <= 0.0:
                    break
        else:
            for tick in reversed(sorted_bid_ticks):
                level = bids[tick]
                consumed_size = min(remaining_size, level.size)
                remaining_size -= consumed_size
                if consumed_size > 0.0:
                    last_touched_price = level.price
                if remaining_size <= 0.0:
                    break

        if remaining_size > 0.0:
            return float("inf")

        return abs(last_touched_price - touch_anchor_price)

    def get_size_for_price_impact_bps(
        self, impact_bps: float, is_buy: bool, is_base_currency: bool = True
    ) -> float:
        """Get cumulative size available within an impact band from touch.

        Args:
            impact_bps: Price depth in basis points from touch price.
            is_buy: If True, aggregate ask-side depth up to touch + band.
                If False, aggregate bid-side depth down to touch - band.
            is_base_currency: If True return base size, else quote notional.
        """
        self._ensure_bbo_available()
        if impact_bps <= 0.0:
            return 0.0

        asks = self._asks
        bids = self._bids
        sorted_ask_ticks = self._sorted_ask_ticks
        sorted_bid_ticks = self._sorted_bid_ticks
        best_ask_ticks = self._best_ask_ticks
        best_bid_ticks = self._best_bid_ticks
        inv_tick_size = self._inv_tick_size
        tick_size = self._tick_size

        total_base = 0.0
        total_quote = 0.0

        if is_buy:
            max_price = asks[best_ask_ticks].price * (1.0 + impact_bps / 10_000.0)
            max_ticks = price_to_ticks_fast(max_price, inv_tick_size)
            for tick in sorted_ask_ticks:
                if tick > max_ticks:
                    break
                level = asks[tick]
                total_base += level.size
                total_quote += level.size * level.price
        else:
            min_price = bids[best_bid_ticks].price * (1.0 - impact_bps / 10_000.0)
            min_ticks = price_to_ticks_fast(min_price, inv_tick_size)
            if min_ticks * tick_size < min_price:
                min_ticks += 1
            for tick in reversed(sorted_bid_ticks):
                if tick < min_ticks:
                    break
                level = bids[tick]
                total_base += level.size
                total_quote += level.size * level.price

        return total_base if is_base_currency else total_quote

    def does_bbo_price_change(self, bid_price: float, ask_price: float) -> bool:
        """Check whether proposed prices differ from current BBO.

        Args:
            bid_price (float): Candidate best bid price.
            ask_price (float): Candidate best ask price.

        Returns:
            bool: True if either price would move the BBO ticks.

        Raises:
            ValueError: If the orderbook is uninitialized or one side is empty.

        """
        self._ensure_bbo_available()
        best_bid_ticks = self._best_bid_ticks
        best_ask_ticks = self._best_ask_ticks
        inv_tick_size = self._inv_tick_size
        other_bid_ticks = price_to_ticks_fast(bid_price, inv_tick_size)
        other_ask_ticks = price_to_ticks_fast(ask_price, inv_tick_size)
        return best_bid_ticks != other_bid_ticks or best_ask_ticks != other_ask_ticks

    def does_bbo_cross(self, bid_price: float, ask_price: float) -> bool:
        """Check whether proposed prices would cross the current BBO.

        Args:
            bid_price (float): Candidate best bid price.
            ask_price (float): Candidate best ask price.

        Returns:
            bool: True if the bid exceeds the current best ask or the ask
            is below the current best bid.

        Raises:
            ValueError: If the orderbook is uninitialized or one side is empty.

        """
        self._ensure_bbo_available()
        best_bid_ticks = self._best_bid_ticks
        best_ask_ticks = self._best_ask_ticks
        inv_tick_size = self._inv_tick_size
        other_bid_ticks = price_to_ticks_fast(bid_price, inv_tick_size)
        other_ask_ticks = price_to_ticks_fast(ask_price, inv_tick_size)
        return best_bid_ticks > other_ask_ticks or best_ask_ticks < other_bid_ticks
