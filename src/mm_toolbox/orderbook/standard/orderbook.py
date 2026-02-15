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

        self._is_initialized = False
        self._is_populated = False

        if initial_bids is not None and initial_asks is not None:
            self.consume_snapshot(asks=initial_asks, bids=initial_bids)

    def is_initialized(self) -> bool:
        """Return True if the orderbook has received at least one update."""
        return self._is_initialized

    def is_populated(self) -> bool:
        """Return True when both bid and ask sides are currently available."""
        return self._is_populated

    def _ensure_initialized(self) -> None:
        """Check if the orderbook has received at least one update."""
        if not self._is_initialized:
            raise ValueError("Orderbook is not populated.")

    def _ensure_bbo_available(self) -> None:
        """Ensure both sides are available for top-of-book calculations."""
        self._ensure_initialized()
        if not self._is_populated:
            raise ValueError("Orderbook side unavailable.")

    def _refresh_population_state(self) -> None:
        """Refresh two-sided availability state."""
        self._is_populated = (
            len(self._sorted_bid_ticks) > 0 and len(self._sorted_ask_ticks) > 0
        )

    @property
    def _best_ask_ticks(self) -> int:
        """Get best ask ticks."""
        return self._sorted_ask_ticks[0]

    @property
    def _best_bid_ticks(self) -> int:
        """Get best bid ticks."""
        return self._sorted_bid_ticks[-1]

    def reset(self) -> None:
        """Reset the orderbook to its initial empty state."""
        self._asks.clear()
        self._bids.clear()
        self._sorted_ask_ticks.clear()
        self._sorted_bid_ticks.clear()
        self._is_initialized = False
        self._is_populated = False

    def _ensure_level_precision(self, level: OrderbookLevel) -> None:
        """Populate ticks/lots unless trusted pre-computed values can be reused."""
        level.add_precision_info(
            inv_tick_size=self._inv_tick_size,
            inv_lot_size=self._inv_lot_size,
            unsafe=self._trust_input_precision,
        )

    @staticmethod
    def _resort_ticks(
        levels: dict[int, OrderbookLevel],
        sorted_ticks: list[int],
    ) -> None:
        """Rebuild sorted tick cache from the side dictionary."""
        sorted_ticks[:] = sorted(levels)

    def _consume_side_deltas(
        self,
        levels: list[OrderbookLevel],
        side_levels: dict[int, OrderbookLevel],
        sorted_ticks: list[int],
    ) -> None:
        """Apply a delta batch to one side and rebuild sorted ticks once."""
        if not levels:
            return

        for level in levels:
            self._ensure_level_precision(level)
            ticks = level.ticks
            if level.lots == 0:
                side_levels.pop(ticks, None)
            else:
                side_levels[ticks] = level

        self._resort_ticks(side_levels, sorted_ticks)

    def _prune_better_bids(self, bid_ticks: int) -> None:
        """Remove stale bid levels that are better than an authoritative BBO bid."""
        cutoff = bisect_right(self._sorted_bid_ticks, bid_ticks)
        if cutoff >= len(self._sorted_bid_ticks):
            return
        stale_ticks = self._sorted_bid_ticks[cutoff:]
        for stale_tick in stale_ticks:
            self._bids.pop(stale_tick, None)
        del self._sorted_bid_ticks[cutoff:]

    def _prune_better_asks(self, ask_ticks: int) -> None:
        """Remove stale ask levels that are better than an authoritative BBO ask."""
        cutoff = bisect_left(self._sorted_ask_ticks, ask_ticks)
        if cutoff <= 0:
            return
        stale_ticks = self._sorted_ask_ticks[:cutoff]
        for stale_tick in stale_ticks:
            self._asks.pop(stale_tick, None)
        del self._sorted_ask_ticks[:cutoff]

    def consume_snapshot(
        self,
        asks: list[OrderbookLevel],
        bids: list[OrderbookLevel],
    ) -> None:
        """Consume a snapshot of the orderbook.

        Args:
            asks: List of ask levels.
            bids: List of bid levels.
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
        self._resort_ticks(self._asks, self._sorted_ask_ticks)

        for bid in bids:
            self._ensure_level_precision(bid)
            self._bids[bid.ticks] = bid
        self._resort_ticks(self._bids, self._sorted_bid_ticks)

        self._is_initialized = True
        self._is_populated = (
            len(self._sorted_bid_ticks) > 0 and len(self._sorted_ask_ticks) > 0
        )

    def consume_deltas(
        self,
        asks: list[OrderbookLevel],
        bids: list[OrderbookLevel],
    ) -> None:
        """Consume deltas of the orderbook.

        Args:
            asks: List of ask levels.
            bids: List of bid levels.
        """
        self._consume_side_deltas(asks, self._asks, self._sorted_ask_ticks)
        self._consume_side_deltas(bids, self._bids, self._sorted_bid_ticks)
        if self._is_initialized:
            self._refresh_population_state()

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
        self._ensure_level_precision(bid)
        self._ensure_level_precision(ask)

        bid_ticks, bid_lots = bid.ticks, bid.lots
        ask_ticks, ask_lots = ask.ticks, ask.lots

        if bid_lots == 0:
            if len(self._sorted_bid_ticks) > 0:
                best_bid_ticks = self._best_bid_ticks
                if best_bid_ticks in self._bids:
                    del self._bids[best_bid_ticks]
                    self._sorted_bid_ticks.pop()
        else:
            self._prune_better_bids(bid_ticks)
            if bid_ticks not in self._bids:
                insort(self._sorted_bid_ticks, bid_ticks)
            self._bids[bid_ticks] = bid

        if ask_lots == 0:
            if len(self._sorted_ask_ticks) > 0:
                best_ask_ticks = self._best_ask_ticks
                if best_ask_ticks in self._asks:
                    del self._asks[best_ask_ticks]
                    self._sorted_ask_ticks.pop(0)
        else:
            self._prune_better_asks(ask_ticks)
            if ask_ticks not in self._asks:
                insort(self._sorted_ask_ticks, ask_ticks)
            self._asks[ask_ticks] = ask

        self._is_initialized = True
        self._refresh_population_state()

    def get_asks(self, depth: int | None = None) -> list[OrderbookLevel]:
        """Get ask levels sorted by price (lowest first)."""
        self._ensure_initialized()
        if depth is None:
            return [self._asks[tick] for tick in self._sorted_ask_ticks]
        if depth <= 0:
            return []
        return [self._asks[tick] for tick in self._sorted_ask_ticks[:depth]]

    def get_bids(self, depth: int | None = None) -> list[OrderbookLevel]:
        """Get bid levels sorted by price (highest first)."""
        self._ensure_initialized()
        if depth is None:
            return [self._bids[tick] for tick in reversed(self._sorted_bid_ticks)]
        if depth <= 0:
            return []
        return [self._bids[tick] for tick in reversed(self._sorted_bid_ticks[-depth:])]

    def iter_asks(self, depth: int | None = None) -> Iterator[OrderbookLevel]:
        """Iterate over ask levels sorted by price (lowest -> highest)."""
        self._ensure_initialized()
        if depth is not None and depth <= 0:
            return
        ticks = (
            self._sorted_ask_ticks if depth is None else self._sorted_ask_ticks[:depth]
        )
        for tick in ticks:
            yield self._asks[tick]

    def iter_bids(self, depth: int | None = None) -> Iterator[OrderbookLevel]:
        """Iterate over bid levels sorted by price (highest -> lowest)."""
        self._ensure_initialized()
        if depth is not None and depth <= 0:
            return
        ticks = (
            reversed(self._sorted_bid_ticks)
            if depth is None
            else reversed(self._sorted_bid_ticks[-depth:])
        )
        for tick in ticks:
            yield self._bids[tick]

    def get_bbo(self) -> tuple[OrderbookLevel, OrderbookLevel]:
        """Get best bid and offer as a tuple."""
        self._ensure_bbo_available()
        best_bid_ticks = self._best_bid_ticks
        best_ask_ticks = self._best_ask_ticks
        return self._bids[best_bid_ticks], self._asks[best_ask_ticks]

    def get_bbo_spread(self) -> float:
        """Get the bid-ask spread."""
        self._ensure_bbo_available()
        best_ask_ticks = self._best_ask_ticks
        best_bid_ticks = self._best_bid_ticks
        spread_ticks = best_ask_ticks - best_bid_ticks
        return price_from_ticks(spread_ticks, self._tick_size)

    def get_mid_price(self) -> float:
        """Get the mid price between best bid and ask."""
        self._ensure_bbo_available()
        best_ask_ticks = self._best_ask_ticks
        best_bid_ticks = self._best_bid_ticks
        mid_ticks = (best_ask_ticks + best_bid_ticks) // 2
        return price_from_ticks(mid_ticks, self._tick_size)

    def get_wmid_price(self) -> float:
        """Get the weighted mid price between best bid and ask."""
        self._ensure_bbo_available()
        best_bid_ticks = self._best_bid_ticks
        best_ask_ticks = self._best_ask_ticks
        best_bid_lots = self._bids[best_bid_ticks].lots
        best_ask_lots = self._asks[best_ask_ticks].lots

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
        """Get the mid price between the price to buy and sell 'size' on the book."""
        self._ensure_bbo_available()
        if size == 0.0:
            return self.get_mid_price()

        mid_price = self.get_mid_price()
        if not is_base_currency:
            size = size / mid_price

        cum_bid_size = 0.0
        buy_price = None
        for tick in self._sorted_ask_ticks:
            level = self._asks[tick]
            if cum_bid_size + level.size >= size:
                buy_price = level.price
                break
            cum_bid_size += level.size
        if buy_price is None:
            return float("inf")

        cum_ask_size = 0.0
        sell_price = None
        for tick in reversed(self._sorted_bid_ticks):
            level = self._bids[tick]
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
        """Get the direct price impact if a theoretical size were to be
        executed on the book."""
        self._ensure_bbo_available()
        if size == 0.0:
            return 0.0

        mid_price = self.get_mid_price()
        if not is_base_currency:
            size = size / mid_price

        remaining_size = size
        total_cost = 0.0

        if is_buy:
            for tick in self._sorted_ask_ticks:
                level = self._asks[tick]
                consumed_size = min(remaining_size, level.size)
                total_cost += consumed_size * level.price
                remaining_size -= consumed_size

                if remaining_size <= 0.0:
                    break
        else:
            for tick in reversed(self._sorted_bid_ticks):
                level = self._bids[tick]
                consumed_size = min(remaining_size, level.size)
                total_cost += consumed_size * level.price
                remaining_size -= consumed_size

                if remaining_size <= 0.0:
                    break

        if remaining_size > 0.0:
            return float("inf")

        avg_execution_price = total_cost / size
        return abs(avg_execution_price - mid_price)

    def does_bbo_price_change(self, bid_price: float, ask_price: float) -> bool:
        """Check if the best bid/ask price will change."""
        self._ensure_bbo_available()
        my_bid_ticks = self._best_bid_ticks
        my_ask_ticks = self._best_ask_ticks
        other_bid_ticks = price_to_ticks_fast(bid_price, self._inv_tick_size)
        other_ask_ticks = price_to_ticks_fast(ask_price, self._inv_tick_size)
        return my_bid_ticks != other_bid_ticks or my_ask_ticks != other_ask_ticks

    def does_bbo_cross(self, bid_price: float, ask_price: float) -> bool:
        """Check if the best bid/ask price crosses with the given price."""
        self._ensure_bbo_available()
        my_bid_ticks = self._best_bid_ticks
        my_ask_ticks = self._best_ask_ticks
        other_bid_ticks = price_to_ticks_fast(bid_price, self._inv_tick_size)
        other_ask_ticks = price_to_ticks_fast(ask_price, self._inv_tick_size)
        return my_bid_ticks > other_ask_ticks or my_ask_ticks < other_bid_ticks
