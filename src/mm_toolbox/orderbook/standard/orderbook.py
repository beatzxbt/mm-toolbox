from collections.abc import Iterator

from .level import OrderbookLevel, price_to_ticks, price_from_ticks


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
    ) -> None:
        """Initialize orderbook with tick size, lot size, and optional
        initial levels."""
        if tick_size <= 0.0:
            raise ValueError(f"Invalid tick_size; expected >0 but got {tick_size}")
        if lot_size <= 0.0:
            raise ValueError(f"Invalid lot_size; expected >0 but got {lot_size}")

        self._tick_size = tick_size
        self._lot_size = lot_size
        self._size = size

        # {ticks: OrderbookLevel}
        self._asks: dict[int, OrderbookLevel] = {}
        self._bids: dict[int, OrderbookLevel] = {}

        self._sorted_ask_ticks: list[int] = []
        self._sorted_bid_ticks: list[int] = []

        self._is_populated = False

        if initial_bids is not None and initial_asks is not None:
            self.consume_snapshot(asks=initial_asks, bids=initial_bids)

    def _ensure_populated(self) -> None:
        """Check if the orderbook is populated."""
        if not self._is_populated:
            raise ValueError("Orderbook is not populated.")

    def reset(self) -> None:
        """Reset the orderbook to its initial empty state."""
        self._asks.clear()
        self._bids.clear()
        self._sorted_ask_ticks.clear()
        self._sorted_bid_ticks.clear()
        self._is_populated = False

    def _maybe_remove_ask(self, ticks: int) -> None:
        """Remove an ask level if it exists."""
        if ticks in self._asks:
            del self._asks[ticks]
            self._sorted_ask_ticks.remove(ticks)

    def _maybe_remove_bid(self, ticks: int) -> None:
        """Remove a bid level if it exists."""
        if ticks in self._bids:
            del self._bids[ticks]
            self._sorted_bid_ticks.remove(ticks)

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
            ask.add_precision_info(self._tick_size, self._lot_size, unsafe=True)
            ticks = ask.ticks
            self._asks[ticks] = ask
            self._sorted_ask_ticks.append(ticks)
        self._sorted_ask_ticks.sort()

        for bid in bids:
            bid.add_precision_info(self._tick_size, self._lot_size, unsafe=True)
            ticks = bid.ticks
            self._bids[ticks] = bid
            self._sorted_bid_ticks.append(ticks)
        self._sorted_bid_ticks.sort(reverse=True)

        self._is_populated = True

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
        for ask in asks:
            ask.add_precision_info(self._tick_size, self._lot_size, unsafe=True)
            if ask.lots == 0:
                self._maybe_remove_ask(ask.ticks)
            else:
                ticks = ask.ticks
                if ticks not in self._asks:
                    self._sorted_ask_ticks.append(ticks)
                self._asks[ticks] = ask
        self._sorted_ask_ticks.sort()

        for bid in bids:
            bid.add_precision_info(self._tick_size, self._lot_size, unsafe=True)
            if bid.lots == 0:
                self._maybe_remove_bid(bid.ticks)
            else:
                ticks = bid.ticks
                if ticks not in self._bids:
                    self._sorted_bid_ticks.append(ticks)
                self._bids[ticks] = bid
        self._sorted_bid_ticks.sort(reverse=True)

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
        bid.add_precision_info(self._tick_size, self._lot_size, unsafe=True)
        ask.add_precision_info(self._tick_size, self._lot_size, unsafe=True)

        bid_ticks, bid_lots = bid.ticks, bid.lots
        ask_ticks, ask_lots = ask.ticks, ask.lots

        if bid_lots == 0:
            if len(self._sorted_bid_ticks) > 0:
                best_bid_ticks = self._sorted_bid_ticks[0]
                if best_bid_ticks in self._bids:
                    del self._bids[best_bid_ticks]
                    self._sorted_bid_ticks.pop(0)
        else:
            if bid_ticks in self._bids:
                self._bids[bid_ticks] = bid
            else:
                if len(self._sorted_bid_ticks) > 0:
                    old_best_ticks = self._sorted_bid_ticks[0]
                    if old_best_ticks != bid_ticks:
                        del self._bids[old_best_ticks]
                        self._sorted_bid_ticks.pop(0)

                self._bids[bid_ticks] = bid
                if bid_ticks not in self._sorted_bid_ticks:
                    self._sorted_bid_ticks.append(bid_ticks)
                    self._sorted_bid_ticks.sort(reverse=True)

        if ask_lots == 0:
            if len(self._sorted_ask_ticks) > 0:
                best_ask_ticks = self._sorted_ask_ticks[0]
                if best_ask_ticks in self._asks:
                    del self._asks[best_ask_ticks]
                    self._sorted_ask_ticks.pop(0)
        else:
            if ask_ticks in self._asks:
                self._asks[ask_ticks] = ask
            else:
                if len(self._sorted_ask_ticks) > 0:
                    old_best_ticks = self._sorted_ask_ticks[0]
                    if old_best_ticks != ask_ticks and old_best_ticks in self._asks:
                        del self._asks[old_best_ticks]
                        self._sorted_ask_ticks.pop(0)

                self._asks[ask_ticks] = ask
                if ask_ticks not in self._sorted_ask_ticks:
                    self._sorted_ask_ticks.append(ask_ticks)
                    self._sorted_ask_ticks.sort()

    def get_asks(self) -> list[OrderbookLevel]:
        """Get ask levels sorted by price (lowest first)."""
        self._ensure_populated()
        return [self._asks[tick] for tick in self._sorted_ask_ticks]

    def get_bids(self) -> list[OrderbookLevel]:
        """Get bid levels sorted by price (highest first)."""
        self._ensure_populated()
        return [self._bids[tick] for tick in self._sorted_bid_ticks]

    def iter_asks(self) -> Iterator[OrderbookLevel]:
        """Iterate over ask levels sorted by price (lowest -> highest)."""
        self._ensure_populated()
        for tick in self._sorted_ask_ticks:
            yield self._asks[tick]

    def iter_bids(self) -> Iterator[OrderbookLevel]:
        """Iterate over bid levels sorted by price (highest -> lowest)."""
        self._ensure_populated()
        for tick in self._sorted_bid_ticks:
            yield self._bids[tick]

    def get_bbo(self) -> tuple[OrderbookLevel, OrderbookLevel]:
        """Get best bid and offer as a tuple."""
        self._ensure_populated()
        best_bid_ticks = self._sorted_bid_ticks[0]
        best_ask_ticks = self._sorted_ask_ticks[0]
        return self._bids[best_bid_ticks], self._asks[best_ask_ticks]

    def get_bbo_spread(self) -> float:
        """Get the bid-ask spread."""
        self._ensure_populated()
        best_ask_ticks = self._sorted_ask_ticks[0]
        best_bid_ticks = self._sorted_bid_ticks[0]
        spread_ticks = best_ask_ticks - best_bid_ticks
        return price_from_ticks(spread_ticks, self._tick_size)

    def get_mid_price(self) -> float:
        """Get the mid price between best bid and ask."""
        self._ensure_populated()
        best_ask_ticks = self._sorted_ask_ticks[0]
        best_bid_ticks = self._sorted_bid_ticks[0]
        mid_ticks = (best_ask_ticks + best_bid_ticks) // 2
        return price_from_ticks(mid_ticks, self._tick_size)

    def get_wmid_price(self) -> float:
        """Get the weighted mid price between best bid and ask."""
        self._ensure_populated()
        best_bid_ticks = self._sorted_bid_ticks[0]
        best_ask_ticks = self._sorted_ask_ticks[0]
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
        self._ensure_populated()
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
        for tick in self._sorted_bid_ticks:
            level = self._bids[tick]
            if cum_ask_size + level.size >= size:
                sell_price = level.price
                break
            cum_ask_size += level.size
        if sell_price is None:
            return float("inf")

        return (buy_price + sell_price) / 2.0

    def get_volume_average_mid_price(
        self, size: float, is_base_currency: bool = True
    ) -> float:
        """Deprecated: Use get_volume_weighted_mid_price instead."""
        return self.get_volume_weighted_mid_price(size, is_base_currency)

    def get_price_impact(
        self, size: float, is_buy: bool, is_base_currency: bool = True
    ) -> float:
        """Get the direct price impact if a theoretical size were to be
        executed on the book."""
        self._ensure_populated()
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
            for tick in self._sorted_bid_ticks:
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
        self._ensure_populated()
        my_bid_ticks = self._sorted_bid_ticks[0]
        my_ask_ticks = self._sorted_ask_ticks[0]
        other_bid_ticks = price_to_ticks(bid_price, self._tick_size)
        other_ask_ticks = price_to_ticks(ask_price, self._tick_size)
        return my_bid_ticks != other_bid_ticks or my_ask_ticks != other_ask_ticks

    def does_bbo_cross(self, bid_price: float, ask_price: float) -> bool:
        """Check if the best bid/ask price crosses with the given price."""
        self._ensure_populated()
        my_bid_ticks = self._sorted_bid_ticks[0]
        my_ask_ticks = self._sorted_ask_ticks[0]
        other_bid_ticks = price_to_ticks(bid_price, self._tick_size)
        other_ask_ticks = price_to_ticks(ask_price, self._tick_size)
        return my_bid_ticks > other_ask_ticks or my_ask_ticks < other_bid_ticks
