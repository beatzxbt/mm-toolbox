from __future__ import annotations

class OrderbookFast:
    def __init__(
        self, tick_size: float, lot_size: float, max_num_levels: int = 500
    ) -> None: ...
    def clear(self) -> None: ...
    def get_bbo(self) -> tuple[float, float, float, float]: ...
    def get_mid_price(self) -> float: ...
    def get_bbo_spread(self) -> float: ...
    def is_crossed(self, other_bid_price: float, other_ask_price: float) -> bool: ...
    def get_imbalance(self, depth_pct: float) -> float: ...
    def consume_snapshot_raw(
        self,
        asks_price_ticks,
        asks_size_lots,
        asks_norders,
        bids_price_ticks,
        bids_size_lots,
        bids_norders,
    ) -> None: ...
    def consume_deltas_raw(
        self,
        asks_price_ticks,
        asks_size_lots,
        asks_norders,
        bids_price_ticks,
        bids_size_lots,
        bids_norders,
    ) -> None: ...
    def consume_bbo_raw(
        self,
        bid_price_ticks: int,
        bid_size_lots: int,
        bid_norder: int,
        ask_price_ticks: int,
        ask_size_lots: int,
        ask_norder: int,
    ) -> None: ...
    def consume_snapshot(
        self,
        asks_prices,
        asks_sizes,
        asks_norders,
        bids_prices,
        bids_sizes,
        bids_norders,
    ) -> None: ...
    def consume_snapshot_auto(
        self, asks_prices, asks_sizes, bids_prices, bids_sizes
    ) -> None: ...
    def consume_deltas(
        self,
        asks_prices,
        asks_sizes,
        asks_norders,
        bids_prices,
        bids_sizes,
        bids_norders,
    ) -> None: ...
    def consume_deltas_auto(
        self, asks_prices, asks_sizes, bids_prices, bids_sizes
    ) -> None: ...
    def consume_bbo(
        self,
        bid_price: float,
        bid_size: float,
        bid_norder: int,
        ask_price: float,
        ask_size: float,
        ask_norder: int,
    ) -> None: ...
    def consume_bbo_auto(
        self, bid_price: float, bid_size: float, ask_price: float, ask_size: float
    ) -> None: ...
