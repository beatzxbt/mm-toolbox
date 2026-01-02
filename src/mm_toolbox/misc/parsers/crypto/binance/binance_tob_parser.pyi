from typing import Tuple

def parse_bbo(payload: bytes) -> Tuple[float, float, float, float]:
    """Parse Binance bookTicker bytes to (bid_price, bid_qty, ask_price, ask_qty)."""

def parse_bbo_cached(
    payload: bytes, symbol: bytes, unsafe_fast_path: bool = False
) -> Tuple[float, float, float, float]:
    """Parse using cached offsets per symbol; falls back when shape drifts. Set unsafe_fast_path=True to skip guards."""
