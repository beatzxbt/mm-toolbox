"""Private grid construction and exact impact-band arithmetic."""

from decimal import Decimal
from math import floor, log10


def reciprocal(quantum: float) -> float:
    """Cache a canonical decimal scale without reciprocal division drift."""
    exponent = round(log10(quantum))
    if quantum == float(f"1e{exponent}"):
        return float(f"1e{-exponent}")
    return 1.0 / quantum


def canonical_scale(scale: float) -> float:
    """Recognize a caller's slightly inexact reciprocal of a decimal grid."""
    if 1.0 <= scale < 4503599627370496.0 and scale != floor(scale):
        nearest = round(scale)
        if abs(scale - nearest) <= nearest * 2.220446049250313e-16:
            remaining = nearest
            while remaining > 1 and remaining % 10 == 0:
                remaining //= 10
            if remaining == 1:
                return float(nearest)
    return scale


def impact_limit_tick(anchor: int, bps: float, is_buy: bool) -> int:
    """Floor a buy boundary or ceil a sell boundary using exact integer ratios."""
    if bps.is_integer():
        numerator, denominator = int(bps), 1
    else:
        numerator, denominator = Decimal(str(bps)).as_integer_ratio()
    distance = anchor * numerator // (10_000 * denominator)
    if is_buy:
        return min((1 << 64) - 1, anchor + distance)
    return max(0, anchor - distance)
