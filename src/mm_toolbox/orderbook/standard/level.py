"""Orderbook level primitives and data structures.

Provides conversion utilities between float prices/sizes and integer
ticks/lots, plus the ``OrderbookLevel`` struct used to represent a
single price level on either side of an orderbook.
"""

from __future__ import annotations

from math import floor
from typing import Self

from msgspec import Struct


def price_to_ticks(price: float, tick_size: float) -> int:
    """Convert a price to integer ticks.

    Args:
        price (float): The price to convert.
        tick_size (float): The minimum price increment.

    Returns:
        int: Equivalent number of ticks.

    """
    return int(floor(price / tick_size))


def size_to_lots(size: float, lot_size: float) -> int:
    """Convert a size to integer lots.

    Args:
        size (float): The size to convert.
        lot_size (float): The minimum size increment.

    Returns:
        int: Equivalent number of lots.

    """
    return int(floor(size / lot_size))


def price_to_ticks_fast(price: float, inv_tick_size: float) -> int:
    """Convert a price to integer ticks using a pre-computed inverse tick size.

    Args:
        price (float): The price to convert.
        inv_tick_size (float): Pre-computed ``1.0 / tick_size``.

    Returns:
        int: Equivalent number of ticks.

    """
    return int(price * inv_tick_size)


def size_to_lots_fast(size: float, inv_lot_size: float) -> int:
    """Convert a size to integer lots using a pre-computed inverse lot size.

    Args:
        size (float): The size to convert.
        inv_lot_size (float): Pre-computed ``1.0 / lot_size``.

    Returns:
        int: Equivalent number of lots.

    """
    return int(size * inv_lot_size)


def price_from_ticks(ticks: int, tick_size: float) -> float:
    """Convert integer ticks back to a price.

    Args:
        ticks (int): Number of ticks.
        tick_size (float): The minimum price increment.

    Returns:
        float: Reconstructed price.

    """
    return ticks * tick_size


def size_from_lots(lots: int, lot_size: float) -> float:
    """Convert integer lots back to a size.

    Args:
        lots (int): Number of lots.
        lot_size (float): The minimum size increment.

    Returns:
        float: Reconstructed size.

    """
    return lots * lot_size


class OrderbookLevel(Struct):
    """Represents a single price level on one side of an orderbook.

    Attributes:
        price (float): Price at this level.
        size (float): Aggregate size at this level.
        norders (int): Number of orders contributing to the size.
        ticks (int): Integer price ticks. ``-1`` means uninitialized.
        lots (int): Integer size lots. ``-1`` means uninitialized.

    """

    price: float
    size: float
    norders: int
    ticks: int = -1  # To prevent type checker errors, -1 signals uninitialized
    lots: int = -1  # To prevent type checker errors, -1 signals uninitialized

    def __post_init__(self) -> None:
        """Validate orderbook level data after initialization."""
        if self.price < 0.0:
            raise ValueError(f"Invalid price; expected >=0 but got {self.price}")
        if self.size < 0.0:
            raise ValueError(f"Invalid size; expected >=0 but got {self.size}")
        if self.norders < 0:
            raise ValueError(f"Invalid norders; expected >=0 but got {self.norders}")

    @property
    def value(self) -> float:
        """Total notional value at this level.

        Returns:
            float: ``price * size``.

        """
        return self.price * self.size

    def has_precision_info(self) -> bool:
        """Check if ticks and lots have been computed.

        Returns:
            bool: True when both ``ticks`` and ``lots`` are non-negative.

        """
        return self.ticks >= 0 and self.lots >= 0

    def reset(self) -> None:
        """Reset all fields to empty / uninitialized values."""
        self.price = 0.0
        self.size = 0.0
        self.norders = 0
        self.ticks = -1
        self.lots = -1

    def add_precision_info(
        self,
        inv_tick_size: float,
        inv_lot_size: float,
        unsafe: bool = False,
    ) -> None:
        """Compute and store integer ticks and lots for this level.

        Args:
            inv_tick_size (float): Pre-computed ``1.0 / tick_size``.
            inv_lot_size (float): Pre-computed ``1.0 / lot_size``.
            unsafe (bool): If True, skip validation and early-exit when
                precision info is already present. Defaults to False.

        Raises:
            ValueError: If ``inv_tick_size`` or ``inv_lot_size`` are not
                positive when ``unsafe`` is False.

        """
        if unsafe and self.has_precision_info():
            return

        if not unsafe:
            if inv_tick_size <= 0.0:
                raise ValueError(
                    f"Invalid inv_tick_size; expected >0 but got {inv_tick_size}"
                )
            if inv_lot_size <= 0.0:
                raise ValueError(
                    f"Invalid inv_lot_size; expected >0 but got {inv_lot_size}"
                )

        self.ticks = price_to_ticks_fast(self.price, inv_tick_size)
        self.lots = size_to_lots_fast(self.size, inv_lot_size)

    @classmethod
    def from_values(
        cls, price: float, size: float, norders: int, tick_size: float, lot_size: float
    ) -> Self:
        """Create an OrderbookLevel with precision info automatically populated."""
        return cls(
            price=price,
            size=size,
            norders=norders,
            ticks=price_to_ticks(price, tick_size),
            lots=size_to_lots(size, lot_size),
        )
