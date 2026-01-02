from math import floor
from typing import Self

from msgspec import Struct


def price_to_ticks(price: float, tick_size: float) -> int:
    """Convert a price to integer ticks."""
    return int(floor(price / tick_size))


def size_to_lots(size: float, lot_size: float) -> int:
    """Convert a size to integer lots."""
    return int(floor(size / lot_size))


def price_from_ticks(ticks: int, tick_size: float) -> float:
    """Convert integer ticks to price."""
    return ticks * tick_size


def size_from_lots(lots: int, lot_size: float) -> float:
    """Convert integer lots to size."""
    return lots * lot_size


class OrderbookLevel(Struct):
    """Represents a single level in an orderbook."""

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
        """Get the value of the orderbook level."""
        return self.price * self.size

    def has_precision_info(self) -> bool:
        """Check if the orderbook level has precision info."""
        return self.ticks < -1 and self.lots < -1

    def reset(self) -> None:
        """Reset the orderbook level to an empty state."""
        self.price = 0.0
        self.size = 0.0
        self.norders = 0
        self.ticks = -1
        self.lots = -1

    def add_precision_info(
        self, tick_size: float, lot_size: float, unsafe: bool = False
    ) -> None:
        """Add precision information to the orderbook level.

        The unsafe flag is used to bypass validation in favor of performance.
        """
        if not unsafe:
            if tick_size <= 0.0:
                raise ValueError(f"Invalid tick_size; expected >0 but got {tick_size}")
            if lot_size <= 0.0:
                raise ValueError(f"Invalid lot_size; expected >0 but got {lot_size}")

        self.ticks = price_to_ticks(self.price, tick_size)
        self.lots = size_to_lots(self.size, lot_size)

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
