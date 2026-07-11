"""Python-visible raw orderbook level wrappers."""

from __future__ import annotations

from typing import Optional

import numpy as np
import numpy.typing as npt

class PyOrderbookLevel:
    """Python-visible raw orderbook level.

    Attributes:
        price: Raw price value supplied by the caller.
        size: Raw size value supplied by the caller. Zero is allowed as a
            deletion marker.
        norders: Number of orders represented at the level.
    """

    @property
    def price(self) -> float:
        """Raw price for this level."""
        ...

    @property
    def size(self) -> float:
        """Raw size for this level."""
        ...

    @property
    def norders(self) -> int:
        """Number of orders represented at this level."""
        ...

    def __init__(
        self,
        price: float,
        size: float,
        norders: int = 1,
        verify_values: bool = True,
    ) -> None:
        """Initialize a raw orderbook level.

        Args:
            price: Raw price value. Must be finite and non-negative when
                verify_values is true.
            size: Raw size value. Must be finite and non-negative when
                verify_values is true.
            norders: Number of orders represented at the level.
            verify_values: Whether to validate raw price and size immediately.

        Raises:
            ValueError: If verification is enabled and price or size is invalid.
        """
        ...

    def __repr__(self) -> str:
        """Return a debugging representation of the raw level."""
        ...

class PyOrderbookLevels:
    """Python-visible container for raw orderbook levels."""

    def __init__(self) -> None:
        """Initialize an empty level container."""
        ...

    @staticmethod
    def from_list(
        prices: list[float],
        sizes: list[float],
        norders: Optional[list[int]] = None,
        verify_values: bool = True,
    ) -> PyOrderbookLevels:
        """Build levels from Python lists of raw values.

        Args:
            prices: Raw price values.
            sizes: Raw size values.
            norders: Optional order counts. Defaults to one order per level.
            verify_values: Whether to validate raw prices and sizes.

        Returns:
            A PyOrderbookLevels instance containing the supplied rows.

        Raises:
            ValueError: If lengths differ or verification finds invalid values.
        """
        ...

    @staticmethod
    def from_numpy(
        prices: npt.NDArray[np.float64],
        sizes: npt.NDArray[np.float64],
        norders: Optional[npt.NDArray[np.uint64]] = None,
        verify_values: bool = True,
    ) -> PyOrderbookLevels:
        """Build levels from NumPy arrays of raw values.

        Args:
            prices: One-dimensional raw price array.
            sizes: One-dimensional raw size array with the same length.
            norders: Optional one-dimensional order count array.
            verify_values: Whether to validate raw prices and sizes.

        Returns:
            A PyOrderbookLevels instance containing the supplied rows.

        Raises:
            ValueError: If lengths differ or verification finds invalid values.
        """
        ...
