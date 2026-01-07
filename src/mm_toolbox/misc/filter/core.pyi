"""Typed stubs for bounds-based value filter."""

class DataBoundsFilter:
    def __init__(self, threshold_pct: float) -> None:
        """Initialize with threshold percentage in (0, 100) (e.g., 1.0 for ±1%)."""
        ...

    def reset(self, value: float) -> None:
        """Reset bounds centered on value."""
        ...

    def check_and_update(self, value: float, reset: bool = ...) -> bool:
        """Return True if bounds were (re)initialized or updated."""
        ...
