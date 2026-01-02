"""Bounds-based change filter for streaming numeric values.

This filter keeps dynamic lower/upper bounds around a reference value,
defined by a percentage threshold. When a new value falls outside the
current bounds, the reference is reset to that value and new bounds are set.
"""


cdef class DataBoundsFilter:
    """Maintains dynamic bounds around a reference value by percentage."""

    def __cinit__(self, double threshold_pct):
        self._threshold = threshold_pct / 100.0
        self._lower_bound = 0.0
        self._upper_bound = 0.0

    cpdef void reset(self, double value) noexcept:
        """Reset bounds centered on value."""
        self._lower_bound = value * (1.0 - self._threshold)
        self._upper_bound = value * (1.0 + self._threshold)

    cpdef bint check_and_update(self, double value, bint reset=False) noexcept:
        """Return True if bounds were (re)initialized or updated.

        If reset is True, bounds are reset unconditionally around value.
        Otherwise, bounds are reset only when value is outside current bounds.
        """
        if reset:
            self.reset(value)
            return True

        if value < self._lower_bound or value > self._upper_bound:
            self.reset(value)
            return True

        return False


