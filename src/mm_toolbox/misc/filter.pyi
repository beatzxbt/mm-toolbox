class DataBoundsFilter:
    def check_and_update(self, value: float, reset: bool = ...) -> bool:
        """Checks if value is outside current bounds or if reset is True, updates bounds if so.

        Args:
            value (float): Value to check.
            reset (bool): If True, always update bounds.

        Returns:
            bool: True if bounds were updated, False otherwise.

        """
        ...
