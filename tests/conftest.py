import time
from collections.abc import Callable

import pytest

WAIT_TIMEOUT_S = 1.0


@pytest.fixture
def wait_for() -> Callable[[Callable[[], bool], float, float], bool]:
    """Return a helper to poll for a condition instead of sleeping a fixed amount."""

    def _wait_for(
        predicate: Callable[[], bool],
        timeout_s: float = WAIT_TIMEOUT_S,
        interval_s: float = 0.01,
    ) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(interval_s)
        return predicate()

    return _wait_for
