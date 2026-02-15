import time
from collections.abc import Callable

import pytest

WAIT_TIMEOUT_S = 1.0


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add shared live-test options available across the full test suite."""
    try:
        parser.addoption(
            "--run-live",
            action="store_true",
            default=False,
            help="Run live tests that require internet connection (Binance streams)",
        )
    except ValueError:
        # Option may already be registered by a nested conftest.
        pass

    try:
        parser.addoption(
            "--live-timeout",
            action="store",
            default=30,
            type=int,
            help="Timeout for live tests in seconds",
        )
    except ValueError:
        # Option may already be registered by a nested conftest.
        pass


def pytest_configure(config: pytest.Config) -> None:
    """Register shared markers."""
    config.addinivalue_line(
        "markers", "live: mark test as requiring live internet connection"
    )
    config.addinivalue_line("markers", "slow: mark test as slow running")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip live tests unless explicitly enabled."""
    if config.getoption("--run-live"):
        return

    skip_live = pytest.mark.skip(reason="need --run-live option to run")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)


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
