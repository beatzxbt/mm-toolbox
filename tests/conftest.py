import time
from collections.abc import Callable

import pytest

WAIT_TIMEOUT_S = 1.0


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add shared live-test options available across the full test suite.

    Registers ``--run-live`` and ``--live-timeout`` so that network-dependent
    tests can be skipped by default and enabled on demand.
    """
    Returns:
        None
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
    """Register shared markers.

    Adds ``live`` and ``slow`` markers so that the test collection phase can
    categorise and optionally skip them.

    Returns:
        None
    """
    config.addinivalue_line(
        "markers", "live: mark test as requiring live internet connection"
    )
    config.addinivalue_line("markers", "slow: mark test as slow running")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip live tests unless explicitly enabled.

    Given the test collection, when a test carries the ``live`` marker and the
    ``--run-live`` flag is absent, the test is skipped with a clear reason.

    Args:
        config: Pytest configuration object.
        items: List of collected test items.

    Returns:
        None
    """
    if config.getoption("--run-live"):
        return

    skip_live = pytest.mark.skip(reason="need --run-live option to run")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)


@pytest.fixture
def wait_for() -> Callable[[Callable[[], bool], float, float], bool]:
    """Return a helper to poll for a condition instead of sleeping a fixed amount.

    This fixture is especially useful for time-sensitive assertions (e.g. rate-
    limiter refills) where a fixed sleep would make tests flaky or slow.

    Returns:
        A polling function ``(predicate, timeout_s, interval_s) -> bool``.
    """

    def _wait_for(
        predicate: Callable[[], bool],
        timeout_s: float = WAIT_TIMEOUT_S,
        interval_s: float = 0.01,
    ) -> bool:
        """Poll *predicate* until it returns True or *timeout_s* elapses.

        Args:
            predicate: Zero-argument callable returning a boolean.
            timeout_s: Maximum time to wait in seconds.
            interval_s: Sleep duration between polls in seconds.

        Returns:
            True if *predicate* succeeded before the timeout, otherwise the
            final result of *predicate*.
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(interval_s)
        return predicate()

    return _wait_for
