"""Test configuration for websocket tests."""

import pytest


def pytest_addoption(parser):
    """Add custom command line options for websocket tests."""
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="Run live tests that require internet connection (Binance streams)",
    )
    parser.addoption(
        "--live-timeout",
        action="store",
        default=30,
        type=int,
        help="Timeout for live tests in seconds",
    )


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "live: mark test as requiring live internet connection"
    )
    config.addinivalue_line("markers", "slow: mark test as slow running")


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on command line options."""
    if config.getoption("--run-live"):
        # When --run-live is specified, run all tests
        return

    # Skip live tests when --run-live is not specified
    skip_live = pytest.mark.skip(reason="need --run-live option to run")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)


@pytest.fixture(scope="session")
def live_test_config():
    """Configuration for live tests."""
    return {
        "binance_futures_base": "wss://fstream.binance.com/ws",
        "binance_spot_base": "wss://stream.binance.com:9443/ws",
        "test_symbols": ["btcusdt", "ethusdt", "bnbusdt"],
        "connection_timeout": 10,
        "message_wait_time": 5,
        "latency_timeout": 15,
    }


@pytest.fixture
def mock_websocket_server():
    """Mock WebSocket server for testing (placeholder)."""
    # This could be expanded to create an actual mock WebSocket server
    # for more comprehensive testing without requiring internet
    return {
        "url": "wss://mock.test.server/ws",
        "responses": [
            b'{"type": "test", "data": "mock_data"}',
            b'{"type": "ping", "timestamp": 1234567890}',
        ],
    }
