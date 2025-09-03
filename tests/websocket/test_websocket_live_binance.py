"""Essential live tests with real Binance WebSocket streams.

Only critical live validation. Run with: pytest tests/websocket/test_websocket_live_binance.py --run-live
"""

import asyncio

import msgspec
import pytest

from mm_toolbox.websocket.connection import ConnectionState, WsConnectionConfig
from mm_toolbox.websocket.pool import WsPool, WsPoolConfig
from mm_toolbox.websocket.single import WsSingle


def pytest_runtest_setup(item):
    """Skip live tests unless --run-live is specified."""
    if "live" in item.keywords and not item.config.getoption(
        "--run-live", default=False
    ):
        pytest.skip("Live tests require --run-live option and internet connection")


@pytest.mark.live
class TestLiveFunctionality:
    """Essential live functionality validation."""

    @pytest.mark.asyncio
    async def test_single_connection_real_stream(self):
        """Test single connection works with real Binance stream."""
        config = WsConnectionConfig.default(
            "wss://fstream.binance.com/ws/btcusdt@bookTicker"
        )

        received_data = []
        valid_count = 0

        def message_handler(msg: bytes):
            nonlocal valid_count
            try:
                data = msgspec.json.decode(msg)
                received_data.append(data)

                # Validate book ticker structure
                if isinstance(data, dict) and "s" in data and data["s"] == "BTCUSDT":
                    if "b" in data and "a" in data:  # bid and ask
                        valid_count += 1
            except msgspec.DecodeError:
                pass

        ws = WsSingle(config, on_message=message_handler)

        async with ws:
            # Wait for connection
            await asyncio.sleep(2.0)
            assert ws.get_state().is_connected

            # Wait for data
            await asyncio.sleep(3.0)

            # Validate we received real data
            assert len(received_data) > 0
            assert valid_count > 0
            print(
                f"Received {len(received_data)} messages, {valid_count} valid book tickers"
            )

    @pytest.mark.asyncio
    async def test_pool_with_real_streams(self):
        """Test pool works with real Binance streams."""
        config = WsConnectionConfig.default(
            "wss://fstream.binance.com/ws/btcusdt@bookTicker"
        )
        pool_config = WsPoolConfig(num_connections=2, evict_interval_s=60)

        received_count = 0

        def message_handler(msg: bytes):
            nonlocal received_count
            try:
                data = msgspec.json.decode(msg)
                if isinstance(data, dict) and "s" in data:
                    received_count += 1
            except msgspec.DecodeError:
                pass

        pool = await WsPool.new(config, message_handler, pool_config)

        async with pool:
            # Wait for connections
            await asyncio.sleep(3.0)
            assert pool.get_state() == ConnectionState.CONNECTED
            assert pool.get_connection_count() > 0

            # Wait for data
            await asyncio.sleep(5.0)

            # Validate pool functionality
            assert received_count > 0
            print(
                f"Pool received {received_count} messages from {pool.get_connection_count()} connections"
            )

    @pytest.mark.asyncio
    async def test_latency_measurement_real_connection(self):
        """Test latency measurement works with real connection."""
        config = WsConnectionConfig.default(
            "wss://fstream.binance.com/ws/btcusdt@bookTicker"
        )
        ws = WsSingle(config)

        async with ws:
            # Wait for connection establishment
            await asyncio.sleep(2.0)

            # Wait for latency tracking to stabilize
            await asyncio.sleep(8.0)

            state = ws.get_state()
            assert state.is_connected

            latency_ms = state.latency_ms
            assert 0 < latency_ms < 2000  # Reasonable latency bounds
            print(f"Measured latency: {latency_ms:.2f}ms")

    @pytest.mark.asyncio
    async def test_data_sending_to_real_stream(self):
        """Test sending data to real stream (will likely get ignored but tests the path)."""
        config = WsConnectionConfig.default(
            "wss://fstream.binance.com/ws/btcusdt@bookTicker"
        )
        ws = WsSingle(config)

        async with ws:
            await asyncio.sleep(2.0)

            # Send a test message (will be ignored by Binance but tests our send path)
            test_message = b'{"method": "LIST_SUBSCRIPTIONS", "id": 1}'
            ws.send_data(test_message)

            # Wait a bit more to ensure no errors
            await asyncio.sleep(2.0)

            # Connection should remain stable
            assert ws.get_state().is_connected
