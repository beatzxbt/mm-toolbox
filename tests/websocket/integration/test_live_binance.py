"""Live Binance integration tests for websocket components.

Run with: pytest tests/websocket/integration/test_live_binance.py --run-live
"""

from __future__ import annotations

import asyncio

import msgspec
import pytest

from mm_toolbox.websocket.connection import ConnectionState, WsConnectionConfig
from mm_toolbox.websocket.pool import WsPool, WsPoolConfig
from mm_toolbox.websocket.single import WsSingle


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip live tests unless --run-live is specified.

    Args:
        item (pytest.Item): Test item under execution.

    Returns:
        None: This hook does not return a value.
    """
    if "live" in item.keywords and not item.config.getoption(
        "--run-live", default=False
    ):
        pytest.skip("Live tests require --run-live option and internet connection")


@pytest.mark.live
class TestLiveFunctionality:
    """Essential live functionality validation against Binance streams."""

    @pytest.mark.asyncio
    async def test_single_connection_real_stream(self) -> None:
        """Validate WsSingle receives real Binance book ticker messages.

        Returns:
            None: This test does not return a value.
        """
        config = WsConnectionConfig.default(
            "wss://fstream.binance.com/ws/btcusdt@bookTicker"
        )

        received_data = []
        valid_count = 0

        def message_handler(msg: bytes) -> None:
            """Handle incoming Binance book ticker messages.

            Args:
                msg (bytes): Raw websocket payload.

            Returns:
                None: This handler does not return a value.
            """
            nonlocal valid_count
            try:
                data = msgspec.json.decode(msg)
                received_data.append(data)
                if (
                    isinstance(data, dict)
                    and "s" in data
                    and data["s"] == "BTCUSDT"
                    and "b" in data
                    and "a" in data
                ):
                    valid_count += 1
            except msgspec.DecodeError:
                return None

        ws = WsSingle(config, on_message=message_handler)

        async with ws:
            await asyncio.sleep(2.0)
            assert ws.get_state() == ConnectionState.CONNECTED
            await asyncio.sleep(3.0)
            assert len(received_data) > 0
            assert valid_count > 0

    @pytest.mark.asyncio
    async def test_pool_with_real_streams(self) -> None:
        """Validate WsPool receives messages from Binance streams.

        Returns:
            None: This test does not return a value.
        """
        config = WsConnectionConfig.default(
            "wss://fstream.binance.com/ws/btcusdt@bookTicker"
        )
        pool_config = WsPoolConfig(num_connections=2, evict_interval_s=60)
        received_count = 0

        def message_handler(msg: bytes) -> None:
            """Count incoming Binance messages.

            Args:
                msg (bytes): Raw websocket payload.

            Returns:
                None: This handler does not return a value.
            """
            nonlocal received_count
            try:
                data = msgspec.json.decode(msg)
                if isinstance(data, dict) and "s" in data:
                    received_count += 1
            except msgspec.DecodeError:
                return None

        pool = await WsPool.new(config, message_handler, pool_config)

        async with pool:
            await asyncio.sleep(3.0)
            assert pool.get_state() == ConnectionState.CONNECTED
            assert pool.get_connection_count() > 0
            await asyncio.sleep(5.0)
            assert received_count > 0

    @pytest.mark.asyncio
    async def test_latency_measurement_real_connection(self) -> None:
        """Validate latency measurement on a live connection.

        Returns:
            None: This test does not return a value.
        """
        config = WsConnectionConfig.default(
            "wss://fstream.binance.com/ws/btcusdt@bookTicker"
        )
        ws = WsSingle(config)

        async with ws:
            await asyncio.sleep(2.0)
            await asyncio.sleep(8.0)
            assert ws.get_state() == ConnectionState.CONNECTED
            latency_ms = ws._ws_conn.get_state().latency_ms
            assert 0 < latency_ms < 2000

    @pytest.mark.asyncio
    async def test_data_sending_to_real_stream(self) -> None:
        """Validate send_data path against live Binance stream.

        Returns:
            None: This test does not return a value.
        """
        config = WsConnectionConfig.default(
            "wss://fstream.binance.com/ws/btcusdt@bookTicker"
        )
        ws = WsSingle(config)

        async with ws:
            await asyncio.sleep(2.0)
            ws.send_data(b'{"method": "LIST_SUBSCRIPTIONS", "id": 1}')
            await asyncio.sleep(2.0)
            assert ws.get_state() == ConnectionState.CONNECTED
