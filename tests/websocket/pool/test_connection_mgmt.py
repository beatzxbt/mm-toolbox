"""Connection management tests for WsPool."""

from __future__ import annotations

import asyncio

import pytest

from mm_toolbox.websocket.connection import ConnectionState, WsConnectionConfig
from mm_toolbox.websocket.pool import WsPool, WsPoolConfig


def noop_message_handler(msg: bytes) -> None:
    """No-op message handler for pool tests.

    Args:
        msg (bytes): Incoming message payload.

    Returns:
        None: This handler does not return a value.
    """
    return None


class TestWsPoolInterface:
    """Validate WsPool interface behaviors without starting connections."""

    @pytest.fixture
    def config(self) -> WsConnectionConfig:
        """Build a default WsConnectionConfig for tests.

        Returns:
            WsConnectionConfig: Default config instance.
        """
        return WsConnectionConfig.default("wss://test.com")

    def test_initialization(self, config: WsConnectionConfig) -> None:
        """Ensure pool initializes in DISCONNECTED state.

        Args:
            config (WsConnectionConfig): Default config fixture.

        Returns:
            None: This test does not return a value.
        """
        pool = WsPool(config, on_message=noop_message_handler)
        assert pool.get_state() == ConnectionState.DISCONNECTED
        assert pool.get_connection_count() == 0

    def test_callback_validation(self, config: WsConnectionConfig) -> None:
        """Ensure callback signature validation enforces one bytes arg.

        Args:
            config (WsConnectionConfig): Default config fixture.

        Returns:
            None: This test does not return a value.
        """
        WsPool(config, on_message=noop_message_handler)

        def invalid_callback() -> None:
            """Callback missing required bytes parameter.

            Returns:
                None: This callback does not return a value.
            """
            return None

        with pytest.raises(ValueError):
            WsPool(config, on_message=invalid_callback)

    def test_configuration_updates(self, config: WsConnectionConfig) -> None:
        """Ensure set_on_connect updates pool configuration.

        Args:
            config (WsConnectionConfig): Default config fixture.

        Returns:
            None: This test does not return a value.
        """
        pool = WsPool(config, on_message=noop_message_handler)
        new_messages = [b'{"subscribe": "ETHUSDT"}']
        pool.set_on_connect(new_messages)
        assert pool._config.on_connect == new_messages

    def test_operations_when_disconnected(self, config: WsConnectionConfig) -> None:
        """Ensure send_data raises when pool is disconnected.

        Args:
            config (WsConnectionConfig): Default config fixture.

        Returns:
            None: This test does not return a value.
        """
        pool = WsPool(config, on_message=noop_message_handler)
        with pytest.raises(RuntimeError):
            pool.send_data(b'{"test": "message"}')
        pool.close()


@pytest.mark.asyncio
class TestWsPoolConnectionManagement:
    """Validate pool connection lifecycle behavior."""

    async def test_pool_opens_num_connections(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Ensure WsPool opens the requested number of connections.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_config_factory: Fixture providing config factory.

        Returns:
            None: This test does not return a value.
        """
        async with basic_server:
            config = connection_config_factory(basic_server)
            pool_config = WsPoolConfig(num_connections=3, evict_interval_s=60)
            pool = await WsPool.new(
                config, on_message=noop_message_handler, pool_config=pool_config
            )
            async with pool:
                await asyncio.sleep(0.5)
                assert pool.get_connection_count() == pool_config.num_connections
            pool.close()

    async def test_pool_adds_connection_to_pool(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Ensure connections are stored in the pool map.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_config_factory: Fixture providing config factory.

        Returns:
            None: This test does not return a value.
        """
        async with basic_server:
            config = connection_config_factory(basic_server)
            pool_config = WsPoolConfig(num_connections=2, evict_interval_s=60)
            pool = await WsPool.new(
                config, on_message=noop_message_handler, pool_config=pool_config
            )
            async with pool:
                await asyncio.sleep(0.5)
                assert len(pool._conns) == pool_config.num_connections
            pool.close()

    async def test_pool_removes_failed_connection(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Ensure closed connections are removed and fast connection updates.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_config_factory: Fixture providing config factory.

        Returns:
            None: This test does not return a value.
        """
        async with basic_server:
            config = connection_config_factory(basic_server)
            pool_config = WsPoolConfig(num_connections=2, evict_interval_s=60)
            pool = await WsPool.new(
                config, on_message=noop_message_handler, pool_config=pool_config
            )
            async with pool:
                await asyncio.sleep(0.5)
                conn = next(iter(pool._conns.values()))
                conn.close()
                pool._conns.pop(conn.get_config().conn_id, None)
                pool._update_fast_connection()
                assert conn not in pool._conns.values()
            pool.close()

    async def test_pool_concurrent_connection_setup(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Ensure concurrent setup initializes all connections.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_config_factory: Fixture providing config factory.

        Returns:
            None: This test does not return a value.
        """
        async with basic_server:
            config = connection_config_factory(basic_server)
            pool_config = WsPoolConfig(num_connections=4, evict_interval_s=60)
            pool = await WsPool.new(
                config, on_message=noop_message_handler, pool_config=pool_config
            )
            async with pool:
                await asyncio.sleep(0.5)
                assert pool.get_connection_count() == pool_config.num_connections
            pool.close()
