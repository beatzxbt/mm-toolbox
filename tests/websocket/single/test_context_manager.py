"""Async context manager tests for WsSingle."""

from __future__ import annotations

import asyncio

import pytest

from mm_toolbox.websocket.connection import ConnectionState
from mm_toolbox.websocket.single import WsSingle


async def wait_for_single_state(
    ws: WsSingle, expected: ConnectionState, timeout_s: float = 2.0
) -> None:
    """Wait for a WsSingle instance to reach the expected state.

    Args:
        ws (WsSingle): WsSingle instance.
        expected (ConnectionState): Expected connection state.
        timeout_s (float): Timeout in seconds.

    Returns:
        None: This helper does not return a value.

    Raises:
        AssertionError: If state is not reached in time.
    """
    start = asyncio.get_running_loop().time()
    while (asyncio.get_running_loop().time() - start) < timeout_s:
        if ws.get_state() == expected:
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"Timed out waiting for {expected}")


class TestWsSingleInterface:
    """Validate WsSingle interface behaviors that do not require a connection."""

    @pytest.fixture
    def config(self):
        """Build a config for WsSingle tests.

        Returns:
            WsConnectionConfig: Config instance for WsSingle.
        """
        from mm_toolbox.websocket.connection import WsConnectionConfig

        return WsConnectionConfig.default("wss://test.com")

    def test_initialization(self, config) -> None:
        """Validate initial state and config access.

        Args:
            config: WsConnectionConfig fixture.

        Returns:
            None: This test does not return a value.
        """
        ws = WsSingle(config)
        assert ws.get_config() is config
        assert ws.get_state() == ConnectionState.DISCONNECTED

    def test_configuration_updates(self, config) -> None:
        """Validate set_on_connect updates configuration.

        Args:
            config: WsConnectionConfig fixture.

        Returns:
            None: This test does not return a value.
        """
        ws = WsSingle(config)
        new_messages = [b'{"subscribe": "BTCUSDT"}']
        ws.set_on_connect(new_messages)
        assert ws.get_config().on_connect == new_messages

    def test_operations_when_disconnected(self, config) -> None:
        """Ensure operations are safe when no connection exists.

        Args:
            config: WsConnectionConfig fixture.

        Returns:
            None: This test does not return a value.
        """
        ws = WsSingle(config)
        ws.send_data(b'{"test": "message"}')
        ws.close()
        assert ws.get_state() == ConnectionState.DISCONNECTED


@pytest.mark.asyncio
class TestWsSingleAsyncContextManager:
    """Validate WsSingle async context manager behavior."""

    async def test_async_with_normal_exit(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Ensure async with opens and closes the connection cleanly.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_config_factory: Fixture providing config factory.

        Returns:
            None: This test does not return a value.
        """
        async with basic_server:
            config = connection_config_factory(basic_server)
            async with WsSingle(config) as ws:
                await wait_for_single_state(
                    ws, ConnectionState.CONNECTED, timeout_s=2.0
                )
                ws.send_data(b"test")
            assert ws.get_state() == ConnectionState.DISCONNECTED

    async def test_async_with_exception_in_body(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Ensure exceptions inside context still clean up resources.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_config_factory: Fixture providing config factory.

        Returns:
            None: This test does not return a value.
        """
        async with basic_server:
            config = connection_config_factory(basic_server)
            with pytest.raises(ValueError):
                async with WsSingle(config) as ws:
                    await wait_for_single_state(
                        ws, ConnectionState.CONNECTED, timeout_s=2.0
                    )
                    raise ValueError("test error")
            assert ws.get_state() == ConnectionState.DISCONNECTED

    async def test_async_with_close_exception(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Ensure context exit does not crash if close raises.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_config_factory: Fixture providing config factory.

        Returns:
            None: This test does not return a value.
        """
        async with basic_server:
            config = connection_config_factory(basic_server)
            ws = WsSingle(config)
            async with ws:
                await wait_for_single_state(
                    ws, ConnectionState.CONNECTED, timeout_s=2.0
                )

                original_conn = ws._ws_conn
                assert original_conn is not None

                class _CloseFailConn:
                    """Proxy connection that raises on close."""

                    def __init__(self, conn) -> None:
                        """Initialize the proxy wrapper.

                        Args:
                            conn: The underlying connection instance.

                        Returns:
                            None: This initializer does not return a value.
                        """
                        self._conn = conn

                    def close(self) -> None:
                        """Raise an error to simulate close failures.

                        Returns:
                            None: This helper does not return a value.
                        """
                        raise RuntimeError("boom")

                    def get_state(self):
                        """Return the underlying connection state.

                        Returns:
                            WsConnectionState: Current state snapshot.
                        """
                        return self._conn.get_state()

                ws._ws_conn = _CloseFailConn(original_conn)
            original_conn.close()
            ws._ws_conn = original_conn
            assert ws.get_state() == ConnectionState.DISCONNECTED

    async def test_nested_async_contexts(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Ensure multiple WsSingle contexts operate concurrently.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_config_factory: Fixture providing config factory.

        Returns:
            None: This test does not return a value.
        """
        async with basic_server:
            config1 = connection_config_factory(basic_server)
            config2 = connection_config_factory(basic_server)
            async with WsSingle(config1) as ws1:
                async with WsSingle(config2) as ws2:
                    await wait_for_single_state(
                        ws1, ConnectionState.CONNECTED, timeout_s=2.0
                    )
                    await wait_for_single_state(
                        ws2, ConnectionState.CONNECTED, timeout_s=2.0
                    )
                    ws1.send_data(b"1")
                    ws2.send_data(b"2")
                    await asyncio.sleep(0.1)
            assert ws1.get_state() == ConnectionState.DISCONNECTED
            assert ws2.get_state() == ConnectionState.DISCONNECTED

    async def test_context_manager_with_reconnect(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Ensure auto_reconnect configurations still close cleanly.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_config_factory: Fixture providing config factory.

        Returns:
            None: This test does not return a value.
        """
        async with basic_server:
            config = connection_config_factory(basic_server, auto_reconnect=True)
            async with WsSingle(config) as ws:
                await wait_for_single_state(
                    ws, ConnectionState.CONNECTED, timeout_s=2.0
                )
            assert ws.get_state() == ConnectionState.DISCONNECTED
