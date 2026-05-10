"""Async context manager tests for WsSingle.

Layer-2 tests validating WsSingle async context manager semantics,
interface behaviors without a connection, and edge cases like close
failures and reentrant usage.
"""

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
        ws: WsSingle instance.
        expected: Expected connection state.
        timeout_s: Timeout in seconds.

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
    """Layer-2 tests for WsSingle interface behaviors that do not require a connection."""

    @pytest.fixture
    def config(self):
        """Build a config for WsSingle tests.

        Returns:
            Config instance for WsSingle.
        """
        from mm_toolbox.websocket.connection import WsConnectionConfig

        return WsConnectionConfig.default("wss://test.com")

    def test_initialization(self, config) -> None:
        """Given a fresh WsSingle, Then config and state accessors return the expected defaults."""
        ws = WsSingle(config)
        assert ws.get_config() is config
        assert ws.get_state() == ConnectionState.DISCONNECTED

    def test_configuration_updates(self, config) -> None:
        """Given a WsSingle, When set_on_connect is called, Then the config is updated."""
        ws = WsSingle(config)
        new_messages = [b'{"subscribe": "BTCUSDT"}']
        ws.set_on_connect(new_messages)
        assert ws.get_config().on_connect == new_messages

    def test_operations_when_disconnected(self, config) -> None:
        """Given a disconnected WsSingle, When send_data or close is called, Then no exception is raised and state remains DISCONNECTED."""
        ws = WsSingle(config)
        ws.send_data(b'{"test": "message"}')
        ws.close()
        assert ws.get_state() == ConnectionState.DISCONNECTED


@pytest.mark.asyncio
class TestWsSingleAsyncContextManager:
    """Layer-2 tests for WsSingle async context manager behavior."""

    async def test_async_with_normal_exit(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Given a local server, When WsSingle is used with async with, Then it connects, sends, and exits to DISCONNECTED cleanly."""
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
        """Given an exception inside the context body, When raised, Then the pool still cleans up and ends in DISCONNECTED."""
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
        """Given a connection whose close() raises, When the context exits, Then the exception is swallowed and state ends DISCONNECTED.

        Close failures must not propagate into user code; otherwise a
network blip during teardown would crash the application."""
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
                        """Initialize the proxy wrapper."""
                        self._conn = conn

                    def close(self) -> None:
                        """Raise an error to simulate close failures."""
                        raise RuntimeError("boom")

                    def get_state(self):
                        """Return the underlying connection state."""
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
        """Given two WsSingle instances, When nested in async with blocks, Then both connect and send concurrently without interference."""
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
        """Given auto_reconnect=True, When WsSingle is used with async with, Then it connects and exits cleanly despite reconnection configuration."""
        async with basic_server:
            config = connection_config_factory(basic_server, auto_reconnect=True)
            async with WsSingle(config) as ws:
                await wait_for_single_state(
                    ws, ConnectionState.CONNECTED, timeout_s=2.0
                )
            assert ws.get_state() == ConnectionState.DISCONNECTED

    async def test_aenter_fallback_when_reconnect_fails(
        self,
        basic_server,
        connection_config_factory,
        monkeypatch,
    ) -> None:
        """Given auto_reconnect=True and a failing reconnect, When __aenter__ runs, Then it raises RuntimeError rather than hanging.

        This prevents indefinite hangs when the server is unreachable and
the reconnect iterator would loop forever."""
        async with basic_server:
            config = connection_config_factory(basic_server, auto_reconnect=True)

            async def _raising_reconnect(*args, **kwargs):
                raise RuntimeError("reconnect fail")

            monkeypatch.setattr(
                "mm_toolbox.websocket.connection.ws_connect",
                _raising_reconnect,
            )
            ws = WsSingle(config)
            with pytest.raises(RuntimeError):
                async with ws:
                    pass
