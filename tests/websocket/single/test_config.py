"""Configuration tests for WsSingle-related settings.

Focuses on WsConnectionConfig validation as used by WsSingle.
"""

from __future__ import annotations


from mm_toolbox.websocket.connection import WsConnectionConfig


class TestWsConnectionConfig:
    """Test WsConnectionConfig creation and validation only."""

    def test_basic_creation(self) -> None:
        """Validate explicit config construction.

        Returns:
            None: This test does not return a value.
        """
        config = WsConnectionConfig(
            conn_id=12345,
            wss_url="wss://test.com/ws",
            on_connect=[b'{"test": "msg"}'],
            auto_reconnect=True,
        )
        assert config.conn_id == 12345
        assert config.wss_url == "wss://test.com/ws"
        assert config.on_connect == [b'{"test": "msg"}']
        assert config.auto_reconnect is True

    def test_default_factory(self) -> None:
        """Validate default config factory behavior.

        Returns:
            None: This test does not return a value.
        """
        config = WsConnectionConfig.default("wss://api.binance.com/ws")
        assert config.wss_url == "wss://api.binance.com/ws"
        assert isinstance(config.conn_id, int) and config.conn_id > 0
        assert config.on_connect == []
        assert config.auto_reconnect is True

    def test_connection_id_uniqueness(self) -> None:
        """Validate connection IDs are sufficiently unique.

        Returns:
            None: This test does not return a value.
        """
        ids = {WsConnectionConfig.default("wss://test.com").conn_id for _ in range(100)}
        assert len(ids) >= 95
