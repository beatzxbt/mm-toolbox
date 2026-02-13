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
        assert config.max_frame_size == 1_048_576
        assert config.latency_ping_interval_ms == 100

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
        assert config.max_frame_size == 1_048_576
        assert config.latency_ping_interval_ms == 100

    def test_default_factory_custom_max_frame_size(self) -> None:
        """Validate default config factory accepts max_frame_size override.

        Returns:
            None: This test does not return a value.
        """
        config = WsConnectionConfig.default(
            "wss://api.binance.com/ws",
            max_frame_size=2048,
        )
        assert config.max_frame_size == 2048

    def test_default_factory_custom_latency_ping_interval(self) -> None:
        """Validate default config factory accepts latency ping interval override.

        Returns:
            None: This test does not return a value.
        """
        config = WsConnectionConfig.default(
            "wss://api.binance.com/ws",
            latency_ping_interval_ms=250,
        )
        assert config.latency_ping_interval_ms == 250

    def test_connection_id_uniqueness(self) -> None:
        """Validate connection IDs are sufficiently unique.

        Returns:
            None: This test does not return a value.
        """
        ids = {WsConnectionConfig.default("wss://test.com").conn_id for _ in range(100)}
        assert len(ids) >= 95

    def test_invalid_max_frame_size(self) -> None:
        """Validate max_frame_size must be positive.

        Returns:
            None: This test does not return a value.
        """
        try:
            WsConnectionConfig.default("wss://test.com", max_frame_size=0)
            raise AssertionError("Expected ValueError for max_frame_size=0")
        except ValueError as exc:
            assert "max_frame_size" in str(exc)

    def test_invalid_latency_ping_interval(self) -> None:
        """Validate latency_ping_interval_ms must be positive.

        Returns:
            None: This test does not return a value.
        """
        try:
            WsConnectionConfig.default(
                "wss://test.com",
                latency_ping_interval_ms=0,
            )
            raise AssertionError("Expected ValueError for latency_ping_interval_ms=0")
        except ValueError as exc:
            assert "latency_ping_interval_ms" in str(exc)
