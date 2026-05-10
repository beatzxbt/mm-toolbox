"""Configuration tests for WsSingle-related settings.

Layer-1 tests focusing on WsConnectionConfig validation as used by WsSingle.
Covers explicit construction, default factory behavior, custom overrides,
connection ID uniqueness, and validation of positive constraints.
"""

from __future__ import annotations


from mm_toolbox.websocket.connection import WsConnectionConfig


class TestWsConnectionConfig:
    """Layer-1 tests for WsConnectionConfig creation and validation."""

    def test_basic_creation(self) -> None:
        """Given explicit parameters, When WsConnectionConfig is constructed, Then all fields are stored correctly."""
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
        assert config.latency_ping_interval_ms == 1000

    def test_default_factory(self) -> None:
        """Given no arguments, When WsConnectionConfig.default is called, Then sensible defaults are returned."""
        config = WsConnectionConfig.default("wss://api.binance.com/ws")
        assert config.wss_url == "wss://api.binance.com/ws"
        assert isinstance(config.conn_id, int) and config.conn_id > 0
        assert config.on_connect == []
        assert config.auto_reconnect is True
        assert config.max_frame_size == 1_048_576
        assert config.latency_ping_interval_ms == 1000

    def test_default_factory_custom_max_frame_size(self) -> None:
        """Given a max_frame_size override, When default is called, Then the override is respected."""
        config = WsConnectionConfig.default(
            "wss://api.binance.com/ws",
            max_frame_size=2048,
        )
        assert config.max_frame_size == 2048

    def test_default_factory_custom_latency_ping_interval(self) -> None:
        """Given a latency_ping_interval_ms override, When default is called, Then the override is respected."""
        config = WsConnectionConfig.default(
            "wss://api.binance.com/ws",
            latency_ping_interval_ms=250,
        )
        assert config.latency_ping_interval_ms == 250

    def test_connection_id_uniqueness(self) -> None:
        """Given 100 default factory calls, When IDs are compared, Then at least 95 are unique.

                Collisions are statistically unlikely but possible; this test
        ensures the entropy source is functioning."""
        ids = {WsConnectionConfig.default("wss://test.com").conn_id for _ in range(100)}
        assert len(ids) >= 95

    def test_invalid_max_frame_size(self) -> None:
        """Given max_frame_size=0, When default is called, Then ValueError is raised.

        Zero or negative frame sizes would break the websocket handshake."""
        try:
            WsConnectionConfig.default("wss://test.com", max_frame_size=0)
            raise AssertionError("Expected ValueError for max_frame_size=0")
        except ValueError as exc:
            assert "max_frame_size" in str(exc)

    def test_invalid_latency_ping_interval(self) -> None:
        """Given latency_ping_interval_ms=0, When default is called, Then ValueError is raised.

        Zero or negative intervals would disable heartbeat checks entirely."""
        try:
            WsConnectionConfig.default(
                "wss://test.com",
                latency_ping_interval_ms=0,
            )
            raise AssertionError("Expected ValueError for latency_ping_interval_ms=0")
        except ValueError as exc:
            assert "latency_ping_interval_ms" in str(exc)
