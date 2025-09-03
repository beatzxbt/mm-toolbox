"""Unit tests for websocket components."""

import pytest

from mm_toolbox.ringbuffer.bytes import BytesRingBuffer
from mm_toolbox.websocket.connection import (
    ConnectionState,
    LatencyTrackerState,
    WsConnectionConfig,
    WsConnectionState,
)
from mm_toolbox.websocket.pool import WsPool, WsPoolConfig
from mm_toolbox.websocket.single import WsSingle


class TestWsConnectionConfig:
    """Test WsConnectionConfig creation and validation only."""

    def test_basic_creation(self):
        """Test basic config creation."""
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

    def test_default_factory(self):
        """Test default factory method."""
        config = WsConnectionConfig.default("wss://api.binance.com/ws")
        assert config.wss_url == "wss://api.binance.com/ws"
        assert isinstance(config.conn_id, int) and config.conn_id > 0
        assert config.on_connect == []
        assert config.auto_reconnect is True

    def test_connection_id_uniqueness(self):
        """Test connection ID uniqueness across multiple calls."""
        ids = {WsConnectionConfig.default("wss://test.com").conn_id for _ in range(100)}
        assert (
            len(ids) >= 95
        )  # Allow small collision chance but ensure good distribution


class TestLatencyTrackerState:
    """Test LatencyTrackerState functionality only."""

    def test_default_creation(self):
        """Test default latency tracker."""
        tracker = LatencyTrackerState.default()
        assert tracker.latency_ms == 1000.0
        assert tracker.latency_ema is not None

    def test_ema_integration(self):
        """Test EMA updates work correctly."""
        tracker = LatencyTrackerState.default()
        tracker.latency_ema.update(50.0)
        tracker.latency_ms = 50.0

        assert tracker.latency_ms == 50.0
        assert tracker.latency_ema.get_value() < 1000.0


class TestWsConnectionState:
    """Test WsConnectionState properties only."""

    @pytest.fixture
    def sample_state(self):
        ringbuffer = BytesRingBuffer(max_capacity=64, only_insert_unique=False)
        latency = LatencyTrackerState.default()
        latency.latency_ms = 25.5

        return WsConnectionState(
            seq_id=42,
            state=ConnectionState.CONNECTED,
            ringbuffer=ringbuffer,
            latency=latency,
        )

    def test_state_properties(self, sample_state):
        """Test state property access."""
        assert sample_state.seq_id == 42
        assert sample_state.state == ConnectionState.CONNECTED
        assert sample_state.is_connected is True
        assert sample_state.latency_ms == 25.5

    def test_connection_state_transitions(self, sample_state):
        """Test connection state transitions."""
        assert sample_state.is_connected is True

        sample_state.state = ConnectionState.DISCONNECTED
        assert sample_state.is_connected is False

        sample_state.state = ConnectionState.CONNECTING
        assert sample_state.is_connected is False

    def test_recent_message_access(self, sample_state):
        """Test recent message retrieval."""
        messages = [b'{"msg": 1}', b'{"msg": 2}']
        for msg in messages:
            sample_state.ringbuffer.insert(msg)

        assert sample_state.recent_message == messages[-1]


class TestWsPoolConfig:
    """Test WsPoolConfig validation only."""

    def test_valid_creation(self):
        """Test valid pool config creation."""
        config = WsPoolConfig(num_connections=3, evict_interval_s=30)
        assert config.num_connections == 3
        assert config.evict_interval_s == 30

    def test_default_factory(self):
        """Test default pool config."""
        config = WsPoolConfig.default()
        assert config.num_connections == 5
        assert config.evict_interval_s == 60

    def test_validation_boundaries(self):
        """Test validation edge cases."""
        # Valid boundary values
        WsPoolConfig(num_connections=2, evict_interval_s=1)

        # Invalid values should raise
        with pytest.raises(ValueError):
            WsPoolConfig(num_connections=1, evict_interval_s=60)

        with pytest.raises(ValueError):
            WsPoolConfig(num_connections=5, evict_interval_s=0)


class TestWsSingleInterface:
    """Test WsSingle public interface only (no connection behavior)."""

    @pytest.fixture
    def config(self):
        return WsConnectionConfig.default("wss://test.com")

    def test_initialization(self, config):
        """Test WsSingle initialization."""
        ws = WsSingle(config)
        assert ws.get_config() is config
        assert ws.get_state() == ConnectionState.DISCONNECTED

    def test_callback_validation(self, config):
        """Test callback signature validation."""

        def valid_callback(msg: bytes) -> None:
            pass

        # Valid callback should work
        ws = WsSingle(config, on_message=valid_callback)
        assert ws._on_message is valid_callback

        # Invalid callbacks should raise
        with pytest.raises(ValueError):
            WsSingle(config, on_message=lambda: None)  # No parameters

        with pytest.raises(ValueError):
            WsSingle(config, on_message=lambda x, y: None)  # Too many parameters

    def test_configuration_updates(self, config):
        """Test configuration update methods."""
        ws = WsSingle(config)

        new_messages = [b'{"subscribe": "BTCUSDT"}']
        ws.set_on_connect(new_messages)
        assert ws.get_config().on_connect == new_messages

    def test_operations_when_disconnected(self, config):
        """Test operations when no connection exists."""
        ws = WsSingle(config)

        # These should not raise exceptions
        ws.send_data(b'{"test": "message"}')
        ws.close()
        assert ws.get_state() == ConnectionState.DISCONNECTED


class TestWsPoolInterface:
    """Test WsPool public interface only (no connection behavior)."""

    @pytest.fixture
    def config(self):
        return WsConnectionConfig.default("wss://test.com")

    def test_initialization(self, config):
        """Test WsPool initialization."""

        def callback(msg: bytes) -> None:
            pass

        pool = WsPool(config, callback)
        assert pool.get_state() == ConnectionState.DISCONNECTED
        assert pool.get_connection_count() == 0

    def test_callback_validation(self, config):
        """Test callback signature validation."""

        def valid_callback(msg: bytes) -> None:
            pass

        # Valid callback should work
        pool = WsPool(config, valid_callback)  # noqa: F841

        # Invalid callbacks should raise
        with pytest.raises(ValueError):
            WsPool(config, lambda: None)  # No parameters

    def test_configuration_updates(self, config):
        """Test configuration update methods."""
        pool = WsPool(config, None)

        new_messages = [b'{"subscribe": "ETHUSDT"}']
        pool.set_on_connect(new_messages)
        # Should not raise and update internal config

    def test_operations_when_disconnected(self, config):
        """Test operations when pool is disconnected."""
        pool = WsPool(config, None)

        # Should raise when trying to send data
        with pytest.raises(RuntimeError):
            pool.send_data(b'{"test": "message"}')

        # These should not raise
        pool.set_on_connect([b'{"test": "msg"}'])
        pool.close()


class TestConnectionStateEnum:
    """Test ConnectionState enum behavior only."""

    def test_enum_values(self):
        """Test enum values are correct."""
        assert ConnectionState.DISCONNECTED == 0
        assert ConnectionState.CONNECTING == 1
        assert ConnectionState.CONNECTED == 2

    def test_enum_ordering(self):
        """Test enum ordering relationships."""
        assert ConnectionState.DISCONNECTED < ConnectionState.CONNECTING
        assert ConnectionState.CONNECTING < ConnectionState.CONNECTED

    def test_enum_equality(self):
        """Test enum equality with integers."""
        assert ConnectionState.CONNECTED == 2
        assert ConnectionState.DISCONNECTED != 1
