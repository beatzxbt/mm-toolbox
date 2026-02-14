"""Configuration tests for WsPool."""

from __future__ import annotations

import pytest

from mm_toolbox.websocket.pool import WsPoolConfig


class TestWsPoolConfig:
    """Test WsPoolConfig validation only."""

    def test_valid_creation(self) -> None:
        """Validate explicit pool configuration.

        Returns:
            None: This test does not return a value.
        """
        config = WsPoolConfig(
            num_connections=3,
            evict_interval_s=30,
            hash_capacity=4_096,
        )
        assert config.num_connections == 3
        assert config.evict_interval_s == 30
        assert config.hash_capacity == 4_096

    def test_default_factory(self) -> None:
        """Validate default pool configuration.

        Returns:
            None: This test does not return a value.
        """
        config = WsPoolConfig.default()
        assert config.num_connections == 5
        assert config.evict_interval_s == 60
        assert config.hash_capacity == 16_384

    def test_validation_boundaries(self) -> None:
        """Validate boundary conditions for pool config.

        Returns:
            None: This test does not return a value.
        """
        WsPoolConfig(num_connections=2, evict_interval_s=1, hash_capacity=1)

        with pytest.raises(ValueError):
            WsPoolConfig(num_connections=1, evict_interval_s=60)

        with pytest.raises(ValueError):
            WsPoolConfig(num_connections=5, evict_interval_s=0)

        with pytest.raises(ValueError):
            WsPoolConfig(num_connections=5, evict_interval_s=60, hash_capacity=0)
