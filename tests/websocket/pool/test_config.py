"""Configuration tests for WsPool.

Layer-1 tests verifying WsPoolConfig creation, default factory values,
and validation boundary enforcement.

Key coverage:
- Explicit construction with all fields.
- Default factory values (num_connections=5, evict_interval_s=60, hash_capacity=16384).
- Boundary validation: num_connections >= 2, evict_interval_s > 0, hash_capacity > 0.
"""

from __future__ import annotations

import pytest

from mm_toolbox.websocket.pool import WsPoolConfig


class TestWsPoolConfig:
    """Layer-1 tests for WsPoolConfig validation."""

    def test_valid_creation(self) -> None:
        """Given explicit parameters, When WsPoolConfig is constructed, Then all fields are stored correctly."""
        config = WsPoolConfig(
            num_connections=3,
            evict_interval_s=30,
            hash_capacity=4_096,
        )
        assert config.num_connections == 3
        assert config.evict_interval_s == 30
        assert config.hash_capacity == 4_096

    def test_default_factory(self) -> None:
        """Given no arguments, When WsPoolConfig.default is called, Then sensible defaults are returned."""
        config = WsPoolConfig.default()
        assert config.num_connections == 5
        assert config.evict_interval_s == 60
        assert config.hash_capacity == 16_384

    def test_validation_boundaries(self) -> None:
        """Given edge-case inputs, When validation runs, Then out-of-range values raise ValueError.

                Minimum thresholds exist to prevent nonsensical configs (e.g., a
        pool with one connection or zero eviction interval)."""
        WsPoolConfig(num_connections=2, evict_interval_s=1, hash_capacity=1)

        with pytest.raises(ValueError):
            WsPoolConfig(num_connections=1, evict_interval_s=60)

        with pytest.raises(ValueError):
            WsPoolConfig(num_connections=5, evict_interval_s=0)

        with pytest.raises(ValueError):
            WsPoolConfig(num_connections=5, evict_interval_s=60, hash_capacity=0)
