"""Layer 1 — Primitives tests for advanced logger configuration.

Covers ``LoggerConfig`` validation: accepted log levels, format-string
requirements, path types, flush-interval bounds, shared-memory capacity,
batch limits, and boolean coercion.
"""

import math

import pytest

from mm_toolbox.logging.advanced.config import LoggerConfig
from mm_toolbox.logging.advanced.pylog import PyLogLevel


class TestLoggerConfig:
    """Layer 1 — ``LoggerConfig`` primitive validation tests."""

    @pytest.mark.parametrize(
        "base_level",
        [
            PyLogLevel.TRACE,
            PyLogLevel.DEBUG,
            PyLogLevel.INFO,
            PyLogLevel.WARNING,
            PyLogLevel.ERROR,
        ],
    )
    def test_valid_base_level(self, base_level):
        """Given any valid ``PyLogLevel``, construction succeeds."""
        config = LoggerConfig(base_level=base_level)
        assert config.base_level == base_level

    def test_default_base_level(self):
        """Given no arguments, the default level is ``INFO``."""
        config = LoggerConfig()
        assert config.base_level == PyLogLevel.INFO

    @pytest.mark.parametrize("invalid_level", ["INFO", 2.5, -1, 5, 255])
    def test_invalid_base_level_raises(self, invalid_level):
        with pytest.raises(ValueError, match="Unknown base log level"):
            LoggerConfig(base_level=invalid_level)

    @pytest.mark.parametrize(
        "invalid_format",
        [
            "%(asctime)s [%(levelname)s] %(name)s",
            "",
            123,
            None,
        ],
    )
    def test_invalid_str_format(self, invalid_format):
        """Given a format string missing ``%(message)s`` or a non-string, construction raises."""
        with pytest.raises((ValueError, TypeError)):
            LoggerConfig(str_format=invalid_format)

    @pytest.mark.parametrize("invalid_path", [123, b"ipc://test"])
    def test_invalid_path_type(self, invalid_path):
        """Given a non-string path, construction raises TypeError."""
        with pytest.raises(TypeError):
            LoggerConfig(path=invalid_path)

    @pytest.mark.parametrize(
        "invalid_interval", [0.0, -1.0, math.nan, math.inf, -math.inf, "1.0", None]
    )
    def test_invalid_flush_interval(self, invalid_interval):
        """Given a non-positive or non-numeric interval, construction raises."""
        with pytest.raises((ValueError, TypeError)):
            LoggerConfig(flush_interval_s=invalid_interval)

    @pytest.mark.parametrize("invalid_shm_capacity_bytes", [0, -1, -1024])
    def test_invalid_shm_capacity_bytes(self, invalid_shm_capacity_bytes):
        """Given a non-positive capacity, construction raises ValueError."""
        with pytest.raises(ValueError):
            LoggerConfig(shm_capacity_bytes=invalid_shm_capacity_bytes)

    @pytest.mark.parametrize("invalid_max_batch_messages", [0, -1, -100])
    def test_invalid_max_batch_messages(self, invalid_max_batch_messages):
        """Given a non-positive limit, construction raises ValueError."""
        with pytest.raises(ValueError):
            LoggerConfig(max_batch_messages=invalid_max_batch_messages)

    @pytest.mark.parametrize("invalid_max_batch_bytes", [0, -1, -1024])
    def test_invalid_max_batch_bytes(self, invalid_max_batch_bytes):
        """Given a non-positive limit, construction raises ValueError."""
        with pytest.raises(ValueError):
            LoggerConfig(max_batch_bytes=invalid_max_batch_bytes)
