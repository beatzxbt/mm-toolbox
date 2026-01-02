"""Tests for standard logger configuration."""

import pytest

from mm_toolbox.logging.standard import LoggerConfig


class TestLoggerConfig:
    """Validate LoggerConfig inputs."""

    def test_default_values(self) -> None:
        cfg = LoggerConfig()
        assert cfg.base_level.name == "INFO"
        assert cfg.do_stdout is True
        assert cfg.flush_interval_s == 1.0
        assert cfg.buffer_size == 10000

    def test_invalid_flush_interval(self) -> None:
        with pytest.raises(ValueError):
            LoggerConfig(flush_interval_s=0.0)

    def test_invalid_buffer_size(self) -> None:
        with pytest.raises(ValueError):
            LoggerConfig(buffer_size=0)
