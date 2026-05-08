"""Tests for standard logger configuration."""

import pytest

from mm_toolbox.logging.standard import LoggerConfig, LogLevel


class TestLoggerConfigDefaults:
    def test_default_values(self) -> None:
        cfg = LoggerConfig()
        assert cfg.base_level.name == "INFO"
        assert cfg.do_stdout is True
        assert cfg.flush_on_size is True
        assert cfg.flush_size_threshold == 1000
        assert cfg.flush_on_interval is False
        assert cfg.flush_interval_s == 1.0
        assert cfg.auto_flush_on_exit is True


class TestLoggerConfigValidation:
    def test_invalid_flush_size_threshold(self) -> None:
        with pytest.raises(ValueError):
            LoggerConfig(flush_on_size=True, flush_size_threshold=0)

    def test_negative_flush_size_threshold(self) -> None:
        with pytest.raises(ValueError):
            LoggerConfig(flush_on_size=True, flush_size_threshold=-1)

    def test_invalid_flush_interval(self) -> None:
        with pytest.raises(ValueError):
            LoggerConfig(flush_on_interval=True, flush_interval_s=0.0)

    def test_negative_flush_interval(self) -> None:
        with pytest.raises(ValueError):
            LoggerConfig(flush_on_interval=True, flush_interval_s=-1.0)

    def test_flush_interval_ignored_when_disabled(self) -> None:
        # Should not raise even with invalid interval if feature is off
        cfg = LoggerConfig(flush_on_interval=False, flush_interval_s=0.0)
        assert cfg.flush_interval_s == 0.0

    def test_invalid_format_string(self) -> None:
        with pytest.raises(ValueError):
            LoggerConfig(str_format="%(asctime)s - %(levelname)s")

    @pytest.mark.parametrize(
        "level",
        [
            LogLevel.TRACE,
            LogLevel.DEBUG,
            LogLevel.INFO,
            LogLevel.WARNING,
            LogLevel.ERROR,
        ],
    )
    def test_all_log_levels_accepted(self, level: LogLevel) -> None:
        cfg = LoggerConfig(base_level=level)
        assert cfg.base_level == level

    def test_custom_format_with_all_placeholders(self) -> None:
        fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        cfg = LoggerConfig(str_format=fmt)
        assert cfg.str_format == fmt
