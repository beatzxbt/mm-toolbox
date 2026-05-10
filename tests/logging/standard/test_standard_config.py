"""Layer 1 — Primitives tests for standard logger configuration.

Covers ``LoggerConfig`` default values, validation of flush thresholds and
intervals, format-string requirements, and acceptance of every ``LogLevel``
enum member.
"""

import pytest

from mm_toolbox.logging.standard import LoggerConfig, LogLevel


class TestLoggerConfigDefaults:
    """Layer 1 — Default-value primitive tests."""

    def test_default_values(self) -> None:
        """Given no constructor arguments, defaults match the documented specification."""
        cfg = LoggerConfig()
        assert cfg.base_level.name == "INFO"
        assert cfg.do_stdout is True
        assert cfg.flush_on_size is True
        assert cfg.flush_size_threshold == 1000
        assert cfg.flush_on_interval is False
        assert cfg.flush_interval_s == 1.0
        assert cfg.auto_flush_on_exit is True


class TestLoggerConfigValidation:
    """Layer 1 — Input-validation primitive tests."""

    def test_invalid_flush_size_threshold(self) -> None:
        """Given ``flush_on_size=True`` and threshold=0, construction raises ValueError."""
        with pytest.raises(ValueError):
            LoggerConfig(flush_on_size=True, flush_size_threshold=0)

    def test_negative_flush_size_threshold(self) -> None:
        """Given a negative flush threshold, construction raises ValueError."""
        with pytest.raises(ValueError):
            LoggerConfig(flush_on_size=True, flush_size_threshold=-1)

    def test_invalid_flush_interval(self) -> None:
        """Given ``flush_on_interval=True`` and interval=0.0, construction raises ValueError."""
        with pytest.raises(ValueError):
            LoggerConfig(flush_on_interval=True, flush_interval_s=0.0)

    def test_negative_flush_interval(self) -> None:
        """Given a negative flush interval, construction raises ValueError."""
        with pytest.raises(ValueError):
            LoggerConfig(flush_on_interval=True, flush_interval_s=-1.0)

    def test_flush_interval_ignored_when_disabled(self) -> None:
        """Given ``flush_on_interval=False``, an invalid interval is accepted without error."""
        cfg = LoggerConfig(flush_on_interval=False, flush_interval_s=0.0)
        assert cfg.flush_interval_s == 0.0

    def test_invalid_format_string(self) -> None:
        """Given a format string missing ``%(message)s``, construction raises ValueError."""
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
        """Given any valid ``LogLevel``, construction succeeds and stores the exact level."""
        cfg = LoggerConfig(base_level=level)
        assert cfg.base_level == level

    def test_custom_format_with_all_placeholders(self) -> None:
        """Given a format string with all required placeholders, construction succeeds."""
        fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        cfg = LoggerConfig(str_format=fmt)
        assert cfg.str_format == fmt
