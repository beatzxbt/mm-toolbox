"""Tests for time utilities.

Layer 1 tests: validate time unit functions (seconds, milliseconds, microseconds,
nanoseconds) for monotonicity, type correctness, and scale relationships.
Also tests ISO 8601 formatting and parsing with various precision levels.
"""

import re
import time as _time

from mm_toolbox.time import (
    iso8601_to_unix,
    time_iso8601,
    time_ms,
    time_ns,
    time_s,
    time_us,
)


class TestTime:
    """Layer 1: Test basic time unit functionality."""

    def test_time_function_types(self):
        """Given time functions, When called, Then return expected types."""
        assert isinstance(time_s(), int)
        assert isinstance(time_ms(), int)
        assert isinstance(time_us(), int)
        assert isinstance(time_ns(), int)
        assert isinstance(time_iso8601(), str)

    def test_time_units_monotonicity(self):
        """Given time passage, When time functions called, Then values monotonically increase."""
        s0 = time_s()
        ms0 = time_ms()
        us0 = time_us()
        ns0 = time_ns()

        _time.sleep(0.01)

        s1 = time_s()
        ms1 = time_ms()
        us1 = time_us()
        ns1 = time_ns()

        assert s1 >= s0
        assert ms1 >= ms0
        assert us1 >= us0
        assert ns1 >= ns0

    def test_time_units_relative_values(self):
        """Given concurrent time capture, When compared, Then correct relative magnitudes."""
        # Capture all timestamps as close together as possible
        start_s = time_s()
        start_ms = time_ms()
        start_us = time_us()
        start_ns = time_ns()

        # Basic sanity checks - each unit should be larger than the previous
        assert start_ms > start_s
        assert start_us > start_ms
        assert start_ns > start_us

    def test_time_units_scale_relationships(self):
        """Given concurrent time capture, When scale-checked, Then approximate relationships hold."""
        # Capture timestamps in order of decreasing precision to minimize
        # cumulative drift
        start_s = time_s()
        start_ms = time_ms()
        start_us = time_us()
        start_ns = time_ns()

        # scale checks (approximate) - allow for execution delays between calls
        # On some systems, consecutive clock_gettime() calls can have significant delays
        assert (
            abs(start_ms - start_s * 1000) < 1000
        )  # Allow up to 1 second for system variability
        assert abs(start_us - start_ms * 1000) < 10_000  # Allow microsecond precision
        assert (
            abs(start_ns - start_us * 1000) < 10_000_000
        )  # Allow nanosecond precision


class TestTimeISO8601:
    """Layer 1: Test ISO 8601 time formatting and parsing."""

    def test_iso8601_format_validation(self):
        """Given current time, When formatted, Then matches expected ISO 8601 pattern."""
        text = time_iso8601()
        # Expected format: YYYY-MM-DDTHH:MM:SS.sssZ
        assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$", text)

    def test_iso8601_current_time_roundtrip(self):
        """Given current time, When formatted and parsed, Then approximately recovers original."""
        s_now = time_s()
        text = time_iso8601()
        parsed = iso8601_to_unix(text)
        assert abs(parsed - s_now) < 2  # Allow up to 2 seconds difference

    def test_iso8601_with_custom_timestamp(self):
        """Given known timestamp, When formatted, Then correct ISO 8601 string produced."""
        # Test with a known timestamp (2023-01-01 00:00:00 UTC)
        test_timestamp = 1672531200.0  # 2023-01-01 00:00:00 UTC
        text = time_iso8601(test_timestamp)

        # Should format correctly
        assert text.startswith("2023-01-01T00:00:00")
        assert text.endswith("Z")

        # Should parse back to original timestamp
        parsed = iso8601_to_unix(text)
        assert abs(parsed - test_timestamp) < 1.0

    def test_iso8601_with_millisecond_timestamp(self):
        """Given millisecond unix timestamp, When formatted, Then includes milliseconds."""
        test_timestamp_ms = 1672531200123.0  # 2023-01-01 00:00:00.123 UTC
        text = time_iso8601(test_timestamp_ms)
        assert text == "2023-01-01T00:00:00.123Z"

    def test_iso8601_with_microsecond_timestamp(self):
        """Given microsecond unix timestamp, When formatted, Then includes microseconds."""
        test_timestamp_us = 1672531200123456.0  # 2023-01-01 00:00:00.123456 UTC
        text = time_iso8601(test_timestamp_us)
        assert text == "2023-01-01T00:00:00.123456Z"

    def test_iso8601_with_nanosecond_timestamp(self):
        """Given nanosecond unix timestamp, When formatted, Then includes nanoseconds."""
        test_timestamp_ns = 1672531200123456789.0
        text = time_iso8601(test_timestamp_ns)

        assert text.startswith("2023-01-01T00:00:00.")
        assert text.endswith("Z")
        fractional = text.split(".")[1][:-1]
        assert len(fractional) == 9
        assert fractional.isdigit()

    def test_iso8601_millisecond_precision(self):
        """Given current time, When formatted, Then includes millisecond precision."""
        text = time_iso8601()

        # Extract millisecond part
        ms_part = text.split(".")[1][:-1]  # Remove 'Z' at end
        assert len(ms_part) == 3  # Should be 3 digits
        assert ms_part.isdigit()  # Should be numeric

    def test_iso8601_parse_validation(self):
        """Given various ISO 8601 strings, When parsed, Then returns positive timestamps."""
        # Test current time format
        current_text = time_iso8601()
        parsed_current = iso8601_to_unix(current_text)
        assert parsed_current > 0

        # Test with known formatted string
        known_iso = "2023-06-15T12:30:45.123Z"
        parsed_known = iso8601_to_unix(known_iso)
        assert parsed_known > 0
