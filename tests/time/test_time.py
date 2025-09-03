"""Tests for time utilities."""

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
    """Test basic time unit functionality."""

    def test_time_function_types(self):
        """Test that time functions return expected types."""
        assert isinstance(time_s(), int)
        assert isinstance(time_ms(), int)
        assert isinstance(time_us(), int)
        assert isinstance(time_ns(), int)
        assert isinstance(time_iso8601(), str)

    def test_time_units_monotonicity(self):
        """Test that time units are monotonically increasing."""
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
        """Test that time units have expected relative magnitudes."""
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
        """Test scale relationships between time units."""
        # Capture timestamps in order of decreasing precision to minimize cumulative drift
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
    """Test ISO 8601 time formatting and parsing."""

    def test_iso8601_format_validation(self):
        """Test that ISO 8601 output matches expected format."""
        text = time_iso8601()
        # Expected format: YYYY-MM-DDTHH:MM:SS.sssZ
        assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$", text)

    def test_iso8601_current_time_roundtrip(self):
        """Test ISO 8601 formatting and parsing roundtrip with current time."""
        s_now = time_s()
        text = time_iso8601()
        parsed = iso8601_to_unix(text)
        assert abs(parsed - s_now) < 2  # Allow up to 2 seconds difference

    def test_iso8601_with_custom_timestamp(self):
        """Test ISO 8601 formatting with custom timestamps."""
        # Test with a known timestamp (2023-01-01 00:00:00 UTC)
        test_timestamp = 1672531200.0  # 2023-01-01 00:00:00 UTC
        text = time_iso8601(test_timestamp)

        # Should format correctly
        assert text.startswith("2023-01-01T00:00:00")
        assert text.endswith("Z")

        # Should parse back to original timestamp
        parsed = iso8601_to_unix(text)
        assert abs(parsed - test_timestamp) < 1.0

    def test_iso8601_millisecond_precision(self):
        """Test that ISO 8601 formatting includes millisecond precision."""
        text = time_iso8601()

        # Extract millisecond part
        ms_part = text.split(".")[1][:-1]  # Remove 'Z' at end
        assert len(ms_part) == 3  # Should be 3 digits
        assert ms_part.isdigit()  # Should be numeric

    def test_iso8601_parse_validation(self):
        """Test ISO 8601 parsing with various formats."""
        # Test current time format
        current_text = time_iso8601()
        parsed_current = iso8601_to_unix(current_text)
        assert parsed_current > 0

        # Test with known formatted string
        known_iso = "2023-06-15T12:30:45.123Z"
        parsed_known = iso8601_to_unix(known_iso)
        assert parsed_known > 0
