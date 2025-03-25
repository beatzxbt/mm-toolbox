import unittest
import time
from mm_toolbox.time import (
    time_s,
    time_ms,
    time_us,
    time_ns,
    time_iso8601,
    iso8601_to_unix,
    unix_to_iso8601,
)


class TestTime(unittest.TestCase):
    def test_time_s(self):
        # Test that time_s returns a value close to Python's time.time()
        python_time = time.time()
        custom_time = time_s()
        self.assertAlmostEqual(custom_time, python_time, delta=0.1)
    
    def test_time_ms(self):
        # Test that time_ms returns milliseconds (1000x seconds)
        s_time = time_s()
        ms_time = time_ms()
        # Check that ms is approximately 1000x seconds
        self.assertAlmostEqual(ms_time, s_time * 1000, delta=100)
    
    def test_time_us(self):
        # Test that time_us returns microseconds (1000x milliseconds)
        ms_time = time_ms()
        us_time = time_us()
        # Check that us is approximately 1000x milliseconds
        self.assertAlmostEqual(us_time, ms_time * 1000, delta=1000)
    
    def test_time_ns(self):
        # Test that time_ns returns nanoseconds (1000x microseconds)
        us_time = time_us()
        ns_time = time_ns()
        # Check that ns is approximately 1000x microseconds
        self.assertAlmostEqual(ns_time, us_time * 1000, delta=10000)
    
    def test_time_iso8601(self):
        time_s_now = time_s()
        result = time_iso8601()
        expected_pattern = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z"
        self.assertRegex(result, expected_pattern)

        # Verify the conversion back to Unix timestamp is accurate
        unix_time_from_iso8601 = iso8601_to_unix(result)
        self.assertAlmostEqual(unix_time_from_iso8601, time_s_now, delta=2)
    
    def test_iso8601_to_unix(self):
        # Test with a known timestamp
        iso_time = "2023-01-01T12:00:00.000Z"
        unix_time = iso8601_to_unix(iso_time)
        # 2023-01-01T12:00:00Z corresponds to a specific Unix timestamp
        # This may need adjustment based on timezone handling
        expected_timestamp = 1672574400.0  # This is 2023-01-01T12:00:00Z in Unix time
        self.assertAlmostEqual(unix_time, expected_timestamp, delta=1)
    
    def test_unix_to_iso8601(self):
        # Test with a known Unix timestamp (seconds)
        unix_time = 1672574400.0  # 2023-01-01T12:00:00Z
        iso_time = unix_to_iso8601(unix_time)
        # The exact format might vary slightly due to local timezone
        # So we'll check that it contains the expected date
        self.assertIn("2023-01-01T12:00:00.000Z", iso_time)

        # Test with a known Unix timestamp (milliseconds)
        unix_time = 1672574400000.0  # 2023-01-01T12:00:00.000Z
        iso_time = unix_to_iso8601(unix_time)
        # The exact format might vary slightly due to local timezone
        # So we'll check that it contains the expected date
        self.assertIn("2023-01-01T12:00:00.000Z", iso_time)

        # Test with a known Unix timestamp (microseconds)
        unix_time = 1672574400000000.0  # 2023-01-01T12:00:00.000000Z
        iso_time = unix_to_iso8601(unix_time)
        # The exact format might vary slightly due to local timezone
        # So we'll check that it contains the expected date
        self.assertIn("2023-01-01T12:00:00.000000Z", iso_time)

        # Test with a known Unix timestamp (nanoseconds)
        unix_time = 1672574400000000000.0  # 2023-01-01T12:00:00.000000000Z
        iso_time = unix_to_iso8601(unix_time)
        # The exact format might vary slightly due to local timezone
        # So we'll check that it contains the expected date
        self.assertIn("2023-01-01T12:00:00.000000000Z", iso_time)
    
    def test_iso8601_unix_roundtrip(self):
        # Test round trip conversion
        original_time = time_s()
        iso_time = unix_to_iso8601(original_time)
        back_to_unix = iso8601_to_unix(iso_time)
        self.assertAlmostEqual(original_time, back_to_unix, delta=1)


if __name__ == "__main__":
    unittest.main()
