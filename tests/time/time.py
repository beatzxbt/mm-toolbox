import unittest
import time
import re
from mm_toolbox.src.time import (
    time_iso8601,
    iso8601_to_unix,
) 

class TestTimeFunctions(unittest.TestCase):
    def test_time_iso8601(self):
        result = time_iso8601()
        expected_pattern = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z"
        self.assertRegex(result, expected_pattern)

        # Verify the conversion back to Unix timestamp is accurate
        unix_time_from_iso8601 = iso8601_to_unix(result)
        self.assertAlmostEqual(unix_time_from_iso8601, time.time(), delta=2)

    def test_iso8601_to_unix(self):
        # Test a known ISO 8601 timestamp
        iso8601_timestamp = "2023-04-04T00:28:50.516Z"
        expected_unix = 1680569330
        self.assertEqual(iso8601_to_unix(iso8601_timestamp), expected_unix)

        # Test another timestamp with fractional seconds
        iso8601_timestamp = "2024-08-18T13:22:52.488Z"
        expected_unix = 1723882972
        self.assertEqual(iso8601_to_unix(iso8601_timestamp), expected_unix)

    def test_iso8601_to_unix_invalid(self):
        # Test an invalid ISO 8601 timestamp
        with self.assertRaises(ValueError):
            iso8601_to_unix("Invalid timestamp")

if __name__ == "__main__":
    unittest.main()
