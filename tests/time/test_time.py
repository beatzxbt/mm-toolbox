import unittest
from mm_toolbox.src.time import (
    time_s,
    time_iso8601,
    iso8601_to_unix,
) 

class TestTimeFunctions(unittest.TestCase):
    def test_time_iso8601(self):
        time_s_now = time_s()
        result = time_iso8601()
        expected_pattern = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z"
        self.assertRegex(result, expected_pattern)

        # Verify the conversion back to Unix timestamp is accurate
        unix_time_from_iso8601 = iso8601_to_unix(result)
        self.assertAlmostEqual(unix_time_from_iso8601, time_s_now, delta=2)

if __name__ == "__main__":
    unittest.main()
