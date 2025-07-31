import unittest
import numpy as np

from mm_toolbox.moving_average import SimpleMovingAverage as SMA


class TestSMA(unittest.TestCase):
    def setUp(self):
        self.window = 5
        self.data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.sma = SMA(self.window, fast=False)
        self.fast_sma = SMA(self.window, fast=True)

    def test_initialize(self):
        result = self.sma.initialize(self.data)

        expected_value = 3.0  # (1 + 2 + 3 + 4 + 5) / 5
        self.assertAlmostEqual(result, expected_value)
        self.assertAlmostEqual(self.sma.get_value(), expected_value)
        self.assertEqual(len(self.sma), 1)

    def test_initialize_fast(self):
        result = self.fast_sma.initialize(self.data)

        expected_value = 3.0
        self.assertAlmostEqual(result, expected_value)
        self.assertAlmostEqual(self.fast_sma.get_value(), expected_value)

        # Should raise error when trying to access values in fast mode
        with self.assertRaises(ValueError):
            len(self.fast_sma)

    def test_next(self):
        self.sma.initialize(self.data)
        next_value = self.sma.next(6.0)

        expected_value = 4.0  # (2 + 3 + 4 + 5 + 6) / 5
        self.assertAlmostEqual(next_value, expected_value)
        # next() doesn't update internal state
        self.assertAlmostEqual(self.sma.get_value(), 3.0)

    def test_update(self):
        self.sma.initialize(self.data)
        update_value = self.sma.update(6.0)

        expected_value = 4.0  # (2 + 3 + 4 + 5 + 6) / 5
        self.assertAlmostEqual(update_value, expected_value)
        self.assertAlmostEqual(self.sma.get_value(), expected_value)

        # Check that values are stored in the buffer
        values = self.sma.get_values()
        self.assertEqual(len(values), 2)
        self.assertAlmostEqual(values[-1], expected_value)

    def test_update_fast(self):
        self.fast_sma.initialize(self.data)
        update_value = self.fast_sma.update(6.0)

        expected_value = 4.0
        self.assertAlmostEqual(update_value, expected_value)
        self.assertAlmostEqual(self.fast_sma.get_value(), expected_value)

        # Should raise error when trying to access values in fast mode
        with self.assertRaises(ValueError):
            self.fast_sma.get_values()

    def test_invalid_window(self):
        with self.assertRaises(ValueError):
            SMA(1)

    def test_initialize_insufficient_data(self):
        with self.assertRaises(ValueError):
            self.sma.initialize(np.array([1.0, 2.0]))

    def test_uninitialized_access(self):
        with self.assertRaises(ValueError):
            self.sma.next(6.0)

        with self.assertRaises(ValueError):
            self.sma.update(6.0)


if __name__ == "__main__":
    unittest.main()
