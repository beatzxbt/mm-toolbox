import unittest
import numpy as np

from mm_toolbox.moving_average import WeightedMovingAverage as WMA


class TestWMA(unittest.TestCase):
    def setUp(self):
        self.window = 5
        self.data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.wma = WMA(self.window, fast=False)
        self.fast_wma = WMA(self.window, fast=True)

    def test_initialize(self):
        result = self.wma.initialize(self.data)

        expected_value = 3.6666666666666665  # (1*1 + 2*2 + 3*3 + 4*4 + 5*5) / (1+2+3+4+5)
        self.assertAlmostEqual(result, expected_value)
        self.assertAlmostEqual(self.wma.get_value(), expected_value)
        self.assertEqual(len(self.wma), 1)

    def test_initialize_fast(self):
        result = self.fast_wma.initialize(self.data)

        expected_value = 3.6666666666666665
        self.assertAlmostEqual(result, expected_value)
        self.assertAlmostEqual(self.fast_wma.get_value(), expected_value)
        
        # Should raise error when trying to access values in fast mode
        with self.assertRaises(ValueError):
            len(self.fast_wma)

    def test_next(self):
        self.wma.initialize(self.data)
        next_value = self.wma.next(6.0)

        expected_value = 4.6666666666666665  # (2*1 + 3*2 + 4*3 + 5*4 + 6*5) / (1+2+3+4+5)
        self.assertAlmostEqual(next_value, expected_value)
        # next() doesn't update internal state
        self.assertAlmostEqual(self.wma.get_value(), 3.6666666666666665)

    def test_update(self):
        self.wma.initialize(self.data)
        update_value = self.wma.update(6.0)

        expected_value = 4.6666666666666665
        self.assertAlmostEqual(update_value, expected_value)
        self.assertAlmostEqual(self.wma.get_value(), expected_value)
        
        # Check that values are stored in the buffer
        values = self.wma.get_values()
        self.assertEqual(len(values), 2)
        self.assertAlmostEqual(values[-1], expected_value)

    def test_update_fast(self):
        self.fast_wma.initialize(self.data)
        update_value = self.fast_wma.update(6.0)

        expected_value = 4.6666666666666665
        self.assertAlmostEqual(update_value, expected_value)
        self.assertAlmostEqual(self.fast_wma.get_value(), expected_value)
        
        # Should raise error when trying to access values in fast mode
        with self.assertRaises(ValueError):
            self.fast_wma.get_values()

    def test_invalid_window(self):
        with self.assertRaises(ValueError):
            WMA(1)


if __name__ == "__main__":
    unittest.main()
