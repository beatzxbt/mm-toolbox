import unittest
import numpy as np

from mm_toolbox.moving_average import ExponentialMovingAverage as EMA


class TestEMA(unittest.TestCase):
    def setUp(self):
        self.window = 5
        self.alpha = 0.5
        self.data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.ema = EMA(self.window, is_fast=False, alpha=self.alpha)
        self.fast_ema = EMA(self.window, is_fast=True, alpha=self.alpha)

    def test_initialize(self):
        result = self.ema.initialize(self.data)

        expected_value = 4.0625
        self.assertAlmostEqual(result, expected_value)
        self.assertAlmostEqual(self.ema.get_value(), expected_value)
        self.assertEqual(len(self.ema), self.window)

    def test_initialize_fast(self):
        result = self.fast_ema.initialize(self.data)

        expected_value = 4.0625
        self.assertAlmostEqual(result, expected_value)
        self.assertAlmostEqual(self.fast_ema.get_value(), expected_value)
        
        # Should raise error when trying to access values in fast mode
        with self.assertRaises(ValueError):
            len(self.fast_ema)

    def test_next(self):
        self.ema.initialize(self.data)
        next_value = self.ema.next(6.0)

        expected_value = 5.03125
        self.assertAlmostEqual(next_value, expected_value)
        # next() doesn't update internal state
        self.assertAlmostEqual(self.ema.get_value(), 4.0625)

    def test_update(self):
        self.ema.initialize(self.data)
        update_value = self.ema.update(6.0)

        expected_value = 5.03125
        self.assertAlmostEqual(update_value, expected_value)
        self.assertAlmostEqual(self.ema.get_value(), expected_value)
        
        # Check that values are stored in the buffer
        values = self.ema.get_values()
        self.assertEqual(len(values), self.window)
        self.assertAlmostEqual(values[-1], expected_value)

    def test_update_fast(self):
        self.fast_ema.initialize(self.data)
        update_value = self.fast_ema.update(6.0)

        expected_value = 5.03125
        self.assertAlmostEqual(update_value, expected_value)
        self.assertAlmostEqual(self.fast_ema.get_value(), expected_value)
        
        # Should raise error when trying to access values in fast mode
        with self.assertRaises(ValueError):
            self.fast_ema.get_values()

    def test_default_alpha(self):
        # Test that default alpha is 3/(window+1)
        default_ema = EMA(self.window)
        default_ema.initialize(self.data)
        
        # Calculate expected values with default alpha
        expected_alpha = 3.0 / float(self.window + 1)
        expected_value = self.data[0]
        for i in range(1, len(self.data)):
            expected_value = expected_alpha * self.data[i] + (1.0 - expected_alpha) * expected_value
            
        self.assertAlmostEqual(default_ema.get_value(), expected_value)

    def test_invalid_window(self):
        # Test that window <= 1 raises ValueError
        with self.assertRaises(ValueError):
            EMA(1)
        
        with self.assertRaises(ValueError):
            EMA(0)
            
    def test_warm_requirement(self):
        # Test that methods raise error if not initialized
        uninitialized_ema = EMA(self.window)
        
        with self.assertRaises(ValueError):
            uninitialized_ema.next(1.0)
            
        with self.assertRaises(ValueError):
            uninitialized_ema.update(1.0)


if __name__ == "__main__":
    unittest.main()
