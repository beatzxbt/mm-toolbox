import unittest
import numpy as np

from mm_toolbox.ringbuffer import RingBufferSingleDimFloat
from mm_toolbox.moving_average import ExponentialMovingAverage as EMA


class TestEMA(unittest.TestCase):
    def setUp(self):
        self.window = 5
        self.alpha = 0.5
        self.data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.ema = EMA(self.window, self.alpha, fast=False)
        self.fast_ema = EMA(self.window, self.alpha, fast=True)

    def test_class_initialization(self):
        self.assertEqual(self.ema.window, self.window)
        self.assertEqual(self.ema.alpha, self.alpha)
        self.assertFalse(self.ema.fast)
        self.assertEqual(self.ema.value, 0.0)

        self.assertIsInstance(self.ema._values, RingBufferSingleDimFloat)

    def test_class_initialization_fast(self):
        self.assertEqual(self.fast_ema.window, self.window)
        self.assertEqual(self.fast_ema.alpha, self.alpha)
        self.assertTrue(self.fast_ema.fast)
        self.assertEqual(self.fast_ema.value, 0.0)

        self.assertIsInstance(self.fast_ema._values, RingBufferSingleDimFloat)

    def test_initialize(self):
        self.ema.initialize(self.data)

        expected_values = np.array([1.0, 1.5, 2.25, 3.125, 4.0625])
        self.assertEqual(len(self.ema), len(expected_values))
        self.assertAlmostEqual(self.ema.value, expected_values[-1])
        np.testing.assert_array_almost_equal(self.ema.values, expected_values)

    def test_initialize_fast(self):
        self.fast_ema.initialize(self.data)

        self.assertEqual(len(self.ema), 0)
        self.assertAlmostEqual(self.fast_ema.value, 4.0625)

    def test_update(self):
        self.ema.initialize(self.data)
        self.ema.update(6.0)

        expected_values = np.array([1.5, 2.25, 3.125, 4.0625, 5.03125])

        self.assertEqual(len(self.ema), len(expected_values))
        self.assertAlmostEqual(self.ema.value, expected_values[-1])
        np.testing.assert_array_almost_equal(self.ema.values, expected_values)

    def test_update_fast(self):
        self.fast_ema.initialize(self.data)
        self.fast_ema.update(6.0)

        self.assertEqual(len(self.fast_ema), 0)
        self.assertAlmostEqual(self.fast_ema.value, 5.03125)


if __name__ == "__main__":
    unittest.main()
