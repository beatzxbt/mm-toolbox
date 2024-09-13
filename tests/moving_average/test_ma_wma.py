import unittest
import numpy as np

from mm_toolbox.ringbuffer import RingBufferSingleDimFloat
from mm_toolbox.moving_average import WeightedMovingAverage as WMA


class TestWMA(unittest.TestCase):
    def setUp(self):
        self.window = 5
        self.data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.wma = WMA(self.window, fast=False)
        self.fast_wma = WMA(self.window, fast=True)

    def test_class_initialization(self):
        self.assertEqual(self.wma.window, self.window)
        self.assertFalse(self.wma.fast)
        self.assertEqual(self.wma.value, 0.0)

        self.assertIsInstance(self.wma._values, RingBufferSingleDimFloat)
        self.assertIsInstance(self.wma._input_values, RingBufferSingleDimFloat)
        self.assertIsInstance(self.wma._weights, np.ndarray)
        self.assertIsInstance(self.wma._weight_sum, float)

    def test_class_initialization_fast(self):
        self.assertEqual(self.fast_wma.window, self.window)
        self.assertTrue(self.fast_wma.fast)
        self.assertEqual(self.fast_wma.value, 0.0)
        
        self.assertIsInstance(self.fast_wma._values, RingBufferSingleDimFloat)
        self.assertIsInstance(self.fast_wma._input_values, RingBufferSingleDimFloat)
        self.assertIsInstance(self.fast_wma._weights, np.ndarray)
        self.assertIsInstance(self.fast_wma._weight_sum, float)

    def test_initialize(self):
        self.wma.initialize(self.data)

        expected_values = np.array([0.33333333, 0.93333333, 1.73333333, 2.66666667, 3.66666667])
        self.assertEqual(len(self.wma.values), len(expected_values))
        self.assertAlmostEqual(self.wma.value, expected_values[-1])
        np.testing.assert_array_almost_equal(self.wma.values, expected_values)

    def test_initialize_fast(self):
        self.fast_wma.initialize(self.data)

        self.assertAlmostEqual(self.fast_wma.value, 3.66666667)
        self.assertEqual(len(self.fast_wma.values), 0)

    def test_update(self):
        self.wma.initialize(self.data)
        self.wma.update(6.0)

        expected_values = np.array([0.93333333, 1.73333333, 2.66666667, 3.66666667, 4.66666667])

        self.assertEqual(len(self.wma.values), len(expected_values))
        self.assertAlmostEqual(self.wma.value, expected_values[-1])
        np.testing.assert_array_almost_equal(self.wma.values, expected_values)

    def test_update_fast(self):
        self.wma.initialize(self.data)
        self.wma.update(6.0)

        self.assertEqual(len(self.fast_wma.values), 0)
        self.assertAlmostEqual(self.wma.value, 4.66666667)


if __name__ == "__main__":
    unittest.main()
