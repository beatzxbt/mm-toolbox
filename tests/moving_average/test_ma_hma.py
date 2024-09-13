import unittest
import numpy as np

from mm_toolbox.ringbuffer import RingBufferSingleDimFloat
from mm_toolbox.moving_average.hma import HullMovingAverage as HMA
from mm_toolbox.moving_average.wma import WeightedMovingAverage as WMA

class TestHMA(unittest.TestCase):
    def setUp(self):
        self.window = 5
        self.data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.hma = HMA(self.window, fast=False)
        self.fast_hma = HMA(self.window, fast=True)

    def test_class_initialization(self):
        self.assertEqual(self.hma.window, self.window)
        self.assertFalse(self.hma.fast)
        self.assertEqual(self.hma.value, 0.0)

        self.assertIsInstance(self.hma._values, RingBufferSingleDimFloat)
        self.assertIsInstance(self.hma._short_wma, WMA)
        self.assertIsInstance(self.hma._long_wma, WMA)
        self.assertIsInstance(self.hma._smooth_wma, WMA)

        self.assertEqual(self.hma._short_wma.window, int(self.window / 2))
        self.assertEqual(self.hma._long_wma.window, self.window)
        self.assertEqual(self.hma._smooth_wma.window, int(np.sqrt(self.window)))

    def test_class_initialization_fast(self):
        self.assertEqual(self.fast_hma.window, self.window)
        self.assertTrue(self.fast_hma.fast)
        self.assertEqual(self.fast_hma.value, 0.0)

        self.assertIsInstance(self.fast_hma._values, RingBufferSingleDimFloat)
        self.assertIsInstance(self.fast_hma._short_wma, WMA)
        self.assertIsInstance(self.fast_hma._long_wma, WMA)
        self.assertIsInstance(self.fast_hma._smooth_wma, WMA)

        self.assertEqual(self.fast_hma._short_wma.window, int(self.window / 2))
        self.assertEqual(self.fast_hma._long_wma.window, self.window)
        self.assertEqual(self.fast_hma._smooth_wma.window, int(np.sqrt(self.window)))

    def test_initialize(self):
        self.hma.initialize(self.data)

        expected_values = np.array([7.93333333])
        self.assertEqual(len(self.hma.values), len(expected_values))
        self.assertAlmostEqual(self.hma.value, expected_values[-1])
        np.testing.assert_array_almost_equal(self.hma.values, expected_values)

    def test_initialize_fast(self):
        self.fast_hma.initialize(self.data)

        self.assertAlmostEqual(self.fast_hma.value, 7.93333333)
        self.assertEqual(len(self.fast_hma.values), 0)

    def test_update(self):
        self.hma.initialize(self.data)
        self.hma.update(6.0)

        expected_values = np.array([7.93333333, 7.24444444])

        self.assertEqual(len(self.hma.values), len(expected_values))
        self.assertAlmostEqual(self.hma.value, expected_values[-1])
        np.testing.assert_array_almost_equal(self.hma.values, expected_values)

    def test_update_fast(self):
        self.fast_hma.initialize(self.data)
        self.fast_hma.update(6.0)

        expected_values = np.array([7.93333333, 7.24444444])

        self.assertEqual(len(self.fast_hma.values), 0)
        self.assertAlmostEqual(self.fast_hma.value, expected_values[-1])

if __name__ == "__main__":
    unittest.main()
