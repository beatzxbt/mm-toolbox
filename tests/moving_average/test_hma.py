import unittest
import numpy as np

from mm_toolbox.src.moving_average.hma import HullMovingAverage as HMA

class TestHMA(unittest.TestCase):
    def setUp(self):
        self.window = 10
        self.data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        self.hma = HMA(self.window, fast=False)
        self.fast_hma = HMA(self.window, fast=True)

    def test_class_initialization(self):
        self.assertEqual(self.hma.window, self.window)
        self.assertFalse(self.hma.fast)
        self.assertTrue(self.fast_hma.fast)
        self.assertEqual(self.hma.value, 0.0)
        self.assertEqual(self.hma.short_ema.window, self.window // 2)
        self.assertEqual(self.hma.long_ema.window, self.window)
        self.assertEqual(self.hma.smooth_ema.window, int(self.window ** 0.5))
        self.assertEqual(len(self.hma.ringbuffer), 0)

    def test_class_initialization_fast(self):
        self.assertEqual(self.fast_hma.window, self.window)
        self.assertTrue(self.fast_hma.fast)
        self.assertEqual(self.fast_hma.value, 0.0)
        self.assertEqual(self.fast_hma.short_ema.window, self.window // 2)
        self.assertEqual(self.fast_hma.long_ema.window, self.window)
        self.assertEqual(self.fast_hma.smooth_ema.window, int(self.window ** 0.5))
        self.assertEqual(len(self.fast_hma.ringbuffer), 0)

    def test_recursive_hma(self):
        initial_value = 5.0
        self.hma.short_ema.value = 3.0
        self.hma.long_ema.value = 2.0

        result = self.hma._recursive_hma_(initial_value)
        expected_value = 3.0 * 2.0 - 2.0
        self.assertAlmostEqual(result, expected_value)
        self.assertEqual(self.hma.smooth_ema.value, expected_value)
        
    def test_initialize(self):
        self.hma.initialize(self.data)
        expected_value = self.hma.value
        self.assertEqual(len(self.hma.ringbuffer), self.window)
        self.assertAlmostEqual(self.hma.value, expected_value)

    def test_initialize_fast(self):
        self.fast_hma.initialize(self.data)
        expected_value = self.fast_hma.value
        self.assertEqual(len(self.fast_hma.ringbuffer), 0)
        self.assertAlmostEqual(self.fast_hma.value, expected_value)

    def test_update(self):
        self.hma.initialize(self.data)
        new_val = 11.0
        old_value = self.hma.value
        self.hma.update(new_val)
        self.assertNotEqual(self.hma.value, old_value)
        self.assertEqual(len(self.hma.ringbuffer), self.window)
        self.assertAlmostEqual(self.hma.ringbuffer[-1], self.hma.value)

    def test_update_fast(self):
        self.fast_hma.initialize(self.data)
        new_val = 11.0
        old_value = self.fast_hma.value
        self.fast_hma.update(new_val)
        self.assertNotEqual(self.fast_hma.value, old_value)
        self.assertEqual(len(self.fast_hma.ringbuffer), 0)

    # Dunders, as well as .as_array() are not tested. The underlying RingBuffer
    # already extensively tests these functionalities and EMA only acts as a buffer 
    # and funnels arguments to the RingBuffer methods directly.
    
if __name__ == '__main__':
    unittest.main()
