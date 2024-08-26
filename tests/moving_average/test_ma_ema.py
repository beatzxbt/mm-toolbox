import unittest
import numpy as np

from src.mm_toolbox.ringbuffer import RingBufferSingleDimFloat
from src.mm_toolbox.moving_average import ExponentialMovingAverage as EMA


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
        self.assertTrue(self.fast_ema.fast)
        self.assertEqual(self.ema.value, 0.0)
        self.assertIsInstance(self.ema.ringbuffer, RingBufferSingleDimFloat)

    def test_class_initialization_fast(self):
        self.assertEqual(self.fast_ema.window, self.window)
        self.assertEqual(self.fast_ema.alpha, self.alpha)
        self.assertTrue(self.fast_ema.fast)
        self.assertEqual(self.fast_ema.value, 0.0)
        self.assertIsInstance(self.fast_ema.ringbuffer, RingBufferSingleDimFloat)

    def test_recursive_ema(self):
        initial_value = 1.0
        update_value = 2.0
        self.ema.value = initial_value
        result = self.ema._recursive_ema_(update_value)
        expected = self.alpha * update_value + (1.0 - self.alpha) * initial_value
        self.assertAlmostEqual(result, expected)

    def test_recursive_ema_fast(self):
        initial_value = 1.0
        update_value = 2.0
        self.fast_ema.value = initial_value
        result = self.fast_ema._recursive_ema_(update_value)
        expected = self.alpha * update_value + (1.0 - self.alpha) * initial_value
        self.assertAlmostEqual(result, expected)

    def test_initialize(self):
        self.ema.initialize(self.data)
        self.assertAlmostEqual(self.ema.value, 4.0625)
        self.assertEqual(len(self.ema.ringbuffer), self.window)
        self.assertEqual(self.ema.ringbuffer[-1], self.ema.value)
        self.assertEqual(self.ema.ringbuffer[0], self.data[0])

    def test_initialize_fast(self):
        self.fast_ema.initialize(self.data)
        self.assertAlmostEqual(self.fast_ema.value, 4.0625)
        self.assertEqual(len(self.ema.ringbuffer), 0)

    def test_update(self):
        self.ema.initialize(self.data)
        new_val = 6.0
        old_value = self.ema.value
        self.ema.update(new_val)
        expected = self.alpha * new_val + (1.0 - self.alpha) * old_value
        self.assertAlmostEqual(self.ema.value, expected)
        self.assertEqual(len(self.ema.ringbuffer), self.window)
        self.assertAlmostEqual(self.ema.ringbuffer[-1], expected)

    def test_update_fast(self):
        self.fast_ema.initialize(self.data)
        new_val = 6.0
        old_value = self.fast_ema.value
        self.fast_ema.update(new_val)
        expected = self.alpha * new_val + (1.0 - self.alpha) * old_value
        self.assertAlmostEqual(self.fast_ema.value, expected)
        self.assertEqual(len(self.fast_ema.ringbuffer), 0)

    # Dunders, as well as .as_array() are not tested. The underlying RingBuffer
    # already extensively tests these functionalities and EMA only acts as a buffer
    # and funnels arguments to the RingBuffer methods directly.


if __name__ == "__main__":
    unittest.main()
