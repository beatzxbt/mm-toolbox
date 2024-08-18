import unittest
import numpy as np
from mm_toolbox.src.weights import ema_weights

class TestEmaWeights(unittest.TestCase):
    def test_default_alpha(self):
        result = ema_weights(5)
        expected = np.array([0.33333333, 0.22222222, 0.14814815, 0.09876543, 0.06584362])
        np.testing.assert_almost_equal(result, expected, decimal=8)

    def test_custom_alpha(self):
        result = ema_weights(5, alpha=0.5)
        expected = np.array([0.5, 0.25, 0.125, 0.0625, 0.03125])
        np.testing.assert_almost_equal(result, expected, decimal=8)

    def test_larger_window_size(self):
        result = ema_weights(10)
        expected = np.array([0.27272727, 0.18181818, 0.12121212, 0.08080808, 0.05387205,
                             0.0359147, 0.02394313, 0.01596209, 0.01064139, 0.00709426])
        np.testing.assert_almost_equal(result, expected, decimal=8)

    def test_invalid_window(self):
        with self.assertRaises(AssertionError):
            ema_weights(1)

    def test_negative_alpha(self):
        with self.assertRaises(AssertionError):
            ema_weights(5, alpha=-0.1)

if __name__ == '__main__':
    unittest.main()
