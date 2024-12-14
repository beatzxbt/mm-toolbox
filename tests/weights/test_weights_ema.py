import unittest
import numpy as np
from mm_toolbox.weights import ema_weights


class TestEmaWeights(unittest.TestCase):
    def test_default_alpha(self):
        window = 5
        result = ema_weights(window)
        expected = ema_weights(window, alpha=3.0 / float(window + 1))
        np.testing.assert_almost_equal(result, expected, decimal=8)

    def test_custom_alpha(self):
        result = ema_weights(5, alpha=0.5)
        expected = np.array([0.03125, 0.0625, 0.125, 0.25, 0.5])
        np.testing.assert_almost_equal(result, expected, decimal=8)

    def test_larger_window_size(self):
        result = ema_weights(10, alpha=0.5)
        expected = np.array(
            [
                0.00097656,
                0.00195312,
                0.00390625,
                0.0078125,
                0.015625,
                0.03125,
                0.0625,
                0.125,
                0.25,
                0.5,
            ]
        )
        np.testing.assert_almost_equal(result, expected, decimal=8)

    def test_invalid_window(self):
        with self.assertRaises(ValueError):
            ema_weights(1)


if __name__ == "__main__":
    unittest.main()
