import unittest
import numpy as np
from mm_toolbox.src.weights import geometric_weights  

class TestGeometricWeights(unittest.TestCase):
    def test_default_ratio(self):
        result = geometric_weights(5)
        expected = np.array([0.2962963, 0.22222222, 0.16666667, 0.125, 0.09375])
        np.testing.assert_almost_equal(result, expected, decimal=8)

    def test_custom_ratio(self):
        result = geometric_weights(5, r=0.5)
        expected = np.array([0.5, 0.25, 0.125, 0.0625, 0.03125])
        np.testing.assert_almost_equal(result, expected, decimal=8)

    def test_invalid_num(self):
        with self.assertRaises(AssertionError):
            geometric_weights(1)

    def test_invalid_ratio(self):
        with self.assertRaises(AssertionError):
            geometric_weights(5, r=-0.1)
        with self.assertRaises(AssertionError):
            geometric_weights(5, r=1.5)

    def test_sum_of_weights(self):
        result = geometric_weights(10, r=0.9)
        self.assertAlmostEqual(result.sum(), 1.0)

    def test_large_num(self):
        result = geometric_weights(100, r=0.95)
        self.assertEqual(len(result), 100)
        self.assertAlmostEqual(result.sum(), 1.0)

if __name__ == '__main__':
    unittest.main()
