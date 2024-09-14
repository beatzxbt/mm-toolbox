import unittest
import numpy as np
from mm_toolbox.weights import geometric_weights


class TestGeometricWeights(unittest.TestCase):
    def test_default_ratio(self):
        result = geometric_weights(5)
        expected = np.array([0.32778489, 0.24583867, 0.184379, 0.13828425, 0.10371319])
        np.testing.assert_almost_equal(result, expected, decimal=8)

    def test_custom_ratio(self):
        result = geometric_weights(5, r=0.5)
        expected = np.array(
            [0.51612903, 0.25806452, 0.12903226, 0.06451613, 0.03225806]
        )
        np.testing.assert_almost_equal(result, expected, decimal=8)

    def test_invalid_num(self):
        with self.assertRaises(AssertionError):
            geometric_weights(1)

    def test_sum_of_weights(self):
        result = geometric_weights(10, r=0.9)
        self.assertAlmostEqual(result.sum(), 1.0)

    def test_large_num(self):
        result = geometric_weights(100, r=0.95)
        self.assertEqual(len(result), 100)
        self.assertAlmostEqual(result.sum(), 1.0)


if __name__ == "__main__":
    unittest.main()
