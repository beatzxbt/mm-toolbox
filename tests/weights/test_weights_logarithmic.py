import unittest
import numpy as np
from mm_toolbox.weights import logarithmic_weights


class TestLogarithmicWeights(unittest.TestCase):
    def test_default(self):
        result = logarithmic_weights(5)
        expected = np.array([0.0, 0.14478295, 0.22947555, 0.2895659, 0.3361756])
        np.testing.assert_almost_equal(result, expected, decimal=8)

    def test_larger_num(self):
        result = logarithmic_weights(100)
        self.assertEqual(len(result), 100)
        self.assertAlmostEqual(result.sum(), 1.0)

    def test_invalid_num(self):
        with self.assertRaises(ValueError):
            logarithmic_weights(1)


if __name__ == "__main__":
    unittest.main()
