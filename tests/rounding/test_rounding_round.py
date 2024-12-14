import unittest
import numpy as np
from mm_toolbox.rounding import Round


class TestRound(unittest.TestCase):
    def setUp(self):
        self.rounder = Round(0.01, 0.001)

    def test_bid(self):
        self.assertAlmostEqual(self.rounder.bid(1.234), 1.23)
        self.assertAlmostEqual(self.rounder.bid(1.231), 1.23)

    def test_ask(self):
        self.assertAlmostEqual(self.rounder.ask(1.234), 1.24)
        self.assertAlmostEqual(self.rounder.ask(1.231), 1.24)

    def test_size(self):
        self.assertAlmostEqual(self.rounder.size(1.234), 1.234)
        self.assertAlmostEqual(self.rounder.size(1.230), 1.230)

    def test_bids(self):
        prices = np.array([1.234, 1.231, 1.237])
        expected = np.array([1.23, 1.23, 1.23])
        np.testing.assert_almost_equal(self.rounder.bids(prices), expected, decimal=2)

    def test_asks(self):
        prices = np.array([1.234, 1.231, 1.237])
        expected = np.array([1.24, 1.24, 1.24])
        np.testing.assert_almost_equal(self.rounder.asks(prices), expected, decimal=2)

    def test_sizes(self):
        sizes = np.array([1.234, 1.231, 1.237])
        expected = np.array([1.234, 1.231, 1.237])
        np.testing.assert_almost_equal(self.rounder.sizes(sizes), expected, decimal=3)

    def test_invalid_tick_size(self):
        with self.assertRaises(ValueError):
            Round(0.0, 0.001)

    def test_invalid_lot_size(self):
        with self.assertRaises(ValueError):
            Round(0.01, 0.0)


if __name__ == "__main__":
    unittest.main()
