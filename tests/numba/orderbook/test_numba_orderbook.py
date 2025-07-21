import unittest
import numpy as np

from mm_toolbox.numba.orderbook import Orderbook


class TestOrderbook(unittest.TestCase):
    def setUp(self):
        self.asks = np.array([[60000 + i, (1 + i)] for i in range(5)], dtype=np.float64)
        self.bids = np.array([[59999 - i, (1 + i)] for i in range(5)], dtype=np.float64)
        self.seq_id = 1
        self.size = 5
        self.orderbook = Orderbook(self.size)

        # Uncommon to use methods within the setup, but makes the rest of the tests
        # a lot cleaner to do. If refresh fails, the 4 next tasks will be expected
        # to fail too (as well as this setup).
        self.orderbook.refresh(self.asks, self.bids, self.seq_id)

    def test_class_initialization(self):
        empty_book = Orderbook(self.size)
        self.assertEqual(empty_book.size, self.size)
        self.assertTrue((empty_book.asks == 0.0).all())
        self.assertTrue((empty_book.bids == 0.0).all())

    def test_full_refresh(self):
        empty_book = Orderbook(self.size)
        empty_book.refresh(self.asks, self.bids, self.seq_id)
        self.assertEqual(empty_book.asks[0, 0], 60000.0)
        self.assertEqual(empty_book.asks[0, 1], 1.0)
        self.assertEqual(empty_book.bids[0, 0], 59999.0)
        self.assertEqual(empty_book.bids[0, 1], 1.0)

    def test_small_refresh(self):
        empty_book = Orderbook(self.size)
        with self.assertRaises(AssertionError):
            empty_book.refresh(self.asks[:2], self.bids[:2], self.seq_id)

    def test_empty_refresh(self):
        empty_book = Orderbook(self.size)
        with self.assertRaises(AssertionError):
            empty_book.refresh(
                np.array([[]], dtype=np.float64),
                np.array([[]], dtype=np.float64),
                self.seq_id,
            )

    def test_update_bbo_existing_bid_ask(self):
        best_bid = self.bids[0].copy()
        best_bid[1] = 10.0  # Change size only

        best_ask = self.asks[0].copy()
        best_ask[1] = 8.0  # Change size only

        self.orderbook.update_bbo(
            best_bid[0],
            best_bid[1],
            best_ask[0],
            best_ask[1],
            self.orderbook.seq_id + 1,
        )

        self.assertEqual(self.orderbook.bids[0, 0], 59999.0)
        self.assertEqual(self.orderbook.bids[0, 1], 10.0)
        self.assertEqual(self.orderbook.asks[0, 0], 60000.0)
        self.assertEqual(self.orderbook.asks[0, 1], 8.0)

    def test_update_bbo_new_ask(self):
        best_bid = self.bids[0].copy()

        best_ask = self.bids[0].copy() - 1.5
        best_ask[1] = 8.0

        self.orderbook.update_bbo(
            best_bid[0],
            best_bid[1],
            best_ask[0],
            best_ask[1],
            self.orderbook.seq_id + 1,
        )

        self.assertEqual(self.orderbook.bids[0, 0], 59997.0)
        self.assertEqual(self.orderbook.bids[0, 1], 3.0)
        self.assertEqual(self.orderbook.asks[0, 0], 59997.5)
        self.assertEqual(self.orderbook.asks[0, 1], 8.0)

        np.testing.assert_array_equal(self.orderbook.bids[-1], np.array([0.0, 0.0]))
        np.testing.assert_array_equal(self.orderbook.bids[-2], np.array([0.0, 0.0]))

    def test_update_bbo_new_bid(self):
        best_bid = self.bids[0].copy() + 1.5
        best_bid[1] = 10.0

        best_ask = self.asks[0].copy()

        self.orderbook.update_bbo(
            best_bid[0],
            best_bid[1],
            best_ask[0],
            best_ask[1],
            self.orderbook.seq_id + 1,
        )

        print(self.orderbook.recordable())
        self.assertEqual(self.orderbook.bids[0, 0], 60000.5)
        self.assertEqual(self.orderbook.bids[0, 1], 10.0)
        self.assertEqual(self.orderbook.asks[0, 0], 60001.0)
        self.assertEqual(self.orderbook.asks[0, 1], 1.0)

        np.testing.assert_array_equal(self.orderbook.asks[-1], np.array([0.0, 0.0]))

    def test_update_bids_existing_bba(self):
        best_bid = self.bids[0].copy()
        best_bid[1] = 10.0  # Change size only

        self.orderbook.update_bids(np.array([best_bid]), self.orderbook.seq_id + 1)

        self.assertEqual(self.orderbook.bids[0, 0], 59999.0)
        self.assertEqual(self.orderbook.bids[0, 1], 10.0)
        self.assertEqual(self.orderbook.asks[0, 0], 60000.0)
        self.assertEqual(self.orderbook.asks[0, 1], 1.0)

    def test_update_asks_existing_bba(self):
        best_ask = self.asks[0].copy()
        best_ask[1] = 10.0  # Change size only

        self.orderbook.update_asks(np.array([best_ask]), self.orderbook.seq_id + 1)

        self.assertEqual(self.orderbook.bids[0, 0], 59999.0)
        self.assertEqual(self.orderbook.bids[0, 1], 1.0)
        self.assertEqual(self.orderbook.asks[0, 0], 60000.0)
        self.assertEqual(self.orderbook.asks[0, 1], 10.0)

    def test_update_bids_existing_deep(self):
        fourth_level = self.bids[3].copy()
        fifth_level = self.bids[4].copy()

        fourth_level[1] = 10.0  # Change size only
        fifth_level[1] = 19.3  # Change size only

        self.orderbook.update_bids(
            np.array([fourth_level, fifth_level]), self.orderbook.seq_id + 1
        )

        self.assertEqual(self.orderbook.bids[3, 0], 59996.0)
        self.assertEqual(self.orderbook.bids[3, 1], 10.0)
        self.assertEqual(self.orderbook.bids[4, 0], 59995.0)
        self.assertEqual(self.orderbook.bids[4, 1], 19.3)

    def test_update_asks_existing_deep(self):
        fourth_level = self.asks[3].copy()
        fifth_level = self.asks[4].copy()

        fourth_level[1] = 10.0  # Change size only
        fifth_level[1] = 19.3  # Change size only

        self.orderbook.update_asks(
            np.array([fourth_level, fifth_level]), self.orderbook.seq_id + 1
        )

        self.assertEqual(self.orderbook.asks[3, 0], 60003.0)
        self.assertEqual(self.orderbook.asks[3, 1], 10.0)
        self.assertEqual(self.orderbook.asks[4, 0], 60004.0)
        self.assertEqual(self.orderbook.asks[4, 1], 19.3)

    def test_update_bids_add_new_deep(self):
        inbounds_level = np.array(
            [59998.5, 0.25]
        )  # This should become the new 1st level
        outbounds_level = np.array(
            [59992.5, 0.4]
        )  # This is out of bounds, shouldnt show up

        self.orderbook.update_bids(
            np.array([inbounds_level, outbounds_level]), self.orderbook.seq_id + 1
        )

        self.assertEqual(self.orderbook.bids[1, 0], 59998.5)
        self.assertEqual(self.orderbook.bids[1, 1], 0.25)
        self.assertEqual(self.orderbook.bids[2, 0], 59998.0)  # Unchanged from before
        self.assertEqual(self.orderbook.bids[2, 1], 2.0)  # Unchanged from before
        self.assertEqual(
            self.orderbook.bids[4, 0], 59996.0
        )  # Previously 4th level moves down to 5th
        self.assertEqual(
            self.orderbook.bids[4, 1], 4.0
        )  # Previously 4th level moves down to 5th
        self.assertEqual(len(self.orderbook.bids), self.size)

    def test_update_asks_add_new_deep(self):
        inbounds_level = np.array(
            [60000.5, 0.25]
        )  # This should become the new 1st level
        outbounds_level = np.array(
            [60006.5, 0.4]
        )  # This is out of bounds, shouldnt show up

        self.orderbook.update_asks(
            np.array([inbounds_level, outbounds_level]), self.orderbook.seq_id + 1
        )

        self.assertEqual(self.orderbook.asks[1, 0], 60000.5)
        self.assertEqual(self.orderbook.asks[1, 1], 0.25)
        self.assertEqual(self.orderbook.asks[2, 0], 60001.0)  # Unchanged from before
        self.assertEqual(self.orderbook.asks[2, 1], 2.0)  # Unchanged from before
        self.assertEqual(
            self.orderbook.asks[4, 0], 60003.0
        )  # Previously 4th level moves down to 5th
        self.assertEqual(
            self.orderbook.asks[4, 1], 4.0
        )  # Previously 4th level moves down to 5th
        self.assertEqual(len(self.orderbook.asks), self.size)

    def test_update_bids_add_new_jump(self):
        jump_level = np.array(
            [60000.5, 1.75]
        )  # This is higher than the current best ask

        self.orderbook.update_bids(np.array([jump_level]), self.orderbook.seq_id + 1)

        self.assertEqual(self.orderbook.bids[0, 0], 60000.5)
        self.assertEqual(self.orderbook.bids[0, 1], 1.75)
        self.assertEqual(self.orderbook.bids[1, 0], 59999.0)  # Unchanged from before
        self.assertEqual(self.orderbook.bids[1, 1], 1.0)  # Unchanged from before

        self.assertEqual(
            self.orderbook.asks[0, 0], 60001.0
        )  # 2nd level becomes the 1st
        self.assertEqual(self.orderbook.asks[0, 1], 2.0)  # 2nd level becomes the 1st
        self.assertEqual(self.orderbook.asks[4, 0], 0.0)  # Last level is fresh (new)
        self.assertEqual(self.orderbook.asks[4, 1], 0.0)  # Last level is fresh (new)

    def test_update_asks_add_new_jump(self):
        jump_level = np.array(
            [59998.5, 1.75]
        )  # This is lower than the current best bid

        self.orderbook.update_asks(np.array([jump_level]), self.orderbook.seq_id + 1)

        self.assertEqual(self.orderbook.asks[0, 0], 59998.5)
        self.assertEqual(self.orderbook.asks[0, 1], 1.75)
        self.assertEqual(self.orderbook.asks[1, 0], 60000.0)  # Unchanged from before
        self.assertEqual(self.orderbook.asks[1, 1], 1.0)  # Unchanged from before

        self.assertEqual(
            self.orderbook.bids[0, 0], 59998.0
        )  # 2nd level becomes the 1st
        self.assertEqual(self.orderbook.bids[0, 1], 2.0)  # 2nd level becomes the 1st
        self.assertEqual(self.orderbook.bids[4, 0], 0.0)  # Last level is fresh (new)
        self.assertEqual(self.orderbook.bids[4, 1], 0.0)  # Last level is fresh (new)

    def test_mid_price(self):
        self.assertAlmostEqual(
            self.orderbook.mid_price, (self.asks[0, 0] + self.bids[0, 0]) / 2
        )

    def test_wmid_price(self):
        self.orderbook.refresh(self.asks, self.bids, self.seq_id)
        bid_price, bid_size = self.bids[0]
        ask_price, ask_size = self.asks[0]
        imb = bid_size / (bid_size + ask_size)
        expected_wmid = (bid_price * imb) + (ask_price * (1.0 - imb))
        self.assertAlmostEqual(self.orderbook.wmid_price, expected_wmid)

    def test_get_vamp(self):
        # Equal book, all quantities produce same VAMP
        self.assertAlmostEqual(
            self.orderbook.get_vamp(self.orderbook.mid_price * 1.0), 59999.5
        )
        self.assertAlmostEqual(
            self.orderbook.get_vamp(self.orderbook.mid_price * 2.0), 59999.5
        )
        self.assertAlmostEqual(
            self.orderbook.get_vamp(self.orderbook.mid_price * 3.0), 59999.5
        )

        # Imbalanced book
        imbalanced_asks = self.asks
        imbalanced_asks[:, 1] *= 2.0
        self.orderbook.refresh(imbalanced_asks, self.bids, self.seq_id)
        self.assertAlmostEqual(
            self.orderbook.get_vamp(self.orderbook.mid_price * 2.0), 60000.555555555555
        )

    def test_get_spread(self):
        self.assertAlmostEqual(
            self.orderbook.bid_ask_spread, self.asks[0, 0] - self.bids[0, 0]
        )

    def test_get_slippage(self):
        self.assertAlmostEqual(
            self.orderbook.get_slippage(self.orderbook.bids, 4.0), 1.8333333333333333
        )

    def test_length(self):
        self.assertEqual(len(self.orderbook), self.size)

    def test_equality(self):
        dummy_orderbook = Orderbook(self.size)
        self.assertNotEqual(dummy_orderbook, self.orderbook)
        dummy_orderbook.refresh(self.asks, self.bids, self.seq_id)
        self.assertEqual(dummy_orderbook, self.orderbook)

    def test_str_representation(self):
        orderbook_str = str(self.orderbook)
        self.assertIn("Orderbook", orderbook_str)
        self.assertIn("size=5", orderbook_str)
        self.assertIn("seq_id", orderbook_str)
        self.assertIn("bids", orderbook_str)
        self.assertIn("asks", orderbook_str)

    def test_integrated(self):
        # A full cycle test with realistic orderbook updates, testing all
        # potential scenarios (more prevalent under high volatility).
        orderbook = Orderbook(size=10)
        start_asks = np.array([[60000.0 + i, 1.0 + i] for i in range(10)])
        start_bids = np.array([[59999.0 - i, 1.0 + i] for i in range(10)])
        orderbook.refresh(start_asks, start_bids, 1)

        # Make sure arrays are initialized properly
        np.testing.assert_array_equal(orderbook.asks, start_asks)
        np.testing.assert_array_equal(orderbook.bids, start_bids)

        # 3 BBA updates roll in for both sides
        best_ask_u1 = np.array([[60000.0, 1.2]])
        best_bid_u1 = np.array([[59999.0, 0.6]])
        orderbook.update_full(best_ask_u1, best_bid_u1, 2)

        self.assertEqual(orderbook.asks[0, 0], 60000.0)
        self.assertEqual(orderbook.asks[0, 1], 1.2)
        self.assertEqual(orderbook.bids[0, 0], 59999.0)
        self.assertEqual(orderbook.bids[0, 1], 0.6)

        # Update 2
        best_ask_u2 = np.array([[60000.0, 1.8]])
        best_bid_u2 = np.array([[59999.0, 0.3]])
        orderbook.update_full(best_ask_u2, best_bid_u2, 3)

        self.assertEqual(orderbook.asks[0, 0], 60000.0)
        self.assertEqual(orderbook.asks[0, 1], 1.8)
        self.assertEqual(orderbook.bids[0, 0], 59999.0)
        self.assertEqual(orderbook.bids[0, 1], 0.3)

        # Update 3
        best_ask_u3 = np.array([[60000.0, 2.9]])
        best_bid_u3 = np.array([[59999.0, 0.2]])
        orderbook.update_full(best_ask_u3, best_bid_u3, 4)

        self.assertEqual(orderbook.asks[0, 0], 60000.0)
        self.assertEqual(orderbook.asks[0, 1], 2.9)
        self.assertEqual(orderbook.bids[0, 0], 59999.0)
        self.assertEqual(orderbook.bids[0, 1], 0.2)

        # Some orderbook feeds do not give 0 sizes on BBA changes
        best_ask_change = np.array([[60001.0, 0.4]])
        best_bid_change = np.array([[60000.0, 5.5]])
        orderbook.update_full(best_ask_change, best_bid_change, 6)

        self.assertEqual(orderbook.asks[0, 0], 60001.0)
        self.assertEqual(orderbook.asks[0, 1], 0.4)
        self.assertEqual(orderbook.bids[0, 0], 60000.0)
        self.assertEqual(orderbook.bids[0, 1], 5.5)

        # A stale update comes in with previous prices
        best_ask_stale = np.array([[60000.0, 7.5]])
        best_bid_stale = np.array([[59999.0, 0.1]])
        orderbook.update_full(best_ask_stale, best_bid_stale, 5)

        self.assertEqual(orderbook.asks[0, 0], 60001.0)
        self.assertEqual(orderbook.asks[0, 1], 0.4)
        self.assertEqual(orderbook.bids[0, 0], 60000.0)
        self.assertEqual(orderbook.bids[0, 1], 5.5)

        # A full orderbook update comes in for all prices, some levels are removed
        fresh_asks = np.array([[60001.0 + i, 1.0 + i] for i in range(10)])
        fresh_asks[3, 1] = 0.0

        fresh_bids = np.array([[60000.0 - i, 1.0 + i] for i in range(10)])
        fresh_bids[3, 1] = 0.0

        orderbook.update_full(fresh_asks, fresh_bids, 7)

        self.assertEqual(orderbook.asks[0, 0], 60001.0)
        self.assertEqual(orderbook.asks[0, 1], 1.0)
        self.assertEqual(orderbook.bids[0, 0], 60000.0)
        self.assertEqual(orderbook.bids[0, 1], 1.0)

        # Zero Levels which are otherwise there, are not present
        self.assertNotEqual(orderbook.asks[3, 0], 60004.0)
        self.assertNotEqual(orderbook.asks[3, 1], 4.0)
        self.assertNotEqual(orderbook.bids[3, 0], 59997.0)
        self.assertNotEqual(orderbook.bids[3, 1], 4.0)


if __name__ == "__main__":
    unittest.main()
