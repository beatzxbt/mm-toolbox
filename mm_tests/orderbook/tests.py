import sys
from pathlib import Path
project_path = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_path))

import unittest
import numpy as np
from mm_toolbox.time.time import time_ms
from mm_toolbox.orderbook.orderbook import *

class TestOrderbook(unittest.TestCase):
    def setUp(self):
        self.tick_size = 0.01
        self.lot_size = 0.1
        self.num_levels = 10
        self.orderbook = Orderbook(self.tick_size, self.lot_size, self.num_levels)

    def helper_reset_warmup_orderbook(self): 
        self.orderbook.reset()
        self.orderbook.warmup_asks(101.0, 10.0)
        self.orderbook.warmup_bids(99.0, 10.0)
        self.orderbook.warmed_up = True
    
    def test_initialization(self):
        # Tests that the order book initializes correctly
        self.assertEqual(self.orderbook.tick_size, 0.01)
        self.assertEqual(self.orderbook.lot_size, 0.1)
        self.assertEqual(self.orderbook.num_levels, 10)
        self.assertFalse(self.orderbook.warmed_up)
        self.assertEqual(self.orderbook.last_updated_timestamp, 0)
        self.assertEqual(self.orderbook.asks.shape, (10, 2))
        self.assertEqual(self.orderbook.bids.shape, (10, 2))
        self.assertTrue(np.all(self.orderbook.asks == 0))
        self.assertTrue(np.all(self.orderbook.bids == 0))

    def test_reset(self):
        # Set internal arrays/ptrs to some random values
        self.orderbook.bids[:, 0] = np.arange(10, 100, 10, dtype=np.int64)
        self.orderbook.bids[:, 1] = np.arange(1, 10, 1, dtype=np.int64)
        self.orderbook.asks[:, 0] = np.arange(110, 210, 10, dtype=np.int64)
        self.orderbook.asks[:, 1] = np.arange(1, 10, 1, dtype=np.int64)
        self.orderbook.last_updated_timestamp = time_ms()
        self.orderbook.warmed_up = True

        self.orderbook.reset()

        # Check if everything has been reset properly
        self.assertEqual(self.orderbook.last_updated_timestamp, 0)
        self.assertFalse(self.orderbook.warmed_up)
        self.assertTrue(np.all(self.orderbook.asks == 0))
        self.assertTrue(np.all(self.orderbook.bids == 0))
        self.assertEqual(self.orderbook.asks.shape, (10, 2))
        self.assertEqual(self.orderbook.bids.shape, (10, 2))
    
    def test_warmup_asks(self):
        # Tests the warming up of the ask levels
        self.orderbook.reset()
        self.orderbook.warmup_asks(101.0, 0.9)

        self.assertTrue(self.orderbook.warmed_up)
    
        # Check best ask is correct
        self.assertEqual(self.orderbook.asks[0, 0], self.orderbook.normalize(101.0, self.tick_size))
        self.assertEqual(self.orderbook.asks[0, 1], self.orderbook.normalize(0.9, self.lot_size))

        # Check worst ask is exactly best ask + number of levels & size is zero
        self.assertEqual(self.orderbook.asks[self.num_levels-1, 0], self.orderbook.asks[0, 0] - self.num_levels)
        self.assertEqual(self.orderbook.asks[self.num_levels-1, 1], self.orderbook.normalize(0.0, self.lot_size))
        
    def test_warmup_bids(self):
        # Tests the warming up of the bid levels
        self.orderbook.warmup_bids(99.0, 7.1)
        self.orderbook.reset()

        self.assertTrue(self.orderbook.warmed_up)
        
        # Check best bid is correct
        self.assertEqual(self.orderbook.bids[-1, 0], self.orderbook.normalize(99.0, self.tick_size))
        self.assertEqual(self.orderbook.bids[-1, 1], self.orderbook.normalize(7.1, self.lot_size))

        # Check worst bid is exactly best bid - number of levels & size is zero
        self.assertEqual(self.orderbook.bids[0, 0], self.orderbook.bids[-1, 0] - self.num_levels)
        self.assertEqual(self.orderbook.bids[0, 1], self.orderbook.normalize(0.0, self.lot_size))

    def test_single_ingest_l2_update(self):
        # Tests processing a single L2 ask update (warmup + single update)
        self.helper_reset_warmup_orderbook()
        asks = np.array([[101.0, 9.3]], dtype=np.float64)
        bids = np.array([[99.0, 15.1]], dtype=np.float64)
        self.orderbook.ingest_l2_update(time_ms(), asks, bids)

        # Check best ask is correct
        self.assertEqual(self.orderbook.asks[0, 0], self.orderbook.normalize(101.0, self.tick_size))
        self.assertEqual(self.orderbook.asks[0, 1], self.orderbook.normalize(15.1, self.lot_size))

        # Check best bid is correct
        self.assertEqual(self.orderbook.bids[-1, 0], self.orderbook.normalize(99.0, self.tick_size))
        self.assertEqual(self.orderbook.bids[-1, 1], self.orderbook.normalize(9.3, self.lot_size))

    def test_multiple_ingest_l2_update(self):
        # Tests processing multiple L2 ask updates (warmup + multi update)
        self.helper_reset_warmup_orderbook()
        asks = np.array([[101.1, 9.3], [101.5, 95.0], [103.3, 59.1]], dtype=np.float64)
        bids = np.array([[99.0, 15.1], [98.4, 9.1], [97.9, 0.9]], dtype=np.float64)
        self.orderbook.ingest_l2_update(time_ms(), asks, bids)
        
        # Define expected results for the first three levels of bids and asks
        expected_bids = [
            (self.orderbook.normalize(99.0, self.tick_size), self.orderbook.normalize(15.1, self.lot_size)),
            (self.orderbook.normalize(98.4, self.tick_size), self.orderbook.normalize(9.1, self.lot_size)),
            (self.orderbook.normalize(97.9, self.tick_size), self.orderbook.normalize(0.9, self.lot_size))
        ]
        
        expected_asks = [
            (self.orderbook.normalize(101.1, self.tick_size), self.orderbook.normalize(9.3, self.lot_size)),
            (self.orderbook.normalize(101.5, self.tick_size), self.orderbook.normalize(95.0, self.lot_size)),
            (self.orderbook.normalize(103.3, self.tick_size), self.orderbook.normalize(59.1, self.lot_size))
        ]
        
        # Check the first three levels of bids and asks
        for level in range(3):
            with self.subTest(level=level):
                self.assertEqual(self.orderbook.bids[level, 0], expected_bids[level][0], "Bid price mismatch at level {level}")
                self.assertEqual(self.orderbook.bids[level, 1], expected_bids[level][1], "Bid size mismatch at level {level}")
                
                self.assertEqual(self.orderbook.asks[level, 0], expected_asks[level][0], "Ask price mismatch at level {level}")
                self.assertEqual(self.orderbook.asks[level, 1], expected_asks[level][1], "Ask size mismatch at level {level}")

    def test_single_ingest_buy_trade_update(self):
        # Tests processing a single trade marked as a buy
        self.helper_reset_warmup_orderbook()
        ts, isBuy, price, size = (time_ms(), True, 101.0, 2.0)
        self.orderbook.ingest_trade_update(ts, isBuy, price, size)
        
        self.assertEqual(self.orderbook.asks[0, 0], self.orderbook.normalize(101.0, self.tick_size))
        self.assertEqual(self.orderbook.asks[0, 1], self.orderbook.normalize(10.0-2.0, self.lot_size))

    def test_single_ingest_sell_trade_update(self):
        # Tests processing a single trade marked as a sell
        self.helper_reset_warmup_orderbook()
        ts, isBuy, price, size = (time_ms(), False, 99.0, 2.0)
        self.orderbook.ingest_trade_update(ts, isBuy, price, size)
        
        self.assertEqual(self.orderbook.asks[0, 0], self.orderbook.normalize(99.0, self.tick_size))
        self.assertEqual(self.orderbook.asks[0, 1], self.orderbook.normalize(10.0-2.0, self.lot_size))

    def test_mixed_updates(self):
        self.helper_reset_warmup_orderbook()

        # Normal L2 update
        asks = np.array([[101.0, 9.3], [101.1, 95.0], [101.2, 59.1]], dtype=np.float64)
        bids = np.array([[99.0, 15.1], [98.9, 9.1], [98.8, 0.9]], dtype=np.float64)
        self.orderbook.ingest_l2_update(time_ms(), asks, bids)

        # Partial fill on best ask
        self.orderbook.ingest_trade_update(time_ms(), True, 101.0, 5.0)

        # Full fill on best ask
        self.orderbook.ingest_trade_update(time_ms(), True, 101.0, 4.3)

        # Bid chasing new best ask
        asks = np.array([[]], dtype=np.float64) # NOTE: Simulating exchange not sending size = 0 as expected
        bids = np.array([[99.1, 3.1]], dtype=np.float64)
        self.orderbook.ingest_l2_update(time_ms(), asks, bids)

        # Full fill on 2 bid levels
        self.orderbook.ingest_trade_update(time_ms(), False, 99.1, 3.1)
        self.orderbook.ingest_trade_update(time_ms(), False, 99.0, 15.1)

        # Ask chasing new best bid
        asks = np.array([[99.1, 1.0], [99.2, 8.1], [99.3, 0], [99.7, 18.8]], dtype=np.float64)
        bids = np.array([[98.6, 3.1], [98.4, 5.3]], dtype=np.float64)
        self.orderbook.ingest_l2_update(time_ms(), asks, bids)

        # Full fill eating all bid book levels
        self.orderbook.ingest_trade_update(time_ms(), False, 98.8, 0.9)
        self.orderbook.ingest_trade_update(time_ms(), False, 98.6, 3.1)
        self.orderbook.ingest_trade_update(time_ms(), False, 98.4, 5.3)
        self.orderbook.ingest_trade_update(time_ms(), False, 98.2, 0.4) # NOTE: Simulating out of orderbook trade

        # Large L2 update 
        asks = np.array([[98.5, 1.0], [98.7, 8.1], [98.9, 0.0], [99.7, 18.8]], dtype=np.float64)
        bids = np.array([[98.2, 3.1], [97.5, 5.3], [97.2, 0.19]], dtype=np.float64)
        self.orderbook.ingest_l2_update(time_ms(), asks, bids)

        # Define expected results for the first three levels of bids and asks after all updates
        expected_bids = [
            (self.orderbook.normalize(98.2, self.tick_size), self.orderbook.normalize(3.1, self.lot_size)),
            (self.orderbook.normalize(97.5, self.tick_size), self.orderbook.normalize(5.3, self.lot_size)),
            (self.orderbook.normalize(97.2, self.tick_size), self.orderbook.normalize(0.19, self.lot_size))
        ]
        
        expected_asks = [
            (self.orderbook.normalize(98.5, self.tick_size), self.orderbook.normalize(1.0, self.lot_size)),
            (self.orderbook.normalize(98.7, self.tick_size), self.orderbook.normalize(8.1, self.lot_size)),
            (self.orderbook.normalize(98.9, self.tick_size), self.orderbook.normalize(0.0, self.lot_size))
        ]
        
        # Check the first three levels of bids and asks
        for level in range(3):
            with self.subTest(level=level):
                self.assertEqual(self.orderbook.bids[level, 0], expected_bids[level][0], f"Bid price mismatch at level {level}")
                self.assertEqual(self.orderbook.bids[level, 1], expected_bids[level][1], f"Bid size mismatch at level {level}")
                
                self.assertEqual(self.orderbook.asks[level, 0], expected_asks[level][0], f"Ask price mismatch at level {level}")
                self.assertEqual(self.orderbook.asks[level, 1], expected_asks[level][1], f"Ask size mismatch at level {level}")

    def test_get_best_bid(self):
        # Tests retrieval of the best bid
        self.helper_reset_warmup_orderbook()
        best_bid_price, best_bid_size = self.orderbook.get_best_bid()

        self.assertEqual(best_bid_price, 99.0)
        self.assertEqual(best_bid_size, 10.0)

    def test_get_best_ask(self):
        # Tests retrieval of the best ask
        self.helper_reset_warmup_orderbook()
        best_ask_price, best_ask_size = self.orderbook.get_best_ask()

        self.assertEqual(best_ask_price, 101.0)
        self.assertEqual(best_ask_size, 10.0)

    def test_get_mid(self):
        # Tests calculation of the mid price
        self.helper_reset_warmup_orderbook()
        mid_price = self.orderbook.get_mid()

        self.assertEqual(mid_price, 100.0)

    def test_get_wmid(self):
        # Tests calculation of the weighted mid price
        self.helper_reset_warmup_orderbook()
        wmid = self.orderbook.get_wmid()

        expected_wmid = (99.0 * (10.0 / 20.0)) + (101.0 * (10.0 / 20.0))
        self.assertEqual(wmid, expected_wmid)

    def test_get_spread(self):
        # Tests calculation of the book spread
        self.helper_reset_warmup_orderbook()
        spread = self.orderbook.get_spread()

        self.assertEqual(spread, 20.0)

    def test_get_vamp(self):
        # Tests calculation of the volume-weighted average market price
        self.helper_reset_warmup_orderbook()
        asks = np.array([[101.1, 9.3], [101.5, 95.0], [103.3, 59.1]], dtype=np.float64)
        bids = np.array([[99.0, 15.1], [98.4, 9.1], [97.9, 7.9]], dtype=np.float64)
        self.orderbook.ingest_l2_update(time_ms(), asks, bids)
        vamp = self.orderbook.get_vamp(2500.0)

        self.assertEqual(np.round(vamp, 3), 98.737)

    def test_get_slippage(self):
        # Tests calculation of slippage cost
        asks = np.array([[101.1, 9.3], [101.5, 95.0], [103.3, 59.1]], dtype=np.float64)
        bids = np.array([[99.0, 15.1], [98.4, 9.1], [97.9, 7.9]], dtype=np.float64)
        self.orderbook.ingest_l2_update(time_ms(), asks, bids)
        slippage_bid = self.orderbook.get_slippage(self.orderbook.bids[::-1], 2500.0)  
        slippage_ask = self.orderbook.get_slippage(self.orderbook.asks, 2500.0)

        self.assertTrue(isinstance(slippage_bid, float))
        self.assertTrue(isinstance(slippage_ask, float))

if __name__ == "__main__":
    unittest.main()
