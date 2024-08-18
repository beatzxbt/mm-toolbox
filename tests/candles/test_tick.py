import unittest
import numpy as np

from mm_toolbox.src.candles import TickCandles


class TestTickCandles(unittest.TestCase):
    def setUp(self):
        self.candles = TickCandles(ticks_per_bucket=3, num_candles=5)
    
    def test_update(self):
        self.candles.open_price = 104.0 
        self.candles.high_price = 112.0 
        self.candles.low_price = 91.0 
        self.candles.close_price = 109.0 
        self.candles.buy_volume = 10.0 
        self.candles.sell_volume = 94.0
        self.candles.vwap_price = 106.0
        self.candles.total_trades = 9.0
        self.candles.open_timestamp = 1609459260.0  

        self.candles.update(
            timestamp=1609459268.0,
            side=0.0,
            price=114.0,
            size=8.0
        )

        self.assertEqual(self.candles.open_price, 104.0)
        self.assertEqual(self.candles.high_price, 114.0)
        self.assertEqual(self.candles.low_price, 91.0)
        self.assertEqual(self.candles.close_price, 114.0)
        self.assertEqual(self.candles.buy_volume, 10.0)
        self.assertEqual(self.candles.sell_volume, 102.0) 
        self.assertAlmostEqual(self.candles.vwap_price, 106.297, places=3)  
        self.assertEqual(self.candles.total_trades, 10.0)
        self.assertEqual(self.candles.open_timestamp, 1609459260.0)
        self.assertEqual(self.candles.close_timestamp, 1609459268.0)
        self.assertTrue(self.candles.ringbuffer.is_empty)

    def test_stale_update(self):
        self.candles.open_price = 104.0 
        self.candles.high_price = 112.0 
        self.candles.low_price = 91.0 
        self.candles.close_price = 109.0 
        self.candles.buy_volume = 10.0 
        self.candles.sell_volume = 94.0
        self.candles.vwap_price = 106.0
        self.candles.total_trades = 9.0
        self.candles.open_timestamp = 1609459260.0  

        self.candles.update(
            timestamp=1609459250.0,
            side=0.0,
            price=114.0,
            size=8.0
        )

        self.assertEqual(self.candles.open_price, 104.0)
        self.assertEqual(self.candles.high_price, 112.0)
        self.assertEqual(self.candles.low_price, 91.0)
        self.assertEqual(self.candles.close_price, 109.0)
        self.assertEqual(self.candles.buy_volume, 10.0)
        self.assertEqual(self.candles.sell_volume, 94.0)
        self.assertEqual(self.candles.vwap_price, 106.0)
        self.assertEqual(self.candles.total_trades, 9.0)
        self.assertEqual(self.candles.open_timestamp, 0.0)
        self.assertEqual(self.candles.close_timestamp, 0.0)
        self.assertTrue(self.candles.ringbuffer.is_empty)

    def test_new_candle_creation(self):
        self.candles.update(
            timestamp=1609459250.0,
            side=1.0,
            price=114.0,
            size=8.0
        )

        self.candles.update(
            timestamp=1609459251.0,
            side=0.0,
            price=111.0,
            size=9.0
        )

        self.candles.update(
            timestamp=1609459252.0,
            side=0.0,
            price=119.0,
            size=8.0
        )

        self.assertEqual(self.candles.open_price, 114.0)
        self.assertEqual(self.candles.high_price, 119.0)
        self.assertEqual(self.candles.low_price, 111.0)
        self.assertEqual(self.candles.close_price, 119.0)
        self.assertEqual(self.candles.buy_volume, 17.0)
        self.assertEqual(self.candles.sell_volume, 8.0)
        self.assertEqual(self.candles.vwap_price, 114.647, places=3)
        self.assertEqual(self.candles.total_trades, 3.0)
        self.assertEqual(self.candles.open_timestamp, 1609459250.0)
        self.assertEqual(self.candles.close_timestamp, 1609459252.0)
        self.assertFalse(self.candles.ringbuffer.is_empty)

if __name__ == '__main__':
    unittest.main()