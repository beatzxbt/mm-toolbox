import unittest
from mm_toolbox.src.candles import TimeCandles 
from mm_toolbox.src.time import time_s


class TestTimeCandles(unittest.TestCase):
    def setUp(self):
        self.candles = TimeCandles(seconds_per_bucket=60, num_candles=5)
    
    def test_update(self):
        start_time = time_s()

        self.candles.process_trade(timestamp=start_time, side=0.0, price=100.0, size=10.0)
        self.candles.process_trade(timestamp=start_time + 30, side=1.0, price=105.0, size=15.0)

        self.assertEqual(self.candles.open_price, 100.0)
        self.assertEqual(self.candles.high_price, 105.0)
        self.assertEqual(self.candles.low_price, 100.0)
        self.assertEqual(self.candles.close_price, 105.0)
        self.assertEqual(self.candles.buy_volume, 10.0)
        self.assertEqual(self.candles.sell_volume, 15.0)
        self.assertAlmostEqual(self.candles.vwap_price, 103.0, places=3)
        self.assertEqual(self.candles.total_trades, 2)
        self.assertEqual(self.candles.open_timestamp, start_time)
        self.assertEqual(self.candles.close_timestamp, start_time + 30)

    def test_stale_update(self):
        start_time = time_s()

        self.candles.process_trade(timestamp=start_time, side=0.0, price=100.0, size=10.0)
        self.candles.process_trade(timestamp=start_time - 60, side=1.0, price=105.0, size=15.0)  # Stale trade

        self.assertEqual(self.candles.open_price, 100.0)
        self.assertEqual(self.candles.high_price, 100.0)
        self.assertEqual(self.candles.low_price, 100.0)
        self.assertEqual(self.candles.close_price, 100.0)
        self.assertEqual(self.candles.buy_volume, 10.0)
        self.assertEqual(self.candles.sell_volume, 0.0)
        self.assertAlmostEqual(self.candles.vwap_price, 100.0, places=3)
        self.assertEqual(self.candles.total_trades, 1)
        self.assertEqual(self.candles.open_timestamp, start_time)
        self.assertEqual(self.candles.close_timestamp, 0.0)  # No close yet

    def test_new_candle_creation(self):
        start_time = time_s()

        self.candles.process_trade(timestamp=start_time, side=0.0, price=100.0, size=10.0)
        self.candles.process_trade(timestamp=start_time + 61, side=1.0, price=110.0, size=5.0)

        self.assertEqual(len(self.candles), 1)  
        first_candle = self.candles[0]
        self.assertEqual(first_candle.open_price, 100.0)
        self.assertEqual(first_candle.high_price, 100.0)
        self.assertEqual(first_candle.low_price, 100.0)
        self.assertEqual(first_candle.close_price, 100.0)
        self.assertEqual(first_candle.buy_volume, 10.0)
        self.assertEqual(first_candle.sell_volume, 0.0)
        self.assertAlmostEqual(first_candle.vwap_price, 100.0, places=3)
        self.assertEqual(first_candle.total_trades, 1)
        self.assertEqual(first_candle.open_timestamp, start_time)
        self.assertEqual(first_candle.close_timestamp, start_time)

        # Assert second candle (current)
        self.assertEqual(self.candles.open_price, 110.0)
        self.assertEqual(self.candles.high_price, 110.0)
        self.assertEqual(self.candles.low_price, 110.0)
        self.assertEqual(self.candles.close_price, 110.0)
        self.assertEqual(self.candles.buy_volume, 0.0)
        self.assertEqual(self.candles.sell_volume, 5.0)
        self.assertAlmostEqual(self.candles.vwap_price, 110.0, places=3)
        self.assertEqual(self.candles.total_trades, 1)
        self.assertEqual(self.candles.open_timestamp, start_time + 61)
        self.assertEqual(self.candles.close_timestamp, start_time + 61)

if __name__ == '__main__':
    unittest.main()
