import unittest
from mm_toolbox.src.candles import VolumeCandles  


class TestVolumeCandles(unittest.TestCase):
    def setUp(self):
        self.candles = VolumeCandles(volume_per_bucket=100.0, num_candles=5)
    
    def test_update(self):
        start_time = 1609459200.0

        self.candles.process_trade(timestamp=start_time, side=0.0, price=100.0, size=40.0)
        self.candles.process_trade(timestamp=start_time + 10, side=1.0, price=105.0, size=30.0)
        self.candles.process_trade(timestamp=start_time + 20, side=0.0, price=110.0, size=25.0)

        self.assertEqual(self.candles.open_price, 100.0)
        self.assertEqual(self.candles.high_price, 110.0)
        self.assertEqual(self.candles.low_price, 100.0)
        self.assertEqual(self.candles.close_price, 110.0)
        self.assertEqual(self.candles.buy_volume, 65.0) 
        self.assertEqual(self.candles.sell_volume, 30.0)
        self.assertAlmostEqual(self.candles.vwap_price, 103.846, places=3)
        self.assertEqual(self.candles.total_trades, 3)
        self.assertEqual(self.candles.open_timestamp, start_time)
        self.assertEqual(self.candles.close_timestamp, 0.0) 

    def test_stale_update(self):
        start_time = 1609459200.0

        self.candles.process_trade(timestamp=start_time, side=0.0, price=100.0, size=40.0)
        self.candles.process_trade(timestamp=start_time - 60, side=1.0, price=105.0, size=30.0)

        self.assertEqual(self.candles.open_price, 100.0)
        self.assertEqual(self.candles.high_price, 100.0)
        self.assertEqual(self.candles.low_price, 100.0)
        self.assertEqual(self.candles.close_price, 100.0)
        self.assertEqual(self.candles.buy_volume, 40.0)
        self.assertEqual(self.candles.sell_volume, 0.0)
        self.assertAlmostEqual(self.candles.vwap_price, 100.0, places=3)
        self.assertEqual(self.candles.total_trades, 1)
        self.assertEqual(self.candles.open_timestamp, start_time)
        self.assertEqual(self.candles.close_timestamp, 0.0)

    def test_new_candle_creation(self):
        start_time = 1609459200.0

        self.candles.process_trade(timestamp=start_time, side=0.0, price=100.0, size=60.0)
        self.candles.process_trade(timestamp=start_time + 10, side=1.0, price=110.0, size=30.0)
        self.candles.process_trade(timestamp=start_time + 20, side=0.0, price=120.0, size=20.0)

        # Assert first candle
        self.assertEqual(len(self.candles), 1) 
        first_candle = self.candles[0]
        self.assertEqual(first_candle.open_price, 100.0)
        self.assertEqual(first_candle.high_price, 110.0)
        self.assertEqual(first_candle.low_price, 100.0)
        self.assertEqual(first_candle.close_price, 110.0)
        self.assertEqual(first_candle.buy_volume, 60.0)
        self.assertEqual(first_candle.sell_volume, 30.0)
        self.assertAlmostEqual(first_candle.vwap_price, 104.615, places=3)
        self.assertEqual(first_candle.total_trades, 2)
        self.assertEqual(first_candle.open_timestamp, start_time)
        self.assertEqual(first_candle.close_timestamp, start_time + 20)

        # Assert second candle (current)
        self.assertEqual(self.candles.open_price, 120.0)
        self.assertEqual(self.candles.high_price, 120.0)
        self.assertEqual(self.candles.low_price, 120.0)
        self.assertEqual(self.candles.close_price, 120.0)
        self.assertEqual(self.candles.buy_volume, 20.0)
        self.assertEqual(self.candles.sell_volume, 0.0)
        self.assertAlmostEqual(self.candles.vwap_price, 120.0, places=3)
        self.assertEqual(self.candles.total_trades, 1)
        self.assertEqual(self.candles.open_timestamp, start_time + 20)
        self.assertEqual(self.candles.close_timestamp, 0.0)

if __name__ == '__main__':
    unittest.main()
