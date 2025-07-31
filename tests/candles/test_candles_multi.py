import unittest
from mm_toolbox.candles import MultiCandles
from mm_toolbox.time import time_ms


class TestMultiCandles(unittest.TestCase):
    def setUp(self):
        # Initialize with 3-second duration, 5 ticks, and 100 volume limits
        self.dummy_ms = time_ms()
        self.candles = MultiCandles(
            max_duration_secs=3, max_ticks=5, max_volume=100.0, num_candles=5
        )

    def test_update_with_max_ticks(self):
        # Process trades until max_ticks is reached
        for i in range(5):  # max_ticks = 5
            self.candles.process_trade(
                time=self.dummy_ms + (i * 100), is_buy=True, px=110 + i, sz=20.0
            )

        # A new candle should have been inserted after 5 trades
        created_candle = self.candles[0]
        self.assertEqual(created_candle[0], 110.0)  # Open price
        self.assertEqual(created_candle[1], 114.0)  # High price
        self.assertEqual(created_candle[2], 110.0)  # Low price
        self.assertEqual(created_candle[3], 114.0)  # Close price
        self.assertEqual(created_candle[4], 100.0)  # Buy volume (5 trades * 20.0 size)
        self.assertEqual(created_candle[7], 5.0)  # Total trades

    def test_update_with_max_duration(self):
        # Simulate trades that exceed the max duration (3 seconds)
        self.candles.process_trade(time=self.dummy_ms, is_buy=True, px=110.0, sz=30.0)
        self.candles.process_trade(
            time=self.dummy_ms + 1000, is_buy=False, px=111.0, sz=25.0
        )
        self.candles.process_trade(
            time=self.dummy_ms + 2000, is_buy=True, px=112.0, sz=20.0
        )

        # The next trade should trigger a new candle due to max_duration_secs
        self.candles.process_trade(
            time=self.dummy_ms + 3100, is_buy=False, px=113.0, sz=15.0
        )

        created_candle = self.candles[0]
        self.assertEqual(created_candle[0], 110.0)  # Open px
        self.assertEqual(created_candle[1], 112.0)  # High px
        self.assertEqual(created_candle[2], 110.0)  # Low px
        self.assertEqual(created_candle[3], 112.0)  # Close px
        self.assertEqual(created_candle[4], 50.0)  # Buy sz (30 + 20)
        self.assertEqual(created_candle[5], 25.0)  # Sell sz
        self.assertEqual(created_candle[7], 3.0)  # Total trades

    def test_update_with_max_volume(self):
        # Process trades that exceed max_volume (100.0)
        self.candles.process_trade(time=self.dummy_ms, is_buy=True, px=110.0, sz=30.0)
        self.candles.process_trade(
            time=self.dummy_ms + 100, is_buy=False, px=111.0, sz=25.0
        )
        self.candles.process_trade(
            time=self.dummy_ms + 200, is_buy=True, px=112.0, sz=50.0
        )

        # Volume after the third trade exceeds max_volume (100)
        # A new candle should be created
        created_candle = self.candles[0]
        self.assertEqual(created_candle[0], 110.0)  # Open px
        self.assertEqual(created_candle[1], 112.0)  # High px
        self.assertEqual(created_candle[2], 110.0)  # Low px
        self.assertEqual(created_candle[3], 112.0)  # Close px
        self.assertEqual(created_candle[4], 75.0)  # Buy sz
        self.assertEqual(created_candle[5], 25.0)  # Sell sz
        self.assertEqual(created_candle[7], 3.0)  # Total trades

    def test_combined_triggers(self):
        # This test checks how multiple triggers (ticks, duration, volume) work together

        # First three trades, with ticks and volume under limits
        self.candles.process_trade(time=self.dummy_ms, is_buy=True, px=110.0, sz=10.0)
        self.candles.process_trade(
            time=self.dummy_ms + 100.0, is_buy=False, px=111.0, sz=20.0
        )
        self.candles.process_trade(
            time=self.dummy_ms + 200.0, is_buy=True, px=112.0, sz=25.0
        )

        # Another trade pushes volume beyond max_volume
        self.candles.process_trade(
            time=self.dummy_ms + 300.0, is_buy=True, px=113.0, sz=60.0
        )

        created_candle = self.candles[0]
        self.assertEqual(created_candle[0], 110.0)  # Open px
        self.assertEqual(created_candle[1], 113.0)  # High px
        self.assertEqual(created_candle[2], 110.0)  # Low px
        self.assertEqual(created_candle[3], 113.0)  # Close px
        self.assertEqual(created_candle[4], 35.0)  # Buy sz (10 + 25)
        self.assertEqual(created_candle[5], 65.0)  # Sell sz (20 + 60)
        self.assertEqual(created_candle[7], 4.0)  # Total trades
        self.assertEqual(created_candle[8], self.dummy_ms)  # Open time
        self.assertEqual(created_candle[9], self.dummy_ms + 300.0)  # Close time

    def test_exceeding_all_triggers(self):
        # First, process trades that exceed ticks
        for i in range(5):  # max_ticks = 5
            self.candles.process_trade(
                time=self.dummy_ms + (i * 100), is_buy=True, px=110 + i, sz=20.0
            )

        # Now, exceed duration
        self.candles.process_trade(
            time=self.dummy_ms + 4000, is_buy=True, px=115.0, sz=50.0
        )

        # Now, exceed volume
        self.candles.process_trade(
            time=self.dummy_ms + 5000, is_buy=True, px=116.0, sz=150.0
        )

        created_candle = self.candles[0]
        self.assertEqual(created_candle[0], 110.0)  # Open px
        self.assertEqual(created_candle[1], 114.0)  # High px
        self.assertEqual(created_candle[2], 110.0)  # Low px
        self.assertEqual(created_candle[3], 114.0)  # Close px
        self.assertEqual(created_candle[4], 100.0)  # Buy sz (5 trades)
        self.assertEqual(created_candle[5], 0.0)  # Sell sz
        self.assertEqual(created_candle[7], 5.0)  # Total trades


if __name__ == "__main__":
    unittest.main()
