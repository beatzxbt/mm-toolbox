import unittest
from mm_toolbox.candles import MultiTriggerCandles
from mm_toolbox.time import time_ms


class TestMultiTriggerCandles(unittest.TestCase):
    def setUp(self):
        # Initialize with 3-second duration, 5 ticks, and 100 volume limits
        self.dummy_ms = time_ms()
        self.candles = MultiTriggerCandles(
            max_duration_secs=3, max_ticks=5, max_volume=100.0, num_candles=5
        )

    def test_update_with_max_ticks(self):
        # Process trades until max_ticks is reached
        for i in range(5):  # max_ticks = 5
            self.candles.update(
                timestamp=self.dummy_ms + (i * 100), side=0.0, price=110 + i, size=20.0
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
        self.candles.update(timestamp=self.dummy_ms, side=0.0, price=110.0, size=30.0)
        self.candles.update(
            timestamp=self.dummy_ms + 1000, side=1.0, price=111.0, size=25.0
        )
        self.candles.update(
            timestamp=self.dummy_ms + 2000, side=0.0, price=112.0, size=20.0
        )

        # The next trade should trigger a new candle due to max_duration_secs
        self.candles.update(
            timestamp=self.dummy_ms + 3100, side=1.0, price=113.0, size=15.0
        )

        created_candle = self.candles[0]
        self.assertEqual(created_candle[0], 110.0)  # Open price
        self.assertEqual(created_candle[1], 112.0)  # High price
        self.assertEqual(created_candle[2], 110.0)  # Low price
        self.assertEqual(
            created_candle[3], 112.0
        )  # Close price (before duration exceeded)
        self.assertEqual(created_candle[4], 50.0)  # Buy volume (30 + 20)
        self.assertEqual(created_candle[5], 25.0)  # Sell volume
        self.assertEqual(created_candle[7], 3.0)  # Total trades

    def test_update_with_max_volume(self):
        # Process trades that exceed max_volume (100.0)
        self.candles.update(timestamp=self.dummy_ms, side=0.0, price=110.0, size=30.0)
        self.candles.update(
            timestamp=self.dummy_ms + 100, side=1.0, price=111.0, size=25.0
        )
        self.candles.update(
            timestamp=self.dummy_ms + 200, side=0.0, price=112.0, size=50.0
        )

        # Volume after the third trade exceeds max_volume (100)
        # A new candle should be created
        created_candle = self.candles[0]
        self.assertEqual(created_candle[0], 110.0)  # Open price
        self.assertEqual(created_candle[1], 112.0)  # High price
        self.assertEqual(created_candle[2], 110.0)  # Low price
        self.assertEqual(created_candle[3], 112.0)  # Close price
        self.assertEqual(created_candle[4], 75.0)  # Buy volume
        self.assertEqual(created_candle[5], 25.0)  # Sell volume
        self.assertEqual(created_candle[7], 3.0)  # Total trades

    def test_combined_triggers(self):
        # This test checks how multiple triggers (ticks, duration, volume) work together

        # First three trades, with ticks and volume under limits
        self.candles.update(timestamp=self.dummy_ms, side=0.0, price=110.0, size=10.0)
        self.candles.update(
            timestamp=self.dummy_ms + 100, side=1.0, price=111.0, size=20.0
        )
        self.candles.update(
            timestamp=self.dummy_ms + 200, side=0.0, price=112.0, size=25.0
        )

        # Another trade pushes volume beyond max_volume
        self.candles.update(
            timestamp=self.dummy_ms + 300, side=1.0, price=113.0, size=60.0
        )

        created_candle = self.candles[0]
        self.assertEqual(created_candle[0], 110.0)  # Open price
        self.assertEqual(created_candle[1], 113.0)  # High price
        self.assertEqual(created_candle[2], 110.0)  # Low price
        self.assertEqual(created_candle[3], 113.0)  # Close price
        self.assertEqual(created_candle[4], 35.0)  # Buy volume (10 + 25)
        self.assertEqual(created_candle[5], 65.0)  # Sell volume (20 + 60)
        self.assertEqual(created_candle[7], 4.0)  # Total trades
        self.assertEqual(created_candle[8], self.dummy_ms)  # Open timestamp
        self.assertEqual(created_candle[9], self.dummy_ms + 300)  # Close timestamp

    def test_exceeding_all_triggers(self):
        # First, process trades that exceed ticks
        for i in range(5):  # max_ticks = 5
            self.candles.update(
                timestamp=self.dummy_ms + (i * 100), side=0.0, price=110 + i, size=20.0
            )

        # Now, exceed duration
        self.candles.update(
            timestamp=self.dummy_ms + 4000, side=1.0, price=115.0, size=50.0
        )

        # Now, exceed volume
        self.candles.update(
            timestamp=self.dummy_ms + 5000, side=1.0, price=116.0, size=150.0
        )

        created_candle = self.candles[0]
        self.assertEqual(created_candle[0], 110.0)  # Open price
        self.assertEqual(
            created_candle[1], 114.0
        )  # High price (from the first 5 trades)
        self.assertEqual(created_candle[2], 110.0)  # Low price
        self.assertEqual(created_candle[3], 114.0)  # Close price
        self.assertEqual(created_candle[4], 100.0)  # Buy volume (5 trades)
        self.assertEqual(created_candle[5], 0.0)  # Sell volume
        self.assertEqual(created_candle[7], 5.0)  # Total trades


if __name__ == "__main__":
    unittest.main()
