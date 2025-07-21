import unittest
from mm_toolbox.candles import TimeCandles
from mm_toolbox.time import time_ms


class TestTimeCandles(unittest.TestCase):
    def setUp(self):
        self.dummy_ms = time_ms()
        self.candles = TimeCandles(secs_per_bucket=60, num_candles=5)

    def test_incomplete_candle(self):
        self.candles.process_trade(
            time=self.dummy_ms,
            is_buy=True,
            px=114.0,
            sz=8.0
        )

        current_candle = self.candles[0]
        self.assertEqual(current_candle[0], 114.0)
        self.assertEqual(current_candle[1], 114.0)
        self.assertEqual(current_candle[2], 114.0)
        self.assertEqual(current_candle[3], 114.0)
        self.assertEqual(current_candle[4], 8.0)
        self.assertEqual(current_candle[5], 0.0)
        self.assertEqual(current_candle[7], 1.0)
        self.assertEqual(current_candle[8], self.dummy_ms)
        self.assertEqual(current_candle[9], self.dummy_ms)

    def test_complete_candle(self):
        for i in range(4):
            self.candles.process_trade(
                time=self.dummy_ms + (i * 1000.0),
                is_buy=True,
                px=114.0,
                sz=8.0
            )

        first_candle = self.candles[0]
        self.assertEqual(first_candle[0], 114.0)
        self.assertEqual(first_candle[1], 114.0)
        self.assertEqual(first_candle[2], 114.0)
        self.assertEqual(first_candle[3], 114.0)
        self.assertEqual(first_candle[4], 8.0)
        self.assertEqual(first_candle[7], 4.0)
        self.assertEqual(first_candle[8], self.dummy_ms)
        self.assertEqual(first_candle[9], self.dummy_ms + 3000.0)

        second_candle = self.candles[1]
        self.assertEqual(second_candle[0], 114.0)
        self.assertEqual(second_candle[1], 114.0)
        self.assertEqual(second_candle[2], 114.0)
        self.assertEqual(second_candle[3], 114.0)
        self.assertEqual(second_candle[4], 8.0)
        self.assertEqual(second_candle[7], 4.0)
        self.assertEqual(second_candle[8], self.dummy_ms + 4000.0)
        self.assertEqual(second_candle[9], self.dummy_ms + 7000.0)

    def test_stale_update(self):
        # Valid trade
        self.candles.process_trade(
            time=self.dummy_ms,
            is_buy=True,
            px=114.0,
            sz=8.0
        )

        # Stale trade
        self.candles.process_trade(
            time=self.dummy_ms - 1000.0,
            is_buy=True,
            px=111.0,
            sz=10.5,
        )

        # All candle properties are unchanged
        current_candle = self.candles[0]
        self.assertEqual(current_candle[0], 114.0)
        self.assertEqual(current_candle[1], 114.0)
        self.assertEqual(current_candle[2], 114.0)
        self.assertEqual(current_candle[3], 114.0)
        self.assertEqual(current_candle[4], 8.0)
        self.assertEqual(current_candle[5], 0.0)
        self.assertEqual(current_candle[7], 1.0)
        self.assertEqual(current_candle[8], self.dummy_ms)
        self.assertEqual(current_candle[9], self.dummy_ms)

if __name__ == "__main__":
    unittest.main()
