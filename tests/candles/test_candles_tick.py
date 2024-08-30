import unittest

from src.mm_toolbox.candles import TickCandles
from src.mm_toolbox.time import time_ms


class TestTickCandles(unittest.TestCase):
    def setUp(self):
        self.dummy_ms = time_ms()
        self.candles = TickCandles(ticks_per_bucket=3, num_candles=5)

    def test_update(self):
        self.candles.update(timestamp=self.dummy_ms, side=0.0, price=114.0, size=8.0)

        current_candle = self.candles.current_candle
        self.assertEqual(current_candle[0], 114.0)
        self.assertEqual(current_candle[1], 114.0)
        self.assertEqual(current_candle[2], 114.0)
        self.assertEqual(current_candle[3], 114.0)
        self.assertEqual(current_candle[4], 8.0)
        self.assertEqual(current_candle[5], 0.0)
        self.assertEqual(current_candle[7], 1.0)
        self.assertEqual(current_candle[8], self.dummy_ms)
        self.assertEqual(current_candle[9], self.dummy_ms)
        self.assertTrue(self.candles.ringbuffer.is_empty)

    def test_stale_update(self):
        # Real update
        self.candles.update(timestamp=self.dummy_ms, side=0.0, price=114.0, size=8.0)

        current_candle = self.candles.current_candle
        self.assertEqual(current_candle[0], 114.0)
        self.assertEqual(current_candle[1], 114.0)
        self.assertEqual(current_candle[2], 114.0)
        self.assertEqual(current_candle[3], 114.0)
        self.assertEqual(current_candle[4], 8.0)
        self.assertEqual(current_candle[5], 0.0)
        self.assertEqual(current_candle[7], 1.0)
        self.assertEqual(current_candle[8], self.dummy_ms)
        self.assertEqual(current_candle[9], self.dummy_ms)
        self.assertTrue(self.candles.ringbuffer.is_empty)

        # Delayed/stale update
        self.candles.update(
            timestamp=self.dummy_ms - (10.0 * 1000.0),  # 10s older
            side=1.0,
            price=111.0,
            size=10.5,
        )

        # All candle properties are unchanged
        current_candle = self.candles.current_candle
        self.assertEqual(current_candle[0], 114.0)
        self.assertEqual(current_candle[1], 114.0)
        self.assertEqual(current_candle[2], 114.0)
        self.assertEqual(current_candle[3], 114.0)
        self.assertEqual(current_candle[4], 8.0)
        self.assertEqual(current_candle[5], 0.0)
        self.assertEqual(current_candle[7], 1.0)
        self.assertEqual(current_candle[8], self.dummy_ms)
        self.assertEqual(current_candle[9], self.dummy_ms)
        self.assertTrue(self.candles.ringbuffer.is_empty)

    def test_new_candle_creation(self):
        self.candles.update(timestamp=self.dummy_ms, side=1.0, price=114.0, size=8.0)

        self.candles.update(
            timestamp=self.dummy_ms + 50.0, side=0.0, price=115.0, size=9.0
        )

        self.candles.update(
            timestamp=self.dummy_ms + 95.0, side=0.0, price=117.0, size=8.0
        )

        created_candle = self.candles[0]
        self.assertEqual(created_candle[0], 114.0)
        self.assertEqual(created_candle[1], 117.0)
        self.assertEqual(created_candle[2], 114.0)
        self.assertEqual(created_candle[3], 117.0)
        self.assertEqual(created_candle[4], 17.0)
        self.assertEqual(created_candle[5], 8.0)
        self.assertEqual(created_candle[7], 3.0)
        self.assertEqual(created_candle[8], self.dummy_ms)
        self.assertEqual(created_candle[9], self.dummy_ms + 95.0)
        self.assertFalse(self.candles.ringbuffer.is_empty)


if __name__ == "__main__":
    unittest.main()
