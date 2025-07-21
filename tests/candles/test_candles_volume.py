import unittest
from mm_toolbox.candles import VolumeCandles
from mm_toolbox.time import time_ms


class TestVolumeCandles(unittest.TestCase):
    def setUp(self):
        self.dummy_ms = time_ms()
        self.candles = VolumeCandles(
            volume_per_bucket=10.0, 
            num_candles=5,
        )

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
                px=100.0 + i,
                sz=6.0
            )

        first_candle = self.candles[0]
        self.assertEqual(first_candle[0], 100.0)
        self.assertEqual(first_candle[1], 101.0)
        self.assertEqual(first_candle[2], 100.0)
        self.assertEqual(first_candle[3], 101.0)
        self.assertEqual(first_candle[4], 6.0)
        self.assertEqual(first_candle[7], 4.0)
        self.assertEqual(first_candle[8], self.dummy_ms)
        self.assertEqual(first_candle[9], self.dummy_ms + 3000.0)

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

        # Delayed/stale update
        self.candles.update(
            timestamp=self.dummy_ms - 1000.0,  # 1s older
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

    def test_new_candle_creation(self):
        self.candles.update(timestamp=self.dummy_ms, side=1.0, price=114.0, size=8.0)


        self.candles.update(
            timestamp=self.dummy_ms + 50.0,
            side=0.0,
            price=115.0,
            size=1.0,  # Not yet enough
        )


        self.candles.update(
            timestamp=self.dummy_ms + 60.0, side=0.0, price=117.0, size=8.0
        )

        created_candle = self.candles.as_array()[0]
        self.assertEqual(created_candle[0], 114.0)
        self.assertEqual(created_candle[1], 117.0)
        self.assertEqual(created_candle[2], 114.0)
        self.assertEqual(created_candle[3], 117.0)
        self.assertEqual(created_candle[4], 2.0)
        self.assertEqual(created_candle[5], 8.0)
        self.assertEqual(created_candle[7], 3.0)
        self.assertEqual(created_candle[8], self.dummy_ms)
        self.assertEqual(created_candle[9], self.dummy_ms + 60.0)
        self.assertFalse(self.candles.ringbuffer.is_empty)

        current_candle = self.candles.as_array()[1]
        self.assertEqual(current_candle[0], 117.0)
        self.assertEqual(current_candle[1], 117.0)
        self.assertEqual(current_candle[2], 117.0)
        self.assertEqual(current_candle[3], 117.0)
        self.assertEqual(current_candle[4], 7.0)
        self.assertEqual(current_candle[5], 0.0)
        self.assertEqual(current_candle[7], 1.0)
        self.assertEqual(current_candle[8], self.dummy_ms + 60.0)
        self.assertEqual(current_candle[9], self.dummy_ms + 60.0)
        self.assertFalse(self.candles.ringbuffer.is_empty)


if __name__ == "__main__":
    unittest.main()
