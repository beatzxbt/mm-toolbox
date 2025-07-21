import unittest
import numpy as np

from mm_toolbox.candles import TimeCandles

# As we cannot mock the base class directly within this Python file,
# we will use the TimeCandles class to test the base class. The processing
# logic for TimeCandles is tested separately in 'test_candles_time.py'
class TestBaseCandles(unittest.TestCase):
    def setUp(self):
        self.candles = TimeCandles(
            secs_per_bucket=1,
            num_candles=5
        )

    def test_single_helpers(self):
        # Test helpers when no trades have been processed
        # These will be used for future tests, and thus need
        # to be tested first.
        self.assertEqual(self.candles.get_open_px(), 0.0)
        self.assertEqual(self.candles.get_high_px(), -np.inf)
        self.assertEqual(self.candles.get_low_px(), np.inf)
        self.assertEqual(self.candles.get_close_px(), 0.0)
        self.assertEqual(self.candles.get_buy_sz(), 0.0)
        self.assertEqual(self.candles.get_sell_sz(), 0.0)
        self.assertEqual(self.candles.get_vwap_px(), 0.0)
        self.assertEqual(self.candles.get_total_trades(), 0.0)
        self.assertEqual(self.candles.get_open_time(), 0.0)
        self.assertEqual(self.candles.get_close_time(), 0.0)

    def test_insert_candle(self):
        # At time 1609459200.0, a buy of 500.0 is processed.
        self.candles.process_trade(
            time=1609459200.0,
            is_buy=True,
            px=100.0,
            sz=500.0
        )

        # 5s later, another buy of 500.0 is processed.
        self.candles.process_trade(
            time=1609459205.0,
            is_buy=True,
            px=100.0,
            sz=500.0
        )
        
        # Calling .process_trade() implicitly calls .insert_candle()
        # so we can test the stored candle directly.
        self.assertEqual(self.candles.get_open_px(), 100.0)
        self.assertEqual(self.candles.get_high_px(), 100.0)
        self.assertEqual(self.candles.get_low_px(), 100.0)
        self.assertEqual(self.candles.get_close_px(), 100.0)
        self.assertEqual(self.candles.get_buy_sz(), 1000.0)
        self.assertEqual(self.candles.get_sell_sz(), 0.0)
        self.assertEqual(self.candles.get_vwap_px(), 100.0)
        self.assertEqual(self.candles.get_total_trades(), 2.0)
        self.assertEqual(self.candles.get_open_time(), 1609459200.0)
        self.assertEqual(self.candles.get_close_time(), 1609459205.0)

    def test_as_array_without_current(self):
        self.candles.get_open_px() = 100.0
        self.candles.get_high_px() = 110.0
        self.candles.get_low_px() = 90.0
        self.candles.get_close_px() = 105.0
        self.candles.get_buy_sz() = 500.0
        self.candles.get_sell_sz() = 200.0
        self.candles.get_vwap_px() = 102.0
        self.candles.get_total_trades() = 3.0
        self.candles.get_open_time() = 1609459200.0
        self.candles.get_close_time() = 1609459260.0

        candle_array = self.candles.as_array()
        expected_candles = np.array(
            [
                [
                    100.0,
                    110.0,
                    90.0,
                    105.0,
                    500.0,
                    200.0,
                    102.0,
                    3.0,
                    1609459200.0,
                    1609459260.0,
                ]
            ]
        )
        np.testing.assert_array_equal(candle_array, expected_candles)

    def test_as_array_with_current(self):
        self.candles.open_price = 100.0
        self.candles.high_price = 110.0
        self.candles.low_price = 90.0
        self.candles.close_price = 105.0
        self.candles.buy_volume = 500.0
        self.candles.sell_volume = 200.0
        self.candles.vwap_price = 102.0
        self.candles.total_trades = 3.0
        self.candles.open_timestamp = 1609459200.0
        self.candles.close_timestamp = 1609459260.0

        self.candles.insert_candle()

        # Modify current candle data
        self.candles.open_price = 104.0
        self.candles.high_price = 112.0
        self.candles.low_price = 91.0
        self.candles.close_price = 109.0
        self.candles.buy_volume = 10.0
        self.candles.sell_volume = 94.0
        self.candles.vwap_price = 106.0
        self.candles.total_trades = 9.0
        self.candles.open_timestamp = 1609459260.0
        self.candles.close_timestamp = 1609459320.0

        candle_array = self.candles.as_array()
        expected_candles = np.array(
            [
                [
                    100.0,
                    110.0,
                    90.0,
                    105.0,
                    500.0,
                    200.0,
                    102.0,
                    3.0,
                    1609459200.0,
                    1609459260.0,
                ],
                [
                    104.0,
                    112.0,
                    91.0,
                    109.0,
                    10.0,
                    94.0,
                    106.0,
                    9.0,
                    1609459260.0,
                    1609459320.0,
                ],
            ]
        )
        np.testing.assert_array_equal(candle_array, expected_candles)

    def test_reset_current_candle(self):
        self.candles.open_price = 104.0
        self.candles.high_price = 112.0
        self.candles.low_price = 91.0
        self.candles.close_price = 109.0
        self.candles.buy_volume = 10.0
        self.candles.sell_volume = 94.0
        self.candles.vwap_price = 106.0
        self.candles.total_trades = 9.0
        self.candles.open_timestamp = 1609459260.0
        self.candles.close_timestamp = 1609459320.0

        self.candles.reset_current_candle()

        self.assertEqual(self.candles.open_price, 0.0)
        self.assertEqual(self.candles.high_price, -np.inf)
        self.assertEqual(self.candles.low_price, np.inf)
        self.assertEqual(self.candles.close_price, 0.0)
        self.assertEqual(self.candles.buy_volume, 0.0)
        self.assertEqual(self.candles.sell_volume, 0.0)
        self.assertEqual(self.candles.vwap_price, 0.0)
        self.assertEqual(self.candles.total_trades, 0.0)
        self.assertEqual(self.candles.open_timestamp, 0.0)
        self.assertEqual(self.candles.close_timestamp, 0.0)
        self.assertTrue(self.candles.ringbuffer.is_empty)
    
    def test_current_candle(self):
        self.candles.open_price = 104.0
        self.candles.high_price = 112.0
        self.candles.low_price = 91.0
        self.candles.close_price = 109.0
        self.candles.buy_volume = 10.0
        self.candles.sell_volume = 94.0
        self.candles.vwap_price = 106.0
        self.candles.total_trades = 9.0
        self.candles.open_timestamp = 1609459260.0
        self.candles.close_timestamp = 1609459320.0

        current_candle = self.candles.current_candle
        expected_candle = np.array(
            [
                104.0,
                112.0,
                91.0,
                109.0,
                10.0,
                94.0,
                106.0,
                9.0,
                1609459260.0,
                1609459320.0,
            ]
        )
        np.testing.assert_array_equal(current_candle, expected_candle)

    def test_getitem(self):
        self.candles.open_price = 100.0
        self.candles.high_price = 110.0
        self.candles.low_price = 90.0
        self.candles.close_price = 105.0
        self.candles.buy_volume = 500.0
        self.candles.sell_volume = 250.0
        self.candles.vwap_price = 102.0
        self.candles.total_trades = 3.0
        self.candles.open_timestamp = 1609459200.0
        self.candles.close_timestamp = 1609459260.0

        self.candles.insert_candle()

        newest_candle = self.candles[0]
        expected_candle = np.array(
            [
                100.0,
                110.0,
                90.0,
                105.0,
                500.0,
                250.0,
                102.0,
                3.0,
                1609459200.0,
                1609459260.0,
            ]
        )

        self.assertIsInstance(newest_candle, np.ndarray)
        np.testing.assert_array_equal(newest_candle, expected_candle)

    def test_length(self):
        for i in range(0, 30, 10):
            self.candles.open_price = 100.0
            self.candles.high_price = 110.0
            self.candles.low_price = 90.0
            self.candles.close_price = 105.0
            self.candles.buy_volume = 500.0
            self.candles.sell_volume = 250.0
            self.candles.vwap_price = 102.0
            self.candles.total_trades = 3.0
            self.candles.open_timestamp = 1609459200.0 + i
            self.candles.close_timestamp = 1609459260.0 + (i + 10)
            self.candles.insert_candle()

        self.assertEqual(len(self.candles), 3)

    def test_iteration(self):
        self.candles.open_price = 100.0
        self.candles.high_price = 110.0
        self.candles.low_price = 90.0
        self.candles.close_price = 105.0
        self.candles.buy_volume = 500.0
        self.candles.sell_volume = 200.0
        self.candles.vwap_price = 102.0
        self.candles.total_trades = 3.0
        self.candles.open_timestamp = 1609459200.0
        self.candles.close_timestamp = 1609459260.0

        self.candles.insert_candle()

        self.candles.open_price = 105.0
        self.candles.high_price = 120.0
        self.candles.low_price = 100.0
        self.candles.close_price = 110.0
        self.candles.buy_volume = 600.0
        self.candles.sell_volume = 300.0
        self.candles.vwap_price = 108.0
        self.candles.total_trades = 4.0
        self.candles.open_timestamp = 1609459260.0
        self.candles.close_timestamp = 1609459320.0

        self.candles.insert_candle()

        candles = list(self.candles)
        self.assertEqual(len(candles), 2)


if __name__ == "__main__":
    unittest.main()
