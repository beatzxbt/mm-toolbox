import unittest
import numpy as np
from src.mm_toolbox.candles.base import BaseCandles


class MockBaseCandle(BaseCandles):
    def __init__(self, num_candles):
        super().__init__(num_candles)

    def process_trade(
        self, timestamp: float, side: bool, price: float, size: float
    ) -> None:
        # This is tested in actual inherited classes for specific behaviour
        pass


class TestBaseCandles(unittest.TestCase):
    def setUp(self):
        self.candles = MockBaseCandle(num_candles=5)

    def test_class_initialization(self):
        self.assertEqual(self.candles.num_candles, 5)
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

    def test_insert_candle(self):
        self.candles.insert_candle(
            open_price=100.0,
            high_price=110.0,
            low_price=90.0,
            close_price=105.0,
            buy_volume=500.0,
            sell_volume=200.0,
            vwap_price=102.0,
            total_trades=3.0,
            open_timestamp=1609459200.0,
            close_timestamp=1609459260.0,
        )
        stored_candle = self.candles[0]
        expected_candle = np.array(
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
        )
        np.testing.assert_array_equal(stored_candle, expected_candle)

    def test_as_array_without_current(self):
        self.candles.insert_candle(
            open_price=100.0,
            high_price=110.0,
            low_price=90.0,
            close_price=105.0,
            buy_volume=500.0,
            sell_volume=200.0,
            vwap_price=102.0,
            total_trades=3.0,
            open_timestamp=1609459200.0,
            close_timestamp=1609459260.0,
        )
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
        self.candles.insert_candle(
            open_price=100.0,
            high_price=110.0,
            low_price=90.0,
            close_price=105.0,
            buy_volume=500.0,
            sell_volume=200.0,
            vwap_price=102.0,
            total_trades=3.0,
            open_timestamp=1609459200.0,
            close_timestamp=1609459260.0,
        )

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

    def test_as_array_empty(self):
        candle_array = self.candles.as_array()
        expected_candles = np.array([[]], dtype=np.float64)
        np.testing.assert_array_equal(candle_array, expected_candles)

    def test_reset_current_candle(self):
        self.candles.open_price = (104.0,)
        self.candles.high_price = (112.0,)
        self.candles.low_price = (91.0,)
        self.candles.close_price = (109.0,)
        self.candles.buy_volume = (10.0,)
        self.candles.sell_volume = (94.0,)
        self.candles.vwap_price = (106.0,)
        self.candles.total_trades = (9.0,)
        self.candles.open_timestamp = (1609459260.0,)
        self.candles.close_timestamp = 1609459320.0

        self.candles.reset_current_candle()

        self.assertEqual(self.candles.num_candles, 5)
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

    def test_durations(self):
        self.candles.insert_candle(
            open_price=100.0,
            high_price=110.0,
            low_price=90.0,
            close_price=105.0,
            buy_volume=500.0,
            sell_volume=200.0,
            vwap_price=102.0,
            total_trades=3.0,
            open_timestamp=1609459200.0,
            close_timestamp=1609459260.0,
        )
        durations = self.candles.durations()
        self.assertEqual(durations[0], 60.0)

    def test_imbalances(self):
        self.candles.insert_candle(
            open_price=100.0,
            high_price=110.0,
            low_price=90.0,
            close_price=105.0,
            buy_volume=500.0,
            sell_volume=250.0,
            vwap_price=102.0,
            total_trades=3.0,
            open_timestamp=1609459200.0,
            close_timestamp=1609459260.0,
        )
        imbalances = self.candles.imbalances()
        self.assertEqual(imbalances[0], 500.0 / 250.0)

    def test_rsi(self):
        candles = MockBaseCandle(num_candles=15)

        close_prices = np.array(
            [104, 103, 107, 109, 108, 106, 103, 101, 99, 98, 97, 96, 95],
            dtype=np.float64,
        )

        for i in range(len(close_prices)):
            candles.insert_candle(
                open_price=0.0,  # Irrelevant to RSI calculation
                high_price=0.0,  # Irrelevant to RSI calculation
                low_price=0.0,  # Irrelevant to RSI calculation
                close_price=close_prices[i],
                buy_volume=0.0,  # Irrelevant to RSI calculation
                sell_volume=0.0,  # Irrelevant to RSI calculation
                vwap_price=0.0,  # Irrelevant to RSI calculation
                total_trades=0.0,  # Irrelevant to RSI calculation
                open_timestamp=1609459200.0 + (60 * i),
                close_timestamp=1609459260.0 + (60 * i),
            )

        rsi_values = candles.rsi(period=5)
        expected_rsi_values = np.array(
            [
                0.0,
                80.0,
                85.714,
                75.0,
                54.545,
                44.444,
                36.09,
                32.296,
                28.545,
                24.926,
                21.516,
                50.0,
            ]
        )

        np.testing.assert_almost_equal(rsi_values, expected_rsi_values, decimal=3)

    def test_bollinger_bands(self):
        candles = MockBaseCandle(num_candles=13)

        open_prices = np.array(
            [100, 102, 104, 106, 105, 103, 101, 100, 98, 97, 96, 95, 94],
            dtype=np.float64,
        )
        high_prices = np.array(
            [105, 106, 108, 110, 109, 107, 104, 103, 101, 100, 99, 98, 97],
            dtype=np.float64,
        )
        low_prices = np.array(
            [99, 100, 102, 104, 103, 101, 99, 98, 96, 95, 94, 93, 92], dtype=np.float64
        )
        close_prices = np.array(
            [104, 105, 107, 109, 108, 106, 103, 101, 99, 98, 97, 96, 95],
            dtype=np.float64,
        )

        for i in range(len(close_prices)):
            candles.insert_candle(
                open_price=open_prices[i],
                high_price=high_prices[i],
                low_price=low_prices[i],
                close_price=close_prices[i],
                buy_volume=0.0,  # Irrelevant to Bollinger Bands calculation
                sell_volume=0.0,  # Irrelevant to Bollinger Bands calculation
                vwap_price=0.0,  # Irrelevant to Bollinger Bands calculation
                total_trades=0.0,  # Irrelevant to Bollinger Bands calculation
                open_timestamp=1609459200.0 + (60 * i),
                close_timestamp=1609459260.0 + (60 * i),
            )

        lower_band, sma, upper_band = candles.bollinger_bands(period=5, num_std_dev=2.0)

        expected_sma = np.array([106.6, 105.4, 103.4, 101.4, 99.6, 98.2, 97.0])
        expected_lower_band = np.array(
            [102.482, 99.387, 96.876, 95.659, 95.292, 94.759, 94.172]
        )
        expected_upper_band = np.array(
            [110.718, 111.413, 109.924, 107.141, 103.908, 101.641, 99.828]
        )

        np.testing.assert_almost_equal(sma[-7:], expected_sma, decimal=3)
        np.testing.assert_almost_equal(lower_band[-7:], expected_lower_band, decimal=3)
        np.testing.assert_almost_equal(upper_band[-7:], expected_upper_band, decimal=3)

    def test_average_true_range(self):
        self.candles.insert_candle(
            100.0,
            110.0,
            90.0,
            105.0,
            500.0,
            200.0,
            102.0,
            3,
            1609459200.0,
            1609459260.0,
        )
        self.candles.insert_candle(
            105.0,
            120.0,
            100.0,
            110.0,
            600.0,
            300.0,
            108.0,
            4,
            1609459260.0,
            1609459320.0,
        )
        self.candles.insert_candle(
            110.0,
            130.0,
            105.0,
            115.0,
            700.0,
            400.0,
            114.0,
            5,
            1609459320.0,
            1609459380.0,
        )

        atr = self.candles.average_true_range()
        expected_atr = np.array([25.0])
        np.testing.assert_almost_equal(atr[-1:], expected_atr, decimal=1)

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
        self.candles.insert_candle(
            open_price=100.0,
            high_price=110.0,
            low_price=90.0,
            close_price=105.0,
            buy_volume=500.0,
            sell_volume=250.0,
            vwap_price=102.0,
            total_trades=3.0,
            open_timestamp=1609459200.0,
            close_timestamp=1609459260.0,
        )

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
            self.candles.insert_candle(
                open_price=100.0,
                high_price=110.0,
                low_price=90.0,
                close_price=105.0,
                buy_volume=500.0,
                sell_volume=250.0,
                vwap_price=102.0,
                total_trades=3.0,
                open_timestamp=1609459200.0 + i,
                close_timestamp=1609459260.0 + (i + 10),
            )

        self.assertEqual(len(self.candles), 3)

    def test_iteration(self):
        self.candles.insert_candle(
            100.0,
            110.0,
            90.0,
            105.0,
            500.0,
            200.0,
            102.0,
            3,
            1609459200.0,
            1609459260.0,
        )
        self.candles.insert_candle(
            105.0,
            120.0,
            100.0,
            110.0,
            600.0,
            300.0,
            108.0,
            4,
            1609459260.0,
            1609459320.0,
        )
        candles = list(self.candles)
        self.assertEqual(len(candles), 2)


if __name__ == "__main__":
    unittest.main()
