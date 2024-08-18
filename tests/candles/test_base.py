import unittest
import numpy as np
from unittest.mock import MagicMock
from mm_toolbox.src.candles.base import BaseCandles


class TestBaseCandles(unittest.TestCase):
    def setUp(self):
        # Mock the abstract method, not tested here.
        BaseCandles.process_trade = MagicMock()
        self.candles = BaseCandles(num_candles=5)

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
            close_timestamp=1609459260.0  
        )
        candle_array = self.candles.as_array()
        expected_candles = np.array([[100.0, 110.0, 90.0, 105.0, 500.0, 200.0, 102.0, 3.0, 1609459200.0, 1609459260.0]])
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
            close_timestamp=1609459260.0  
        )

        self.candles.open_price = 104.0, 
        self.candles.high_price = 112.0, 
        self.candles.low_price = 91.0, 
        self.candles.close_price = 109.0, 
        self.candles.buy_volume = 10.0, 
        self.candles.sell_volume = 94.0,
        self.candles.vwap_price = 106.0,
        self.candles.total_trades = 9.0,
        self.candles.open_timestamp = 1609459260.0,  
        self.candles.close_timestamp = 1609459320.0  
        
        candle_array = self.candles.as_array()
        expected_candles = np.array([
            [100.0, 110.0, 90.0, 105.0, 500.0, 200.0, 102.0, 3.0, 1609459200.0, 1609459260.0],
            [104.0, 112.0, 91.0, 109.0, 10.0, 94.0, 106.0, 9.0, 1609459260.0, 1609459320.0]
        ])
        np.testing.assert_array_equal(candle_array, expected_candles)

    def test_as_array_empty(self):
        candle_array = self.candles.as_array()
        expected_candles = np.array([[]], dtype=np.float64)
        np.testing.assert_array_equal(candle_array, expected_candles)

    def test_reset_current_candle(self):
        self.candles.open_price = 104.0, 
        self.candles.high_price = 112.0, 
        self.candles.low_price = 91.0, 
        self.candles.close_price = 109.0, 
        self.candles.buy_volume = 10.0, 
        self.candles.sell_volume = 94.0,
        self.candles.vwap_price = 106.0,
        self.candles.total_trades = 9.0,
        self.candles.open_timestamp = 1609459260.0,  
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
            close_timestamp=1609459260.0  
        )
        stored_candle = self.candles[0]
        expected_candle = np.array([100.0, 110.0, 90.0, 105.0, 500.0, 200.0, 102.0, 3.0, 1609459200.0, 1609459260.0])
        np.testing.assert_array_equal(stored_candle, expected_candle)
    
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
            close_timestamp=1609459260.0
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
            close_timestamp=1609459260.0
        )
        imbalances = self.candles.imbalances()
        self.assertEqual(imbalances[0], 500.0 / 250.0)

    def test_rsi(self):
        close_prices = np.array([10, 11, 12, 13, 12, 11, 10, 9, 8, 7])
        self.candles.initialize(np.column_stack((np.arange(10), np.zeros(10), close_prices, np.ones(10))))
        rsi_values = self.candles.rsi(period=3)
        
        # Manually calculated RSI values for the above prices
        expected_rsi_values = np.array([70.0, 66.67, 50.0, 33.33, 30.0, 30.0, 50.0])
        np.testing.assert_almost_equal(rsi_values[-7:], expected_rsi_values, decimal=2)
    
    def test_bollinger_bands(self):
        close_prices = np.array([10, 11, 12, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4])
        self.candles.initialize(np.column_stack((np.arange(13), np.zeros(13), close_prices, np.ones(13))))
        lower_band, sma, upper_band = self.candles.calculate_bollinger_bands(period=5, num_std_dev=2.0)
        
        # Manually calculated Bollinger Bands for the above prices
        expected_sma = np.array([12, 11.4, 10.6, 9.8, 9, 8, 7])
        expected_lower_band = np.array([10.378, 9.342, 8.342, 7.488, 6.724, 5.988, 5.342])
        expected_upper_band = np.array([13.622, 13.458, 12.858, 12.112, 11.276, 10.012, 8.658])
        
        np.testing.assert_almost_equal(sma[-7:], expected_sma, decimal=3)
        np.testing.assert_almost_equal(lower_band[-7:], expected_lower_band, decimal=3)
        np.testing.assert_almost_equal(upper_band[-7:], expected_upper_band, decimal=3)
    
    def test_average_true_range(self):
        self.candles.insert_candle(100.0, 110.0, 90.0, 105.0, 500.0, 200.0, 102.0, 3, 1609459200.0, 1609459260.0)
        self.candles.insert_candle(105.0, 120.0, 100.0, 110.0, 600.0, 300.0, 108.0, 4, 1609459260.0, 1609459320.0)
        self.candles.insert_candle(110.0, 130.0, 105.0, 115.0, 700.0, 400.0, 114.0, 5, 1609459320.0, 1609459380.0)
        
        atr = self.candles.average_true_range()
        expected_atr = np.array([25.0])
        np.testing.assert_almost_equal(atr[-1:], expected_atr, decimal=1)

    def test_current_candle(self):
        self.candles.open_price = 104.0, 
        self.candles.high_price = 112.0, 
        self.candles.low_price = 91.0, 
        self.candles.close_price = 109.0, 
        self.candles.buy_volume = 10.0, 
        self.candles.sell_volume = 94.0,
        self.candles.vwap_price = 106.0,
        self.candles.total_trades = 9.0,
        self.candles.open_timestamp = 1609459260.0,  
        self.candles.close_timestamp = 1609459320.0  

        current_candle = self.candles.current_candle
        expected_candle = np.array([104.0, 112.0, 91.0, 109.0, 10.0, 94.0, 106.0, 9.0, 1609459260.0, 1609459320.0])
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
            close_timestamp=1609459260.0
        )

        newest_candle = self.candles[0]
        expected_candle = np.array([100.0, 110.0, 90.0, 105.0, 500.0, 250.0, 102.0, 3.0, 1609459200.0, 1609459260.0])
        
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
                close_timestamp=1609459260.0 + (i + 10)
            )

        self.assertEqual(len(self.candles), 4)
        
    def test_iteration(self):
        self.candles.insert_candle(100.0, 110.0, 90.0, 105.0, 500.0, 200.0, 102.0, 3, 1609459200.0, 1609459260.0)
        self.candles.insert_candle(105.0, 120.0, 100.0, 110.0, 600.0, 300.0, 108.0, 4, 1609459260.0, 1609459320.0)
        candles = list(self.candles)
        self.assertEqual(len(candles), 2)

if __name__ == '__main__':
    unittest.main()