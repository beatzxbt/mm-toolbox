import unittest
import numpy as np
import time

from mm_toolbox.moving_average import TimeExponentialMovingAverage as TEMA


class TestTEMA(unittest.TestCase):
    def setUp(self):
        self.window = 5
        self.half_life = 10.0
        self.data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.tema = TEMA(self.window, fast=False, half_life_s=self.half_life)
        self.fast_tema = TEMA(self.window, fast=True, half_life_s=self.half_life)

    def test_initialize(self):
        result = self.tema.initialize(self.data)

        # The result should be the last value after processing all inputs
        self.assertIsNotNone(result)
        self.assertEqual(len(self.tema), self.window)
        self.assertEqual(self.tema.get_value(), result)

    def test_initialize_fast(self):
        result = self.fast_tema.initialize(self.data)

        self.assertIsNotNone(result)
        
        # Should raise error when trying to access values in fast mode
        with self.assertRaises(ValueError):
            len(self.fast_tema)

    def test_next(self):
        self.tema.initialize(self.data)
        initial_value = self.tema.get_value()
        next_value = self.tema.next(6.0)

        # next() doesn't update internal state
        self.assertIsNotNone(next_value)
        self.assertEqual(self.tema.get_value(), initial_value)

    def test_update(self):
        self.tema.initialize(self.data)
        initial_value = self.tema.get_value()
        update_value = self.tema.update(6.0)

        self.assertIsNotNone(update_value)
        self.assertEqual(self.tema.get_value(), update_value)
        
        # Check that values are stored in the buffer
        values = self.tema.get_values()
        self.assertEqual(len(values), self.window)
        self.assertEqual(values[-1], update_value)

    def test_update_fast(self):
        self.fast_tema.initialize(self.data)
        initial_value = self.fast_tema.get_value()
        update_value = self.fast_tema.update(6.0)

        self.assertIsNotNone(update_value)
        self.assertEqual(self.fast_tema.get_value(), update_value)
        
        # Should raise error when trying to access values in fast mode
        with self.assertRaises(ValueError):
            self.fast_tema.get_values()

    def test_time_dependency(self):
        # Test that the algorithm is time-dependent
        self.tema.initialize(self.data)
        initial_value = self.tema.get_value()
        
        # First update immediately
        first_update = self.tema.update(6.0)
        
        # Wait a short time and update with same value
        time.sleep(0.1)
        second_update = self.tema.update(6.0)
        
        # The second update should be different due to time factor
        self.assertNotEqual(first_update, second_update)

    def test_invalid_window(self):
        with self.assertRaises(ValueError):
            TEMA(1)

    def test_invalid_half_life(self):
        with self.assertRaises(ValueError):
            TEMA(5, half_life_s=-1.0)


if __name__ == "__main__":
    unittest.main()

