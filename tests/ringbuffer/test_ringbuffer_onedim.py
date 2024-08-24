import unittest
import numpy as np
from mm_toolbox.src.ringbuffer import RingBufferSingleDimFloat, RingBufferSingleDimInt


class TestRingBufferSingleDimFloat(unittest.TestCase):
    def setUp(self):
        self.buffer_capacity = 5
        self.buffer = RingBufferSingleDimFloat(self.buffer_capacity)

    def test_initialization(self):
        self.assertEqual(self.buffer.capacity, self.buffer_capacity)
        self.assertTrue(self.buffer.is_empty)

    def test_append_and_length(self):
        for i in range(self.buffer_capacity):
            self.buffer.append(float(i))

        self.assertEqual(len(self.buffer), self.buffer_capacity)
        self.assertTrue(self.buffer.is_full)

    def test_overwrite_on_full_buffer(self):
        for i in range(self.buffer_capacity):
            self.buffer.append(float(i))

        # Append an additional element to check overwriting behavior
        self.buffer.append(10.0)

        self.assertEqual(len(self.buffer), self.buffer_capacity)
        self.assertEqual(
            self.buffer[0], 1.0
        )  # The first element should have been overwritten

    def test_popright(self):
        for i in range(self.buffer_capacity):
            self.buffer.append(float(i))

        # Pop from right and check
        self.assertEqual(self.buffer.popright(), 4.0)

        # Check length after pop
        self.assertEqual(len(self.buffer), self.buffer_capacity - 1)

    def test_popleft(self):
        for i in range(self.buffer_capacity):
            self.buffer.append(float(i))

        # Pop from left and check
        self.assertEqual(self.buffer.popleft(), 0.0)

        # Check length after pop
        self.assertEqual(len(self.buffer), self.buffer_capacity - 1)

    def test_contains(self):
        for i in range(self.buffer_capacity):
            self.buffer.append(float(i))

        self.assertIn(3.0, self.buffer)
        self.assertNotIn(10.0, self.buffer)

    def test_invalid_append(self):
        # This test is obvious, but it's result is important as
        # this is intended unsafe behaviour.
        with self.assertRaises(Exception):
            self.buffer.append("invalid_type")

    def test_invalid_pop(self):
        empty_buffer = RingBufferSingleDimFloat(3)
        with self.assertRaises(AssertionError):
            empty_buffer.popright()

        with self.assertRaises(AssertionError):
            empty_buffer.popleft()

    def test_equality(self):
        buffer1 = RingBufferSingleDimFloat(3)
        buffer2 = RingBufferSingleDimFloat(3)

        for i in range(3):
            buffer1.append(float(i))
            buffer2.append(float(i))

        self.assertEqual(buffer1, buffer2)

        buffer2.append(4.0)
        self.assertNotEqual(buffer1, buffer2)

    def test_as_array(self):
        for i in range(self.buffer_capacity):
            self.buffer.append(float(i))

        array = self.buffer.as_array()
        expected_array = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_equal(array, expected_array)

    def test_str_representation(self):
        buffer = RingBufferSingleDimFloat(3)
        buffer.append(1.0)
        buffer.append(2.0)

        buffer_str = str(buffer)
        self.assertIn("RingBufferSingleDimFloat", buffer_str)
        self.assertIn("capacity=3", buffer_str)
        self.assertIn("float64", buffer_str)
        self.assertIn("current_length=2", buffer_str)

    def test_property_checks(self):
        self.assertTrue(self.buffer.is_empty)
        self.assertFalse(self.buffer.is_full)

        for i in range(self.buffer_capacity):
            self.buffer.append(float(i))

        self.assertFalse(self.buffer.is_empty)
        self.assertTrue(self.buffer.is_full)

    def test_shape_property(self):
        self.assertEqual(self.buffer.shape, (0,))

        for i in range(self.buffer_capacity):
            self.buffer.append(float(i))

        self.assertEqual(self.buffer.shape, (self.buffer_capacity,))

    def test_getitem(self):
        for i in range(self.buffer_capacity):
            self.buffer.append(float(i))

        self.assertEqual(self.buffer[0], 0.0)

    def test_overwrite_behavior(self):
        for i in range(2 * self.buffer_capacity):
            self.buffer.append(float(i))

        self.assertEqual(len(self.buffer), self.buffer_capacity)

        expected = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
        np.testing.assert_array_equal(self.buffer.as_array(), expected)


class TestRingBufferSingleDimInt(unittest.TestCase):
    def setUp(self):
        self.buffer_capacity = 5
        self.buffer = RingBufferSingleDimInt(self.buffer_capacity)

    def test_initialization(self):
        self.assertEqual(self.buffer.capacity, self.buffer_capacity)
        self.assertTrue(self.buffer.is_empty)

    def test_append_and_length(self):
        for i in range(self.buffer_capacity):
            self.buffer.append(i)

        self.assertEqual(len(self.buffer), self.buffer_capacity)
        self.assertTrue(self.buffer.is_full)

    def test_overwrite_on_full_buffer(self):
        for i in range(self.buffer_capacity):
            self.buffer.append(i)

        # Append an additional element to check overwriting behavior
        self.buffer.append(10)

        self.assertEqual(len(self.buffer), self.buffer_capacity)
        self.assertEqual(
            self.buffer[0], 1
        )  # The first element should have been overwritten

    def test_popright(self):
        for i in range(self.buffer_capacity):
            self.buffer.append(i)

        # Pop from right and check
        self.assertEqual(self.buffer.popright(), 4)

        # Check length after pop
        self.assertEqual(len(self.buffer), self.buffer_capacity - 1)

    def test_popleft(self):
        for i in range(self.buffer_capacity):
            self.buffer.append(i)

        # Pop from left and check
        self.assertEqual(self.buffer.popleft(), 0)

        # Check length after pop
        self.assertEqual(len(self.buffer), self.buffer_capacity - 1)

    def test_contains(self):
        for i in range(self.buffer_capacity):
            self.buffer.append(i)

        self.assertIn(3, self.buffer)
        self.assertNotIn(10, self.buffer)

    def test_invalid_append(self):
        # This test is obvious, but it's result is important as
        # this is intended unsafe behaviour.
        with self.assertRaises(Exception):
            self.buffer.append("invalid_type")

    def test_invalid_pop(self):
        empty_buffer = RingBufferSingleDimInt(3)
        with self.assertRaises(AssertionError):
            empty_buffer.popright()

        with self.assertRaises(AssertionError):
            empty_buffer.popleft()

    def test_equality(self):
        buffer1 = RingBufferSingleDimInt(3)
        buffer2 = RingBufferSingleDimInt(3)

        for i in range(3):
            buffer1.append(i)
            buffer2.append(i)

        self.assertEqual(buffer1, buffer2)

        buffer2.append(4)
        self.assertNotEqual(buffer1, buffer2)

    def test_as_array(self):
        for i in range(self.buffer_capacity):
            self.buffer.append(i)

        array = self.buffer.as_array()
        expected_array = np.array([0, 1, 2, 3, 4])
        np.testing.assert_array_equal(array, expected_array)

    def test_str_representation(self):
        buffer = RingBufferSingleDimInt(3)
        buffer.append(1)
        buffer.append(2)

        buffer_str = str(buffer)
        self.assertIn("RingBufferSingleDimInt", buffer_str)
        self.assertIn("capacity=3", buffer_str)
        self.assertIn("int64", buffer_str)
        self.assertIn("current_length=2", buffer_str)

    def test_property_checks(self):
        self.assertTrue(self.buffer.is_empty)
        self.assertFalse(self.buffer.is_full)

        for i in range(self.buffer_capacity):
            self.buffer.append(i)

        self.assertFalse(self.buffer.is_empty)
        self.assertTrue(self.buffer.is_full)

    def test_shape_property(self):
        self.assertEqual(self.buffer.shape, (0,))

        for i in range(self.buffer_capacity):
            self.buffer.append(i)

        self.assertEqual(self.buffer.shape, (self.buffer_capacity,))

    def test_getitem(self):
        for i in range(self.buffer_capacity):
            self.buffer.append(i)

        self.assertEqual(self.buffer[0], 0)

    def test_overwrite_behavior(self):
        for i in range(2 * self.buffer_capacity):
            self.buffer.append(i)

        self.assertEqual(len(self.buffer), self.buffer_capacity)

        expected = np.array([5, 6, 7, 8, 9])
        np.testing.assert_array_equal(self.buffer.as_array(), expected)


if __name__ == "__main__":
    unittest.main()
