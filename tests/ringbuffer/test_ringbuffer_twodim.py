import unittest
import numpy as np
from mm_toolbox.src.ringbuffer import RingBufferTwoDimFloat, RingBufferTwoDimInt

import unittest
import numpy as np


class TestRingBufferTwoDimFloat(unittest.TestCase):
    def setUp(self):
        self.buffer_capacity = 5
        self.sub_array_len = 3
        self.buffer = RingBufferTwoDimFloat(self.buffer_capacity, self.sub_array_len)

    def test_initialization(self):
        self.assertEqual(self.buffer.capacity, self.buffer_capacity)
        self.assertEqual(self.buffer.sub_array_len, self.sub_array_len)
        self.assertTrue(self.buffer.is_empty)

    def test_append_and_length(self):
        for i in range(self.buffer_capacity):
            self.buffer.append(np.array([i, i + 1, i + 2], dtype=np.float64))

        self.assertEqual(len(self.buffer), self.buffer_capacity)
        self.assertTrue(self.buffer.is_full)

    def test_overwrite_on_full_buffer(self):
        for i in range(self.buffer_capacity):
            self.buffer.append(np.array([i, i + 1, i + 2], dtype=np.float64))

        # Append an additional element to check overwriting behavior
        self.buffer.append(np.array([10, 11, 12], dtype=np.float64))

        self.assertEqual(len(self.buffer), self.buffer_capacity)
        np.testing.assert_array_equal(
            self.buffer[0], np.array([1, 2, 3], dtype=np.float64)
        )  # The first element should have been overwritten

    def test_pop(self):
        for i in range(self.buffer_capacity):
            self.buffer.append(np.array([i, i + 1, i + 2], dtype=np.float64))

        # Pop from right and check
        np.testing.assert_array_equal(
            self.buffer.pop(), np.array([4, 5, 6], dtype=np.float64)
        )

        # Check length after pop
        self.assertEqual(len(self.buffer), self.buffer_capacity - 1)

    def test_popleft(self):
        for i in range(self.buffer_capacity):
            self.buffer.append(np.array([i, i + 1, i + 2], dtype=np.float64))

        # Pop from left and check
        np.testing.assert_array_equal(
            self.buffer.popleft(), np.array([0, 1, 2], dtype=np.float64)
        )

        # Check length after pop
        self.assertEqual(len(self.buffer), self.buffer_capacity - 1)

    def test_contains(self):
        for i in range(self.buffer_capacity):
            self.buffer.append(np.array([i, i + 1, i + 2], dtype=np.float64))

        self.assertIn(np.array([2, 3, 4], dtype=np.float64), self.buffer)
        self.assertNotIn(np.array([10, 11, 12], dtype=np.float64), self.buffer)

    def test_invalid_append(self):
        with self.assertRaises(AssertionError):
            self.buffer.append(
                np.array([1, 2], dtype=np.float64)
            )  # Invalid sub_array_len

        with self.assertRaises(AssertionError):
            self.buffer.append(
                np.array([1, 2, 3, 4], dtype=np.float64)
            )  # Invalid sub_array_len

    def test_invalid_pop(self):
        empty_buffer = RingBufferTwoDimFloat(3, 2)
        with self.assertRaises(AssertionError):
            empty_buffer.pop()

        with self.assertRaises(AssertionError):
            empty_buffer.popleft()

    def test_equality(self):
        buffer1 = RingBufferTwoDimFloat(3, 2)
        buffer2 = RingBufferTwoDimFloat(3, 2)

        for i in range(3):
            buffer1.append(np.array([i, i + 1], dtype=np.float64))
            buffer2.append(np.array([i, i + 1], dtype=np.float64))

        self.assertEqual(buffer1, buffer2)

        buffer2.append(np.array([4, 5], dtype=np.float64))
        self.assertNotEqual(buffer1, buffer2)

    def test_as_array(self):
        for i in range(self.buffer_capacity):
            self.buffer.append(np.array([i, i + 1, i + 2], dtype=np.float64))

        array = self.buffer.as_array()
        expected_array = np.array(
            [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]], dtype=np.float64
        )

        np.testing.assert_array_equal(array, expected_array)

    def test_str_representation(self):
        self.buffer.append(np.array([1, 2, 3], dtype=np.float64))
        self.buffer.append(np.array([4, 5, 6], dtype=np.float64))

        buffer_str = str(self.buffer)
        self.assertIn("RingBufferTwoDimFloat", buffer_str)
        self.assertIn("capacity=5", buffer_str)
        self.assertIn("float64", buffer_str)
        self.assertIn("current_length=2", buffer_str)

    def test_property_checks(self):
        self.assertTrue(self.buffer.is_empty)
        self.assertFalse(self.buffer.is_full)

        for i in range(self.buffer_capacity):
            self.buffer.append(np.array([i, i + 1, i + 2], dtype=np.float64))

        self.assertFalse(self.buffer.is_empty)
        self.assertTrue(self.buffer.is_full)

    def test_shape_property(self):
        self.assertEqual(self.buffer.shape, (0, self.sub_array_len))

        for i in range(self.buffer_capacity):
            self.buffer.append(np.array([i, i + 1, i + 2], dtype=np.float64))

        self.assertEqual(self.buffer.shape, (self.buffer_capacity, self.sub_array_len))

    def test_getitem(self):
        for i in range(self.buffer_capacity):
            self.buffer.append(np.array([i, i + 1, i + 2], dtype=np.float64))

        np.testing.assert_array_equal(
            self.buffer[0], np.array([0, 1, 2], dtype=np.float64)
        )

    def test_overwrite_behavior(self):
        for i in range(2 * self.buffer_capacity):
            self.buffer.append(np.array([i, i + 1, i + 2], dtype=np.float64))

        self.assertEqual(len(self.buffer), self.buffer_capacity)

        expected = np.array(
            [[5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10], [9, 10, 11]], dtype=np.float64
        )

        np.testing.assert_array_equal(self.buffer.as_array(), expected)


class TestRingBufferTwoDimInt(unittest.TestCase):
    def setUp(self):
        self.buffer_capacity = 5
        self.sub_array_len = 3
        self.buffer = RingBufferTwoDimInt(self.buffer_capacity, self.sub_array_len)

    def test_initialization(self):
        self.assertEqual(self.buffer.capacity, self.buffer_capacity)
        self.assertEqual(self.buffer.sub_array_len, self.sub_array_len)
        self.assertTrue(self.buffer.is_empty)

    def test_append_and_length(self):
        for i in range(self.buffer_capacity):
            self.buffer.append(np.array([i, i + 1, i + 2]))

        self.assertEqual(len(self.buffer), self.buffer_capacity)
        self.assertTrue(self.buffer.is_full)

    def test_overwrite_on_full_buffer(self):
        for i in range(self.buffer_capacity):
            self.buffer.append(np.array([i, i + 1, i + 2]))

        # Append an additional element to check overwriting behavior
        self.buffer.append(np.array([10, 11, 12]))

        self.assertEqual(len(self.buffer), self.buffer_capacity)
        np.testing.assert_array_equal(
            self.buffer[0], np.array([1, 2, 3])
        )  # The first element should have been overwritten

    def test_pop(self):
        for i in range(self.buffer_capacity):
            self.buffer.append(np.array([i, i + 1, i + 2]))

        # Pop from right and check
        np.testing.assert_array_equal(self.buffer.pop(), np.array([4, 5, 6]))

        # Check length after pop
        self.assertEqual(len(self.buffer), self.buffer_capacity - 1)

    def test_popleft(self):
        for i in range(self.buffer_capacity):
            self.buffer.append(np.array([i, i + 1, i + 2]))

        # Pop from left and check
        np.testing.assert_array_equal(self.buffer.popleft(), np.array([0, 1, 2]))

        # Check length after pop
        self.assertEqual(len(self.buffer), self.buffer_capacity - 1)

    def test_contains(self):
        for i in range(self.buffer_capacity):
            self.buffer.append(np.array([i, i + 1, i + 2]))

        self.assertIn(np.array([2, 3, 4]), self.buffer)
        self.assertNotIn(np.array([10, 11, 12]), self.buffer)

    def test_invalid_append(self):
        with self.assertRaises(AssertionError):
            self.buffer.append(np.array([1, 2]))  # Invalid sub_array_len

        with self.assertRaises(AssertionError):
            self.buffer.append(np.array([1, 2, 3, 4]))  # Invalid sub_array_len

    def test_invalid_pop(self):
        empty_buffer = RingBufferTwoDimInt(3, 2)
        with self.assertRaises(AssertionError):
            empty_buffer.pop()

        with self.assertRaises(AssertionError):
            empty_buffer.popleft()

    def test_equality(self):
        buffer1 = RingBufferTwoDimInt(3, 2)
        buffer2 = RingBufferTwoDimInt(3, 2)

        for i in range(3):
            buffer1.append(np.array([i, i + 1]))
            buffer2.append(np.array([i, i + 1]))

        self.assertEqual(buffer1, buffer2)

        buffer2.append(np.array([4, 5]))
        self.assertNotEqual(buffer1, buffer2)

    def test_as_array(self):
        for i in range(self.buffer_capacity):
            self.buffer.append(np.array([i, i + 1, i + 2]))

        array = self.buffer.as_array()
        expected_array = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]])

        np.testing.assert_array_equal(array, expected_array)

    def test_str_representation(self):
        buffer = RingBufferTwoDimInt(3, 2)
        buffer.append(np.array([1, 2]))
        buffer.append(np.array([3, 4]))

        buffer_str = str(buffer)
        self.assertIn("RingBufferTwoDimInt", buffer_str)
        self.assertIn("capacity=3", buffer_str)
        self.assertIn("dtype=int64", buffer_str)
        self.assertIn("current_length=2", buffer_str)

    def test_property_checks(self):
        self.assertTrue(self.buffer.is_empty)
        self.assertFalse(self.buffer.is_full)

        for i in range(self.buffer_capacity):
            self.buffer.append(np.array([i, i + 1, i + 2]))

        self.assertFalse(self.buffer.is_empty)
        self.assertTrue(self.buffer.is_full)

    def test_shape_property(self):
        self.assertEqual(self.buffer.shape, (0, self.sub_array_len))

        for i in range(self.buffer_capacity):
            self.buffer.append(np.array([i, i + 1, i + 2]))

        self.assertEqual(self.buffer.shape, (self.buffer_capacity, self.sub_array_len))

    def test_getitem(self):
        for i in range(self.buffer_capacity):
            self.buffer.append(np.array([i, i + 1, i + 2]))

        np.testing.assert_array_equal(self.buffer[0], np.array([0, 1, 2]))

    def test_overwrite_behavior(self):
        for i in range(2 * self.buffer_capacity):
            self.buffer.append(np.array([i, i + 1, i + 2]))

        self.assertEqual(len(self.buffer), self.buffer_capacity)

        expected = np.array([[5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10], [9, 10, 11]])

        np.testing.assert_array_equal(self.buffer.as_array(), expected)


if __name__ == "__main__":
    unittest.main()
