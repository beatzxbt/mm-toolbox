import unittest
import numpy as np
from mm_toolbox.src.ringbuffer import RingBufferMultiDim 


class TestRingBufferMultiDim(unittest.TestCase):
    def setUp(self):
        self.buffer_capacity = 5
        self.single_dim_buffer = RingBufferMultiDim(self.buffer_capacity, dtype=np.int64)
        self.multi_dim_buffer = RingBufferMultiDim((self.buffer_capacity, 3), dtype=np.int64)

    def test_initialization(self):
        self.assertEqual(self.single_dim_buffer.capacity, self.buffer_capacity)
        self.assertEqual(self.multi_dim_buffer.capacity, self.buffer_capacity)
        self.assertTrue(self.single_dim_buffer.is_empty)
        self.assertTrue(self.multi_dim_buffer.is_empty)

    def test_append_and_length(self):
        for i in range(self.buffer_capacity):
            self.single_dim_buffer.append(i)
            self.multi_dim_buffer.append(np.array([i, i+1, i+2]))

        self.assertEqual(len(self.single_dim_buffer), self.buffer_capacity)
        self.assertEqual(len(self.multi_dim_buffer), self.buffer_capacity)
        self.assertTrue(self.single_dim_buffer.is_full)
        self.assertTrue(self.multi_dim_buffer.is_full)

    def test_overwrite_on_full_buffer(self):
        for i in range(self.buffer_capacity):
            self.single_dim_buffer.append(i)
            self.multi_dim_buffer.append(np.array([i, i+1, i+2]))

        # Append additional elements to check overwriting behavior
        self.single_dim_buffer.append(10)
        self.multi_dim_buffer.append(np.array([10, 11, 12]))

        self.assertEqual(len(self.single_dim_buffer), self.buffer_capacity)
        self.assertEqual(len(self.multi_dim_buffer), self.buffer_capacity)

        # Check if the first element was overwritten
        self.assertEqual(self.single_dim_buffer[0], 1)
        np.testing.assert_array_equal(self.multi_dim_buffer[0], np.array([1, 2, 3]))

    def test_popright(self):
        for i in range(self.buffer_capacity):
            self.single_dim_buffer.append(i)
            self.multi_dim_buffer.append(np.array([i, i+1, i+2]))

        # Pop from right and check
        self.assertEqual(self.single_dim_buffer.popright(), 4)
        np.testing.assert_array_equal(self.multi_dim_buffer.popright(), np.array([4, 5, 6]))

        # Check length after pop
        self.assertEqual(len(self.single_dim_buffer), self.buffer_capacity - 1)
        self.assertEqual(len(self.multi_dim_buffer), self.buffer_capacity - 1)

    def test_popleft(self):
        for i in range(self.buffer_capacity):
            self.single_dim_buffer.append(i)
            self.multi_dim_buffer.append(np.array([i, i+1, i+2]))

        # Pop from left and check
        self.assertEqual(self.single_dim_buffer.popleft(), 0)
        np.testing.assert_array_equal(self.multi_dim_buffer.popleft(), np.array([0, 1, 2]))

        # Check length after pop
        self.assertEqual(len(self.single_dim_buffer), self.buffer_capacity - 1)
        self.assertEqual(len(self.multi_dim_buffer), self.buffer_capacity - 1)

    def test_contains(self):
        for i in range(self.buffer_capacity):
            self.single_dim_buffer.append(i)
            self.multi_dim_buffer.append(np.array([i, i+1, i+2]))

        self.assertIn(3, self.single_dim_buffer)
        self.assertNotIn(10, self.single_dim_buffer)

        self.assertIn(np.array([2, 3, 4]), self.multi_dim_buffer)
        self.assertNotIn(np.array([10, 11, 12]), self.multi_dim_buffer)

    def test_invalid_append(self):
        with self.assertRaises(TypeError):
            self.single_dim_buffer.append("invalid_type")

        with self.assertRaises(TypeError):
            self.multi_dim_buffer.append("invalid_type")

    def test_invalid_pop(self):
        empty_buffer = RingBufferMultiDim(3)
        with self.assertRaises(ValueError):
            empty_buffer.popright()

        with self.assertRaises(ValueError):
            empty_buffer.popleft()

    def test_equality(self):
        buffer1 = RingBufferMultiDim(3, dtype=np.int64)
        buffer2 = RingBufferMultiDim(3, dtype=np.int64)

        for i in range(3):
            buffer1.append(i)
            buffer2.append(i)

        self.assertEqual(buffer1, buffer2)

        buffer2.append(4)
        self.assertNotEqual(buffer1, buffer2)

    def test_as_array(self):
        for i in range(self.buffer_capacity):
            self.single_dim_buffer.append(i)
            self.multi_dim_buffer.append(np.array([i, i+1, i+2]))

        single_array = self.single_dim_buffer.as_array()
        multi_array = self.multi_dim_buffer.as_array()

        expected_single_array = np.array([0, 1, 2, 3, 4])
        expected_multi_array = np.array([[0, 1, 2],
                                         [1, 2, 3],
                                         [2, 3, 4],
                                         [3, 4, 5],
                                         [4, 5, 6]])

        np.testing.assert_array_equal(single_array, expected_single_array)
        np.testing.assert_array_equal(multi_array, expected_multi_array)

    def test_str_representation(self):
        buffer = RingBufferMultiDim(3)
        buffer.append(1.0)
        buffer.append(2.0)

        buffer_str = str(buffer)
        self.assertIn("RingBufferMultiDim", buffer_str)
        self.assertIn("capacity=3", buffer_str)
        self.assertIn("float64", buffer_str)
        self.assertIn("current_length=2", buffer_str)

    def test_property_checks(self):
        self.assertTrue(self.single_dim_buffer.is_empty)
        self.assertFalse(self.single_dim_buffer.is_full)

        for i in range(self.buffer_capacity):
            self.single_dim_buffer.append(i)

        self.assertFalse(self.single_dim_buffer.is_empty)
        self.assertTrue(self.single_dim_buffer.is_full)

    def test_shape_property(self):
        self.assertEqual(self.single_dim_buffer.shape, (0,))
        self.assertEqual(self.multi_dim_buffer.shape, (0, 3))

        for i in range(self.buffer_capacity):
            self.single_dim_buffer.append(i)
            self.multi_dim_buffer.append(np.array([i, i+1, i+2]))

        self.assertEqual(self.single_dim_buffer.shape, (self.buffer_capacity,))
        self.assertEqual(self.multi_dim_buffer.shape, (self.buffer_capacity, 3))

    def test_getitem(self):
        for i in range(self.buffer_capacity):
            self.single_dim_buffer.append(i)
            self.multi_dim_buffer.append(np.array([i, i+1, i+2]))

        self.assertEqual(self.single_dim_buffer[0], 0)
        np.testing.assert_array_equal(self.multi_dim_buffer[0], np.array([0, 1, 2]))

        with self.assertRaises(IndexError):
            _ = self.single_dim_buffer[self.buffer_capacity]

    def test_overwrite_behavior(self):
        for i in range(2 * self.buffer_capacity):
            self.single_dim_buffer.append(i)
            self.multi_dim_buffer.append(np.array([i, i+1, i+2]))

        self.assertEqual(len(self.single_dim_buffer), self.buffer_capacity)
        self.assertEqual(len(self.multi_dim_buffer), self.buffer_capacity)

        expected_single = np.array([5, 6, 7, 8, 9])
        expected_multi = np.array([[5, 6, 7],
                                   [6, 7, 8],
                                   [7, 8, 9],
                                   [8, 9, 10],
                                   [9, 10, 11]])

        np.testing.assert_array_equal(self.single_dim_buffer.as_array(), expected_single)
        np.testing.assert_array_equal(self.multi_dim_buffer.as_array(), expected_multi)

if __name__ == '__main__':
    unittest.main()
