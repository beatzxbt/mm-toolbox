import unittest
import numpy as np

from mm_toolbox.ringbuffer.multi import RingBufferMulti


class TestRingBufferMulti(unittest.TestCase):
    def test_initialization_1d(self):
        rb = RingBufferMulti(5, dtype=np.float64)
        self.assertEqual(len(rb), 0)
        self.assertTrue(rb.is_empty())
        self.assertFalse(rb.is_full())

        # Confirm shape and dtype
        raw = rb.unsafe_raw()
        self.assertEqual(raw.shape, (5,))
        self.assertEqual(raw.dtype, np.float64)

    def test_initialization_2d(self):
        rb = RingBufferMulti((4, 3), dtype=np.int32)
        self.assertEqual(len(rb), 0)
        self.assertTrue(rb.is_empty())
        self.assertFalse(rb.is_full())

        raw = rb.unsafe_raw()
        self.assertEqual(raw.shape, (4, 3))
        self.assertEqual(raw.dtype, np.int32)

    def test_append_1d(self):
        rb = RingBufferMulti(3, dtype=np.float64)
        rb.append(1.0)
        rb.append(2.0)
        self.assertEqual(len(rb), 2)
        np.testing.assert_array_equal(rb.unwrapped(), np.array([1.0, 2.0]))
        rb.append(3.0)
        self.assertTrue(rb.is_full())

        # Overwrite oldest
        rb.append(4.0)
        # Now should have [2.0, 3.0, 4.0]
        np.testing.assert_array_equal(rb.unwrapped(), np.array([2.0, 3.0, 4.0]))

    def test_append_2d(self):
        rb = RingBufferMulti((3, 2), dtype=np.float64)
        rb.append(np.array([1.0, 2.0]))
        rb.append([3.0, 4.0])  # lists should also be castable
        self.assertEqual(len(rb), 2)
        expected = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_array_equal(rb.unwrapped(), expected)

        # Fill up
        rb.append(np.array([5.0, 6.0]))
        self.assertTrue(rb.is_full())

        # Overwrite
        rb.append(np.array([7.0, 8.0]))
        # Now [3.0,4.0],[5.0,6.0],[7.0,8.0]
        expected = np.array([[3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        np.testing.assert_array_equal(rb.unwrapped(), expected)

    def test_unsafe_write_and_push(self):
        rb = RingBufferMulti(3, dtype=np.int32)
        # Direct write without push
        rb.unsafe_write(10)
        # Not counted yet
        self.assertEqual(len(rb), 0)
        rb.unsafe_push()
        self.assertEqual(len(rb), 1)
        np.testing.assert_array_equal(rb.unwrapped(), np.array([10], dtype=np.int32))

        # unsafe_write/unsafe_push with overwrite
        rb.unsafe_write(20)
        rb.unsafe_push()
        rb.unsafe_write(30)
        rb.unsafe_push()
        self.assertTrue(rb.is_full())
        # Now full: [10,20,30]
        # Overwrite oldest
        rb.unsafe_write(40)
        rb.unsafe_push()
        # [20,30,40]
        np.testing.assert_array_equal(
            rb.unwrapped(), np.array([20, 30, 40], dtype=np.int32)
        )

    def test_popright_popleft_1d(self):
        rb = RingBufferMulti(3, dtype=np.float64)
        rb.append(1.0)
        rb.append(2.0)
        rb.append(3.0)

        val = rb.popright()
        self.assertEqual(val, 3.0)
        self.assertEqual(len(rb), 2)

        val = rb.popleft()
        self.assertEqual(val, 1.0)
        self.assertEqual(len(rb), 1)

        val = rb.popright()
        self.assertEqual(val, 2.0)
        self.assertTrue(rb.is_empty())

        with self.assertRaises(IndexError):
            rb.popright()
        with self.assertRaises(IndexError):
            rb.popleft()

    def test_popright_popleft_2d(self):
        rb = RingBufferMulti((3, 2), dtype=np.float64)
        rb.append([1.0, 2.0])
        rb.append([3.0, 4.0])
        rb.append([5.0, 6.0])

        val = rb.popright()
        np.testing.assert_array_equal(val, np.array([5.0, 6.0]))
        self.assertEqual(len(rb), 2)

        val = rb.popleft()
        np.testing.assert_array_equal(val, np.array([1.0, 2.0]))
        self.assertEqual(len(rb), 1)

        val = rb.popright()
        np.testing.assert_array_equal(val, np.array([3.0, 4.0]))
        self.assertTrue(rb.is_empty())

        with self.assertRaises(IndexError):
            rb.popright()
        with self.assertRaises(IndexError):
            rb.popleft()

    def test_reset_and_fast_reset(self):
        rb = RingBufferMulti(4, dtype=np.int64)
        rb.append(10)
        rb.append(20)
        rb.append(30)

        old_data = rb.reset()
        np.testing.assert_array_equal(old_data, np.array([10, 20, 30]))
        self.assertTrue(rb.is_empty())
        self.assertEqual(len(rb), 0)

        rb.append(100)
        rb.fast_reset()
        self.assertTrue(rb.is_empty())
        self.assertEqual(len(rb), 0)

    def test_unwrapped_behavior(self):
        rb = RingBufferMulti(5, dtype=np.float64)
        for i in range(5):
            rb.append(float(i))
        # unwrapped: [0,1,2,3,4]
        np.testing.assert_array_equal(
            rb.unwrapped(), np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        )

        # Overwrite one
        rb.append(5.0)
        # now [1,2,3,4,5]
        np.testing.assert_array_equal(
            rb.unwrapped(), np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        )

    def test_raw_and_unsafe_raw(self):
        rb = RingBufferMulti((3, 2), dtype=np.float32)
        rb.append([1.0, 2.0])
        rb.append([3.0, 4.0])

        raw_copy = rb.raw()
        self.assertEqual(raw_copy.shape, (3, 2))
        self.assertEqual(raw_copy.dtype, np.float32)
        np.testing.assert_array_equal(
            raw_copy[0], np.array([1.0, 2.0], dtype=np.float32)
        )
        np.testing.assert_array_equal(
            raw_copy[1], np.array([3.0, 4.0], dtype=np.float32)
        )

        # unsafe_raw is a view
        unsafe = rb.unsafe_raw()
        unsafe[0, 0] = 999.0
        # internal state changes
        self.assertEqual(rb.unwrapped()[0, 0], 999.0)

    def test_is_full_and_empty_states(self):
        rb = RingBufferMulti(2, dtype=np.int16)
        self.assertTrue(rb.is_empty())
        self.assertFalse(rb.is_full())

        rb.append(1)
        self.assertFalse(rb.is_empty())
        self.assertFalse(rb.is_full())

        rb.append(2)
        self.assertFalse(rb.is_empty())
        self.assertTrue(rb.is_full())

        rb.popright()
        self.assertFalse(rb.is_full())
        self.assertFalse(rb.is_empty())

    def test_contains(self):
        rb = RingBufferMulti((3, 2), dtype=np.float64)
        rb.append([10.0, 20.0])
        rb.append([30.0, 40.0])

        self.assertIn([10.0, 20.0], rb)
        self.assertNotIn([50.0, 60.0], rb)
        rb.append([50.0, 60.0])
        self.assertIn([50.0, 60.0], rb)

        # Test scalar on a 2D buffer should fail
        self.assertNotIn(10.0, rb)

        # For a 1D buffer
        rb1d = RingBufferMulti(3, dtype=np.int32)
        rb1d.append(10)
        rb1d.append(20)
        self.assertIn(20, rb1d)
        self.assertNotIn(30, rb1d)

    def test_append_type_casting(self):
        rb = RingBufferMulti((2, 2), dtype=np.int32)
        # Append float array to int32 buffer should cast
        rb.append([1.1, 2.9])
        np.testing.assert_array_equal(
            rb.unwrapped(), np.array([[1, 2]], dtype=np.int32)
        )

        # Append scalar to 2D buffer with wrong shape should fail
        with self.assertRaises(TypeError):
            rb.append(99)  # expects shape (2,), got scalar

        rb.append([3, 4])
        self.assertTrue(rb.is_full())
        # [1,2],[3,4]
        # Overwrite with correct shape
        rb.append([5.5, 6.6])
        # now [3,4],[5,6]
        np.testing.assert_array_equal(
            rb.unwrapped(), np.array([[3, 4], [5, 6]], dtype=np.int32)
        )

    def test_append_wrong_type(self):
        rb = RingBufferMulti(2, dtype=np.float64)
        # Trying to append a string should fail
        with self.assertRaises(TypeError):
            rb.append("string")


if __name__ == "__main__":
    unittest.main()
