import unittest
import numpy as np

from mm_toolbox.ringbuffer.onedim import RingBufferOneDim


class TestRingBufferOneDim(unittest.TestCase):
    def test_initialization(self):
        rb = RingBufferOneDim(5)
        self.assertEqual(len(rb), 0)
        self.assertTrue(rb.is_empty())
        self.assertFalse(rb.is_full())

        rb.append(1.0)
        self.assertEqual(len(rb), 1)

    def test_append_and_iter(self):
        rb = RingBufferOneDim(3)
        rb.append(10.0)
        rb.append(20.0)
        rb.append(30.0)

        self.assertEqual(len(rb), 3)
        self.assertTrue(rb.is_full())

        self.assertEqual(list(rb), [10.0, 20.0, 30.0])

        rb.append(40.0)
        self.assertEqual(list(rb), [20.0, 30.0, 40.0])
        self.assertEqual(len(rb), 3)

    def test_popright_and_popleft(self):
        rb = RingBufferOneDim(3)
        rb.append(1.0)
        rb.append(2.0)
        rb.append(3.0)

        val = rb.popright()
        self.assertEqual(val, 3.0)
        self.assertEqual(list(rb), [1.0, 2.0])

        val = rb.popleft()
        self.assertEqual(val, 1.0)
        self.assertEqual(list(rb), [2.0])

        val = rb.popright()
        self.assertEqual(val, 2.0)
        self.assertTrue(rb.is_empty())

        with self.assertRaises(IndexError):
            rb.popright()
        with self.assertRaises(IndexError):
            rb.popleft()

    def test_contains(self):
        rb = RingBufferOneDim(3)
        rb.append(5.5)
        rb.append(6.6)

        self.assertIn(5.5, rb)
        self.assertNotIn(7.7, rb)
        rb.append(7.7)
        self.assertIn(7.7, rb)

    def test_getitem(self):
        rb = RingBufferOneDim(5)
        for i in range(5):
            rb.append(float(i))

        # Buffer: [0.0, 1.0, 2.0, 3.0, 4.0]
        self.assertEqual(rb[0], 0.0)
        self.assertEqual(rb[4], 4.0)
        self.assertEqual(rb[-1], 4.0)
        self.assertEqual(rb[-5], 0.0)

        with self.assertRaises(IndexError):
            _ = rb[5]
        with self.assertRaises(IndexError):
            _ = rb[-6]

    def test_reset(self):
        rb = RingBufferOneDim(4)
        rb.append(10)
        rb.append(20)
        rb.append(30)

        old_data = rb.reset()
        np.testing.assert_array_equal(old_data, np.array([10.0, 20.0, 30.0]))
        self.assertTrue(rb.is_empty())
        self.assertEqual(len(rb), 0)

    def test_fast_reset(self):
        rb = RingBufferOneDim(4)
        rb.append(100.0)
        rb.append(200.0)

        rb.fast_reset()
        self.assertTrue(rb.is_empty())
        self.assertEqual(len(rb), 0)

    def test_unwrapped(self):
        rb = RingBufferOneDim(5)
        for i in range(5):
            rb.append(float(i))

        uw = rb.unwrapped()
        np.testing.assert_array_equal(uw, np.array([0, 1, 2, 3, 4], dtype=np.double))

        # Overwrite one
        rb.append(5.0)
        uw = rb.unwrapped()
        np.testing.assert_array_equal(uw, np.array([1, 2, 3, 4, 5], dtype=np.double))

    def test_raw_and_unsafe_raw(self):
        rb = RingBufferOneDim(3)
        rb.append(1.0)
        rb.append(2.0)

        raw_copy = rb.raw()
        self.assertEqual(raw_copy.size, 3)
        self.assertEqual(raw_copy[0], 1.0)
        self.assertEqual(raw_copy[1], 2.0)

        # unsafe_raw is a view, can modify internal state.
        unsafe = rb.unsafe_raw()
        unsafe[0] = 999.0
        self.assertEqual(rb[0], 999.0)

    def test_pop_errors(self):
        rb = RingBufferOneDim(2)
        with self.assertRaises(IndexError):
            rb.popright()

        with self.assertRaises(IndexError):
            rb.popleft()

    def test_is_full_and_empty_states(self):
        rb = RingBufferOneDim(2)
        self.assertTrue(rb.is_empty())
        self.assertFalse(rb.is_full())

        rb.append(1.0)
        self.assertFalse(rb.is_empty())
        self.assertFalse(rb.is_full())

        rb.append(2.0)
        self.assertFalse(rb.is_empty())
        self.assertTrue(rb.is_full())

        rb.popright()
        self.assertFalse(rb.is_full())
        self.assertFalse(rb.is_empty())

    def test_large_inputs(self):
        rb = RingBufferOneDim(10000)
        for i in range(10000):
            rb.append(float(i))
        self.assertTrue(rb.is_full())
        self.assertEqual(rb[-1], 9999.0)


if __name__ == "__main__":
    unittest.main()
