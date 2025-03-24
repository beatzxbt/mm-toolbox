import unittest
import numpy as np

from mm_toolbox.ringbuffer.twodim import RingBufferTwoDim


class TestRingBufferTwoDim(unittest.TestCase):
    def test_initialization(self):
        rb = RingBufferTwoDim(5, 3)
        self.assertEqual(len(rb), 0)
        self.assertTrue(rb.is_empty())
        self.assertFalse(rb.is_full())

        # Append one element
        arr = np.array([1.0, 2.0, 3.0])
        rb.append(arr)
        self.assertEqual(len(rb), 1)
        np.testing.assert_array_equal(rb[0], arr)

    def test_append_and_iter(self):
        rb = RingBufferTwoDim(3, 2)
        rb.append(np.array([10.0, 20.0]))
        rb.append(np.array([30.0, 40.0]))
        rb.append(np.array([50.0, 60.0]))

        self.assertEqual(len(rb), 3)
        self.assertTrue(rb.is_full())

        # Iteration should yield in order
        expected = [
            np.array([10.0, 20.0]),
            np.array([30.0, 40.0]),
            np.array([50.0, 60.0]),
        ]
        actual = list(rb)
        for exp, act in zip(expected, actual):
            np.testing.assert_array_equal(exp, act)

        # Append one more, overwriting the oldest
        rb.append(np.array([70.0, 80.0]))
        expected = [
            np.array([30.0, 40.0]),
            np.array([50.0, 60.0]),
            np.array([70.0, 80.0]),
        ]
        actual = list(rb)
        for exp, act in zip(expected, actual):
            np.testing.assert_array_equal(exp, act)

    def test_popright_and_popleft(self):
        rb = RingBufferTwoDim(3, 3)
        rb.append(np.array([1.0, 2.0, 3.0]))
        rb.append(np.array([4.0, 5.0, 6.0]))
        rb.append(np.array([7.0, 8.0, 9.0]))

        val = rb.popright()
        np.testing.assert_array_equal(val, np.array([7.0, 8.0, 9.0]))
        self.assertEqual(len(rb), 2)

        val = rb.popleft()
        np.testing.assert_array_equal(val, np.array([1.0, 2.0, 3.0]))
        self.assertEqual(len(rb), 1)

        val = rb.popright()
        np.testing.assert_array_equal(val, np.array([4.0, 5.0, 6.0]))
        self.assertTrue(rb.is_empty())

        with self.assertRaises(IndexError):
            rb.popright()
        with self.assertRaises(IndexError):
            rb.popleft()

    def test_contains(self):
        rb = RingBufferTwoDim(3, 2)
        rb.append(np.array([10.0, 20.0]))
        rb.append(np.array([30.0, 40.0]))

        self.assertIn(np.array([10.0, 20.0]), rb)
        self.assertNotIn(np.array([50.0, 60.0]), rb)
        rb.append(np.array([50.0, 60.0]))
        self.assertIn(np.array([50.0, 60.0]), rb)

    def test_getitem(self):
        rb = RingBufferTwoDim(5, 2)
        for i in range(5):
            rb.append(np.array([float(i), float(i + 10)]))

        # Buffer:
        # idx 0: [0, 10]
        # idx 1: [1, 11]
        # idx 2: [2, 12]
        # idx 3: [3, 13]
        # idx 4: [4, 14]
        np.testing.assert_array_equal(rb[0], np.array([0.0, 10.0]))
        np.testing.assert_array_equal(rb[4], np.array([4.0, 14.0]))
        np.testing.assert_array_equal(rb[-1], np.array([4.0, 14.0]))
        np.testing.assert_array_equal(rb[-5], np.array([0.0, 10.0]))

        with self.assertRaises(IndexError):
            _ = rb[5]
        with self.assertRaises(IndexError):
            _ = rb[-6]

    def test_reset(self):
        rb = RingBufferTwoDim(4, 2)
        rb.append(np.array([10.0, 20.0]))
        rb.append(np.array([30.0, 40.0]))
        rb.append(np.array([50.0, 60.0]))

        old_data = rb.reset()
        np.testing.assert_array_equal(
            old_data, np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
        )
        self.assertTrue(rb.is_empty())
        self.assertEqual(len(rb), 0)

    def test_fast_reset(self):
        rb = RingBufferTwoDim(4, 2)
        rb.append(np.array([100.0, 200.0]))
        rb.append(np.array([300.0, 400.0]))

        rb.fast_reset()
        self.assertTrue(rb.is_empty())
        self.assertEqual(len(rb), 0)

    def test_unwrapped(self):
        rb = RingBufferTwoDim(5, 2)
        for i in range(5):
            rb.append(np.array([float(i), float(i + 10)]))

        uw = rb.unwrapped()
        expected = np.array(
            [[0.0, 10.0], [1.0, 11.0], [2.0, 12.0], [3.0, 13.0], [4.0, 14.0]]
        )
        np.testing.assert_array_equal(uw, expected)

        # Overwrite one
        rb.append(np.array([5.0, 15.0]))
        uw = rb.unwrapped()
        expected = np.array(
            [[1.0, 11.0], [2.0, 12.0], [3.0, 13.0], [4.0, 14.0], [5.0, 15.0]]
        )
        np.testing.assert_array_equal(uw, expected)

    def test_raw_and_unsafe_raw(self):
        rb = RingBufferTwoDim(3, 3)
        rb.append(np.array([1.0, 2.0, 3.0]))
        rb.append(np.array([4.0, 5.0, 6.0]))

        raw_copy = rb.raw()
        self.assertEqual(raw_copy.shape, (3, 3))
        np.testing.assert_array_equal(raw_copy[0], np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(raw_copy[1], np.array([4.0, 5.0, 6.0]))

        # unsafe_raw is a view, can modify internal state.
        unsafe = rb.unsafe_raw()
        unsafe[0, 0] = 999.0
        np.testing.assert_array_equal(rb[0], np.array([999.0, 2.0, 3.0]))

    def test_pop_errors(self):
        rb = RingBufferTwoDim(2, 2)
        with self.assertRaises(IndexError):
            rb.popright()

        with self.assertRaises(IndexError):
            rb.popleft()

    def test_is_full_and_empty_states(self):
        rb = RingBufferTwoDim(2, 2)
        self.assertTrue(rb.is_empty())
        self.assertFalse(rb.is_full())

        rb.append(np.array([1.0, 2.0]))
        self.assertFalse(rb.is_empty())
        self.assertFalse(rb.is_full())

        rb.append(np.array([3.0, 4.0]))
        self.assertFalse(rb.is_empty())
        self.assertTrue(rb.is_full())

        rb.popright()
        self.assertFalse(rb.is_full())
        self.assertFalse(rb.is_empty())


if __name__ == "__main__":
    unittest.main()
