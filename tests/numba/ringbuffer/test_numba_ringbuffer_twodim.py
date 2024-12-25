import unittest
import numpy as np

from mm_toolbox.numba.ringbuffer import RingBufferTwoDim

class TestRingBufferTwoDim(unittest.TestCase):

    def test_initialization(self):
        rb = RingBufferTwoDim(5, 3)
        self.assertEqual(len(rb), 0)
        self.assertTrue(rb.is_empty())
        self.assertFalse(rb.is_full())

        # append one sub-array
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        rb.append(arr)
        self.assertEqual(len(rb), 1)

    def test_append_and_unwrapped(self):
        rb = RingBufferTwoDim(3, 2)
        rb.append(np.array([10.0, 20.0]))
        rb.append(np.array([30.0, 40.0]))
        rb.append(np.array([50.0, 60.0]))

        self.assertEqual(len(rb), 3)
        self.assertTrue(rb.is_full())

        # unwrapped should yield in the order appended
        expected = np.array([[10.0, 20.0],
                             [30.0, 40.0],
                             [50.0, 60.0]], dtype=np.float64)
        np.testing.assert_array_equal(rb.unwrapped(), expected)

        # Overwrite the oldest
        rb.append(np.array([70.0, 80.0]))
        # Now it should contain: [30,40],[50,60],[70,80]
        expected2 = np.array([[30.0, 40.0],
                              [50.0, 60.0],
                              [70.0, 80.0]], dtype=np.float64)
        np.testing.assert_array_equal(rb.unwrapped(), expected2)
        self.assertEqual(len(rb), 3)

    def test_popright_and_popleft(self):
        rb = RingBufferTwoDim(3, 3)
        rb.append(np.array([1.0, 2.0, 3.0]))
        rb.append(np.array([4.0, 5.0, 6.0]))
        rb.append(np.array([7.0, 8.0, 9.0]))

        val = rb.popright()  # last element
        np.testing.assert_array_equal(val, np.array([7.0, 8.0, 9.0]))
        self.assertEqual(len(rb), 2)
        
        val = rb.popleft()   # first element
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
            rb.append(np.array([float(i), float(i + 10)], dtype=np.float64))

        # buffer: idx=0 => [0,10], idx=1 => [1,11], idx=2 => [2,12], etc.
        np.testing.assert_array_equal(rb[0], np.array([0.0, 10.0]))
        np.testing.assert_array_equal(rb[4], np.array([4.0, 14.0]))
        np.testing.assert_array_equal(rb[-1], np.array([4.0, 14.0])) 
        np.testing.assert_array_equal(rb[-5], np.array([0.0, 10.0]))
        
        with self.assertRaises(IndexError):
            _ = rb[5]
        with self.assertRaises(IndexError):
            _ = rb[-6]

    def test_reset(self):
        rb = RingBufferTwoDim(4, 3)
        rb.append(np.array([10, 20, 30], dtype=np.float64))
        rb.append(np.array([40, 50, 60], dtype=np.float64))
        rb.append(np.array([70, 80, 90], dtype=np.float64))

        old_data = rb.reset()
        expected = np.array([[10.,20.,30.],
                             [40.,50.,60.],
                             [70.,80.,90.]], dtype=np.float64)
        np.testing.assert_array_equal(old_data, expected)
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
            rb.append(np.array([float(i), float(i+10)], dtype=np.float64))

        # unwrapped => [[0,10],[1,11],[2,12],[3,13],[4,14]]
        uw = rb.unwrapped()
        expected = np.array([[0,10],[1,11],[2,12],[3,13],[4,14]], dtype=np.float64)
        np.testing.assert_array_equal(uw, expected)

        # Overwrite one
        rb.append(np.array([5.0,15.0]))
        # => [1,11],[2,12],[3,13],[4,14],[5,15]
        uw = rb.unwrapped()
        expected2 = np.array([[1,11],[2,12],[3,13],[4,14],[5,15]], dtype=np.float64)
        np.testing.assert_array_equal(uw, expected2)

    def test_raw_and_unsafe_raw(self):
        rb = RingBufferTwoDim(3, 3)
        rb.append(np.array([1.0, 2.0, 3.0]))
        rb.append(np.array([4.0, 5.0, 6.0]))

        raw_copy = rb.raw()
        self.assertEqual(raw_copy.shape, (3,3))
        np.testing.assert_array_equal(raw_copy[0], np.array([1.0,2.0,3.0]))
        np.testing.assert_array_equal(raw_copy[1], np.array([4.0,5.0,6.0]))

        # unsafe_raw is a view => modifies internal
        unsafe = rb.unsafe_raw()
        unsafe[0,0] = 999.0
        np.testing.assert_array_equal(rb[0], np.array([999.0,2.0,3.0]))

    def test_pop_errors(self):
        rb = RingBufferTwoDim(2,2)
        with self.assertRaises(IndexError):
            rb.popright()
        with self.assertRaises(IndexError):
            rb.popleft()

    def test_is_full_and_empty_states(self):
        rb = RingBufferTwoDim(2,2)
        self.assertTrue(rb.is_empty())
        self.assertFalse(rb.is_full())

        rb.append(np.array([1.0,2.0]))
        self.assertFalse(rb.is_empty())
        self.assertFalse(rb.is_full())

        rb.append(np.array([3.0,4.0]))
        self.assertFalse(rb.is_empty())
        self.assertTrue(rb.is_full())

        rb.popright()
        self.assertFalse(rb.is_full())
        self.assertFalse(rb.is_empty())

    def test_large_inputs(self):
        rb = RingBufferTwoDim(1000,2)
        for i in range(1000):
            rb.append(np.array([float(i), float(i+10)]))
        self.assertTrue(rb.is_full())
        np.testing.assert_array_equal(rb[-1], np.array([999.0, 1009.0]))


if __name__ == '__main__':
    unittest.main()
