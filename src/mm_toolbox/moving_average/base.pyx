cimport numpy as cnp

from mm_toolbox.ringbuffer.onedim cimport RingBufferOneDim

cdef class MovingAverage:
    def __init__(self, Py_ssize_t window, bint is_fast):
        if window <= 1:
            raise ValueError(f"Moving average window must be positive; expected >1 but got {window}")

        self._window = window
        self._is_fast = is_fast
        self._is_warm = False
        self._value = 0.0
        self._values = RingBufferOneDim(window)

    cdef inline void ensure_warm(self):
        if not self._is_warm:
            raise ValueError("Moving average must be initialized before use.")

    cdef inline void ensure_not_fast(self):
        if self._is_fast:
            raise ValueError("Cannot store/return historical values in fast mode.")

    cdef inline void push_to_ringbuffer(self):
        if not self._is_fast:
            self._values.append(self._value)

    cpdef double initialize(self, cnp.ndarray values):
        """
        Initializes the moving average with the given values. 
        Values must be the length of the window.

        Must be called before any other methods.
        """
        pass

    cpdef double next(self, double value):
        """
        Calculates the next value in the moving average without 
        updating the internal state. Useful for plotting.
        """
        pass

    cpdef double update(self, double value):
        """
        Calculates the next value in the moving average and 
        updates the internal state. 
        """
        pass

    cpdef double get_value(self):
        """
        Returns the current value of the moving average.
        """
        return self._value

    cpdef cnp.ndarray get_values(self):
        """
        Returns the values in the moving average as a numpy array.
        """
        self.ensure_not_fast()
        return self._values.unwrapped()

    def __len__(self):
        """
        Get the number of elements currently in the buffer.

        Returns:
            int: The current size of the buffer.
        """
        self.ensure_not_fast()
        return len(self._values) 

    def __contains__(self, double value):
        """
        Check if a value is present in the buffer.

        Parameters:
            value (float): The value to search for.

        Returns:
            bool: True if the value is in the buffer, False otherwise.
        """
        self.ensure_not_fast()
        return value in self._values

    def __iter__(self):
        """
        Iterate over the elements in the buffer in order from oldest to newest.

        Yields:
            float: Each value in the buffer.
        """
        self.ensure_not_fast()
        return iter(self._values)

    def __getitem__(self, Py_ssize_t idx):
        """
        Get the element at the given index.

        Parameters:
            idx (int): The index of the element to retrieve.

        Returns:
            float: The value at the specified index.

        Raises:
            IndexError: If the index is out of range.
        """
        self.ensure_not_fast()
        return self._values[idx]