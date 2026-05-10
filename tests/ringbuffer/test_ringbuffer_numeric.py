"""Tests for NumericRingBuffer implementation.

Layer 2 tests: validate NumericRingBuffer including dtype resolution,
creation with different dtypes, basic operations, overflow behavior,
async functionality, and timestamp tracking.
"""

import asyncio

import numpy as np
import pytest

from mm_toolbox.ringbuffer.numeric import NumericRingBuffer, resolve_numeric_dtype


class TestDtypeResolutionFunction:
    """Layer 1: Test the dtype resolution function in isolation."""

    def test_numpy_type_objects(self):
        """Given numpy type objects, When resolved, Then correct kind returned."""
        # Test all supported numpy types
        assert resolve_numeric_dtype(np.int8).kind == "i"
        assert resolve_numeric_dtype(np.int16).kind == "i"
        assert resolve_numeric_dtype(np.int32).kind == "i"
        assert resolve_numeric_dtype(np.int64).kind == "i"

        assert resolve_numeric_dtype(np.uint8).kind == "u"
        assert resolve_numeric_dtype(np.uint16).kind == "u"
        assert resolve_numeric_dtype(np.uint32).kind == "u"
        assert resolve_numeric_dtype(np.uint64).kind == "u"

        assert resolve_numeric_dtype(np.float32).kind == "f"
        assert resolve_numeric_dtype(np.float64).kind == "f"

    def test_string_specifications(self):
        """Given string dtype specs, When resolved, Then correct kind returned."""
        # Test integer strings
        assert resolve_numeric_dtype("int8").kind == "i"
        assert resolve_numeric_dtype("int16").kind == "i"
        assert resolve_numeric_dtype("int32").kind == "i"
        assert resolve_numeric_dtype("int64").kind == "i"

        # Test unsigned integer strings
        assert resolve_numeric_dtype("uint8").kind == "u"
        assert resolve_numeric_dtype("uint16").kind == "u"
        assert resolve_numeric_dtype("uint32").kind == "u"
        assert resolve_numeric_dtype("uint64").kind == "u"

        # Test float strings
        assert resolve_numeric_dtype("float32").kind == "f"
        assert resolve_numeric_dtype("float64").kind == "f"

    def test_numpy_dtype_objects(self):
        """Given numpy dtype objects, When resolved, Then correct kind returned."""
        assert resolve_numeric_dtype(np.dtype(np.int32)).kind == "i"
        assert resolve_numeric_dtype(np.dtype(np.float64)).kind == "f"
        assert resolve_numeric_dtype(np.dtype("int64")).kind == "i"

    def test_python_builtin_types(self):
        """Given Python builtin types, When resolved, Then auto-cast correctly."""
        # Python int should map to a suitable integer type
        resolved_int = resolve_numeric_dtype(int)
        assert resolved_int.kind == "i"

        # Python float should map to float64
        resolved_float = resolve_numeric_dtype(float)
        assert resolved_float.kind == "f"
        assert resolved_float.itemsize == 8  # float64

    def test_invalid_dtypes(self):
        """Given invalid dtypes, When resolved, Then raises ValueError."""
        invalid_dtypes = [np.complex64, np.complex128, "complex64", str, bool]

        for invalid_dtype in invalid_dtypes:
            with pytest.raises(ValueError, match="Unsupported dtype"):
                resolve_numeric_dtype(invalid_dtype)


class TestNumericRingBufferBasics:
    """Layer 2: Test basic NumericRingBuffer functionality."""

    def test_creation_with_different_dtypes(self):
        """Given different dtype specs, When creating NumericRingBuffer, Then dtype set correctly."""
        # Test with numpy type
        rb1 = NumericRingBuffer(5, dtype=np.float64)
        assert rb1.unwrapped().dtype == np.float64

        # Test with string
        rb2 = NumericRingBuffer(5, dtype="float64")
        assert rb2.unwrapped().dtype == np.float64

        # Test with numpy dtype object
        rb3 = NumericRingBuffer(5, dtype=np.dtype(np.float64))
        assert rb3.unwrapped().dtype == np.float64

        # Test with Python builtin
        rb4 = NumericRingBuffer(5, dtype=float)
        assert rb4.unwrapped().dtype == np.float64

    def test_basic_operations_float64(self):
        """Given float64 buffer, When operations performed, Then works correctly."""
        rb = NumericRingBuffer(5, dtype=np.float64)
        assert rb.is_empty()
        assert rb.unwrapped().dtype == np.float64

        # Test batch insertion
        data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        rb.insert_batch(data)

        assert len(rb) == 3
        assert not rb.is_empty()
        np.testing.assert_array_equal(rb.unwrapped(), data)

    def test_basic_operations_int32(self):
        """Given int32 buffer, When operations performed, Then works correctly."""
        rb = NumericRingBuffer(5, dtype=np.int32)

        # Test batch insertion
        data = np.array([10, 20, 30], dtype=np.int32)
        rb.insert_batch(data)

        assert len(rb) == 3
        np.testing.assert_array_equal(rb.unwrapped(), data)

    def test_capacity_power_of_2_rounding(self):
        """Given non-power-of-2 capacity, When created, Then rounded up."""
        test_cases = [(3, 4), (5, 8), (10, 16), (16, 16), (17, 32)]

        for requested, expected in test_cases:
            rb = NumericRingBuffer(requested, dtype="float64")
            # Test indirectly by filling and checking actual capacity
            data = np.arange(
                expected + 2, dtype=np.float64
            )  # More than expected capacity
            rb.insert_batch(data)
            assert len(rb.unwrapped()) == expected

    def test_overflow_behavior(self):
        """Given data exceeding capacity, When inserted, Then oldest overwritten."""
        rb = NumericRingBuffer(3, dtype="int32")  # Will round to capacity 4

        # Insert more than capacity
        data = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
        rb.insert_batch(data)

        # Should keep only the last elements that fit in capacity
        unwrapped = rb.unwrapped()
        # Capacity should be 4 (next power of 2 from 3)
        assert len(unwrapped) == 4
        expected = np.array([3, 4, 5, 6], dtype=np.int32)  # Last 4 elements
        np.testing.assert_array_equal(unwrapped, expected)

    def test_clear_operations(self):
        """Given populated buffer, When clear called, Then emptied."""
        rb = NumericRingBuffer(5, dtype="float64")

        # Add data
        data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        rb.insert_batch(data)
        assert len(rb) == 3

        # Test clear
        rb.clear()
        assert rb.is_empty()
        assert len(rb) == 0

    def test_basic_properties(self):
        """Given populated buffer, When properties accessed, Then correct values."""
        rb = NumericRingBuffer(5, dtype="int64")
        data = np.array([10, 20, 30], dtype=np.int64)
        rb.insert_batch(data)

        assert len(rb) == 3
        assert not rb.is_empty()
        assert not rb.is_full()

        # Test that we can access data through unwrapped
        unwrapped = rb.unwrapped()
        np.testing.assert_array_equal(unwrapped, data)

    def test_consume_operations(self):
        """Given populated buffer, When consume called, Then FIFO order respected."""
        rb = NumericRingBuffer(4, dtype=np.int32)
        data = np.array([5, 6, 7], dtype=np.int32)
        rb.insert_batch(data)

        # Test single consume
        consumed = rb.consume()
        assert consumed == 5  # Should be the first (oldest) element
        assert len(rb) == 2

        # Test consume_all
        remaining = rb.consume_all()
        expected = np.array([6, 7], dtype=np.int32)
        np.testing.assert_array_equal(remaining, expected)
        assert rb.is_empty()


class TestNumericRingBufferEdgeCases:
    """Layer 2: Test edge cases and error conditions."""

    def test_empty_buffer_errors(self):
        """Given empty buffer, When consume called, Then raises error."""
        rb = NumericRingBuffer(5, dtype="float64")

        # These should raise errors on empty buffer
        with pytest.raises((IndexError, ValueError)):
            rb.consume()

    def test_single_element_buffer(self):
        """Given capacity=1, When operations performed, Then works correctly."""
        rb = NumericRingBuffer(1, dtype="float64")

        data = np.array([42.0], dtype=np.float64)
        rb.insert_batch(data)

        assert rb.is_full()
        assert len(rb) == 1
        unwrapped = rb.unwrapped()
        np.testing.assert_array_equal(unwrapped, data)

    def test_capacity_validation(self):
        """Given invalid capacity, When creating NumericRingBuffer, Then raises ValueError."""
        with pytest.raises(ValueError):
            NumericRingBuffer(0, dtype="float64")
        with pytest.raises(ValueError):
            NumericRingBuffer(-1, dtype="float64")

    def test_dtype_validation(self):
        """Given invalid dtype, When creating NumericRingBuffer, Then raises ValueError."""
        # Valid dtypes should work
        NumericRingBuffer(5, dtype="int32")
        NumericRingBuffer(5, dtype="uint64")
        NumericRingBuffer(5, dtype="float32")

        # Invalid dtypes should raise errors
        with pytest.raises(ValueError, match="Unsupported dtype"):
            NumericRingBuffer(5, dtype="complex64")

    def test_insert_returns_bool(self):
        """Given item, When inserted, Then returns True."""
        rb = NumericRingBuffer(4, dtype="int64")
        assert rb.insert(1) is True
        assert rb.insert_batch(np.array([2, 3], dtype=np.int64)) is True
        assert len(rb) == 3


class TestNumericRingBufferAsyncFunctionality:
    """Layer 3: Test async functionality for NumericRingBuffer."""

    @pytest.mark.asyncio
    async def test_async_consume_basic(self):
        """Given async consumer waiting, When data inserted, Then consumer wakes."""
        rb = NumericRingBuffer(5, dtype="float64", disable_async=False)

        async def waiter():
            return await rb.aconsume()

        # Start waiting before inserting
        task = asyncio.create_task(waiter())
        await asyncio.sleep(0.01)

        # Insert item - should wake up the waiter
        rb.insert_batch(np.array([42.5], dtype=np.float64))
        result = await asyncio.wait_for(task, timeout=1.0)
        assert result == 42.5

    @pytest.mark.asyncio
    async def test_async_consume_iterable_fifo(self):
        """Given async iterable consumer, When producer feeds data, Then FIFO order."""
        rb = NumericRingBuffer(8, dtype="int64", disable_async=False)

        rb.insert_batch(np.array([10, 20, 30], dtype=np.int64))
        collected = []

        async for item in rb.aconsume_iterable():
            collected.append(item)
            if len(collected) == 3:
                break

        assert collected == [10, 20, 30]
        assert rb.is_empty()

    @pytest.mark.asyncio
    async def test_async_disabled_mode(self):
        """Given async disabled, When async methods called, Then raise RuntimeError."""
        rb = NumericRingBuffer(5, dtype="float64", disable_async=True)

        with pytest.raises(RuntimeError, match="Async operations are disabled"):
            await rb.aconsume()

    @pytest.mark.asyncio
    async def test_async_consume_with_existing_data(self):
        """Given existing data, When aconsume called, Then returns immediately."""
        rb = NumericRingBuffer(5, dtype="int64", disable_async=False)

        # Pre-populate with data
        rb.insert_batch(np.array([100, 200], dtype=np.int64))

        # aconsume should immediately return without waiting
        result = await asyncio.wait_for(rb.aconsume(), timeout=0.1)
        assert result == 100  # Should get the oldest item


class TestNumericRingBufferTimestamps:
    """Layer 2: Test timestamp tracking on numeric ringbuffers."""

    def test_latest_insert_time_ns_initial(self):
        """Given fresh buffer, When checking timestamp, Then returns 0."""
        rb = NumericRingBuffer(4, dtype="float64")
        assert rb.latest_insert_time_ns == 0

    def test_latest_insert_time_ns_updated_on_insert(self):
        """Given insert, When checking timestamp, Then updated and monotonic."""
        rb = NumericRingBuffer(4, dtype="float64")
        rb.insert(1.0)
        t1 = rb.latest_insert_time_ns
        assert t1 > 0
        import time

        time.sleep(0.001)
        rb.insert(2.0)
        t2 = rb.latest_insert_time_ns
        assert t2 > t1

    def test_latest_insert_time_ns_updated_on_insert_batch(self):
        """Given insert_batch, When checking timestamp, Then updated."""
        rb = NumericRingBuffer(4, dtype="float64")
        rb.insert_batch(np.array([1.0, 2.0], dtype=np.float64))
        assert rb.latest_insert_time_ns > 0

    def test_latest_consume_time_ns_initial(self):
        """Given fresh buffer, When checking consume timestamp, Then returns 0."""
        rb = NumericRingBuffer(4, dtype="float64")
        assert rb.latest_consume_time_ns == 0

    def test_latest_consume_time_ns_updated_on_consume(self):
        """Given consume, When checking timestamp, Then updated and monotonic."""
        rb = NumericRingBuffer(4, dtype="float64")
        rb.insert_batch(np.array([1.0, 2.0], dtype=np.float64))
        t1 = rb.latest_insert_time_ns
        rb.consume()
        t2 = rb.latest_consume_time_ns
        assert t2 > 0
        assert t2 >= t1
