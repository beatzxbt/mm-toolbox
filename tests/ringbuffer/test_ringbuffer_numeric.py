"""Tests for NumericRingBuffer implementation."""

import asyncio

import numpy as np
import pytest

from mm_toolbox.ringbuffer.numeric import NumericRingBuffer, resolve_numeric_dtype


class TestDtypeResolutionFunction:
    """Test the dtype resolution function in isolation."""

    def test_numpy_type_objects(self):
        """Test dtype resolution with numpy type objects."""
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
        """Test dtype resolution with string specifications."""
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
        """Test dtype resolution with numpy dtype objects."""
        assert resolve_numeric_dtype(np.dtype(np.int32)).kind == "i"
        assert resolve_numeric_dtype(np.dtype(np.float64)).kind == "f"
        assert resolve_numeric_dtype(np.dtype("int64")).kind == "i"

    def test_python_builtin_types(self):
        """Test auto-casting of Python builtin types."""
        # Python int should map to a suitable integer type
        resolved_int = resolve_numeric_dtype(int)
        assert resolved_int.kind == "i"

        # Python float should map to float64
        resolved_float = resolve_numeric_dtype(float)
        assert resolved_float.kind == "f"
        assert resolved_float.itemsize == 8  # float64

    def test_invalid_dtypes(self):
        """Test that invalid dtypes raise appropriate errors."""
        invalid_dtypes = [np.complex64, np.complex128, "complex64", str, bool]

        for invalid_dtype in invalid_dtypes:
            with pytest.raises(ValueError, match="Unsupported dtype"):
                resolve_numeric_dtype(invalid_dtype)


class TestNumericRingBufferBasics:
    """Test basic NumericRingBuffer functionality."""

    def test_creation_with_different_dtypes(self):
        """Test creating NumericRingBuffer with different dtype specifications."""
        # Test with numpy type
        rb1 = NumericRingBuffer(5, dtype=np.float64)
        assert rb1.raw(copy=True).dtype == np.float64

        # Test with string
        rb2 = NumericRingBuffer(5, dtype="float64")
        assert rb2.raw(copy=True).dtype == np.float64

        # Test with numpy dtype object
        rb3 = NumericRingBuffer(5, dtype=np.dtype(np.float64))
        assert rb3.raw(copy=True).dtype == np.float64

        # Test with Python builtin
        rb4 = NumericRingBuffer(5, dtype=float)
        assert rb4.raw(copy=True).dtype == np.float64

    def test_basic_operations_float64(self):
        """Test basic operations with float64."""
        rb = NumericRingBuffer(5, dtype=np.float64)
        assert rb.is_empty()
        assert rb.raw(copy=True).dtype == np.float64

        # Test batch insertion
        data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        rb.insert_batch(data)

        assert len(rb) == 3
        assert not rb.is_empty()
        np.testing.assert_array_equal(rb.unwrapped(), data)

    def test_basic_operations_int32(self):
        """Test basic operations with int32."""
        rb = NumericRingBuffer(5, dtype=np.int32)

        # Test batch insertion
        data = np.array([10, 20, 30], dtype=np.int32)
        rb.insert_batch(data)

        assert len(rb) == 3
        np.testing.assert_array_equal(rb.unwrapped(), data)

    def test_capacity_power_of_2_rounding(self):
        """Test that capacity gets rounded to power of 2."""
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
        """Test behavior when capacity is exceeded."""
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
        """Test clear operations."""
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
        """Test basic properties and state."""
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
        """Test consume and consume_all."""
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

    def test_indexing(self):
        """Test array-like indexing."""
        rb = NumericRingBuffer(5, dtype="float64")
        data = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
        rb.insert_batch(data)

        # Test positive indexing
        assert rb[0] == 10.0
        assert rb[1] == 20.0
        assert rb[3] == 40.0

        # Test negative indexing
        assert rb[-1] == 40.0
        assert rb[-2] == 30.0

        # Test out of bounds
        with pytest.raises(IndexError):
            _ = rb[10]
        with pytest.raises(IndexError):
            _ = rb[-10]


class TestNumericRingBufferEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_buffer_errors(self):
        """Test that operations on empty buffer raise appropriate errors."""
        rb = NumericRingBuffer(5, dtype="float64")

        # These should raise errors on empty buffer
        with pytest.raises((IndexError, ValueError)):
            rb.consume()

        with pytest.raises(IndexError):
            _ = rb[0]

    def test_single_element_buffer(self):
        """Test buffer with capacity of 1."""
        rb = NumericRingBuffer(1, dtype="float64")

        data = np.array([42.0], dtype=np.float64)
        rb.insert_batch(data)

        assert rb.is_full()
        assert len(rb) == 1
        unwrapped = rb.unwrapped()
        np.testing.assert_array_equal(unwrapped, data)

    def test_capacity_validation(self):
        """Test capacity validation."""
        with pytest.raises(ValueError):
            NumericRingBuffer(0, dtype="float64")
        with pytest.raises(ValueError):
            NumericRingBuffer(-1, dtype="float64")

    def test_dtype_validation(self):
        """Test dtype validation."""
        # Valid dtypes should work
        NumericRingBuffer(5, dtype="int32")
        NumericRingBuffer(5, dtype="uint64")
        NumericRingBuffer(5, dtype="float32")

        # Invalid dtypes should raise errors
        with pytest.raises(ValueError, match="Unsupported dtype"):
            NumericRingBuffer(5, dtype="complex64")


class TestNumericRingBufferAsyncFunctionality:
    """Test async functionality for NumericRingBuffer."""

    @pytest.mark.asyncio
    async def test_async_consume_basic(self):
        """Test basic async consume functionality."""
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
        """Test async consume iterable follows FIFO order."""
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
        """Test that async functions are disabled when disable_async=True."""
        rb = NumericRingBuffer(5, dtype="float64", disable_async=True)

        with pytest.raises(RuntimeError, match="Async operations are disabled"):
            await rb.aconsume()

    @pytest.mark.asyncio
    async def test_async_consume_with_existing_data(self):
        """Test async consume when data already exists."""
        rb = NumericRingBuffer(5, dtype="int64", disable_async=False)

        # Pre-populate with data
        rb.insert_batch(np.array([100, 200], dtype=np.int64))

        # aconsume should immediately return without waiting
        result = await asyncio.wait_for(rb.aconsume(), timeout=0.1)
        assert result == 100  # Should get the oldest item
