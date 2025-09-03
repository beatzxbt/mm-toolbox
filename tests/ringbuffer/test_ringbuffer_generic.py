"""Tests for GenericRingBuffer implementation."""

import asyncio

import pytest

from mm_toolbox.ringbuffer.generic import GenericRingBuffer


class TestGenericRingBufferBasics:
    """Test basic GenericRingBuffer functionality."""

    def test_initialization(self):
        """Test GenericRingBuffer initialization."""
        rb = GenericRingBuffer(5)
        assert rb.is_empty()
        assert not rb.is_full()
        assert len(rb) == 0

    def test_capacity_validation(self):
        """Test capacity validation."""
        with pytest.raises(ValueError):
            GenericRingBuffer(0)
        with pytest.raises(ValueError):
            GenericRingBuffer(-1)

    def test_power_of_2_capacity_rounding(self):
        """Test that capacity gets rounded to power of 2."""
        test_cases = [(3, 4), (5, 8), (10, 16), (16, 16), (17, 32)]

        for requested, expected in test_cases:
            rb = GenericRingBuffer(requested)
            # Test indirectly by filling beyond requested capacity
            data = list(range(expected + 2))
            rb.insert_batch(data)
            assert len(rb.unwrapped()) == expected

    def test_insert_and_peek_operations(self):
        """Test insert and peek operations with various data types."""
        rb = GenericRingBuffer(5)

        # Test with different data types
        test_data = [{"a": 1}, [1, 2, 3], "string", 42, 3.14]

        for item in test_data:
            rb.insert(item)

        assert len(rb) == 5
        assert rb.peekleft() == {"a": 1}  # First inserted
        assert rb.peekright() == 3.14  # Last inserted

        # Test that objects are stored by reference
        dict_obj = {"mutable": "test"}
        rb.clear()
        rb.insert(dict_obj)
        retrieved = rb.peekleft()
        assert retrieved is dict_obj  # Same object reference

    def test_batch_operations(self):
        """Test batch insert and related operations."""
        rb = GenericRingBuffer(4)

        # Test batch insert
        batch_data = ["a", "b", "c"]
        rb.insert_batch(batch_data)
        assert list(rb.unwrapped()) == batch_data

        # Test batch insert with overflow
        large_batch = ["x", "y", "z", "w", "v", "u"]
        rb.clear()
        rb.insert_batch(large_batch)
        # Should keep last 4 elements (capacity = 4)
        expected = ["z", "w", "v", "u"]
        assert list(rb.unwrapped()) == expected

    def test_overwrite_operations(self):
        """Test overwrite_latest functionality."""
        rb = GenericRingBuffer(3)
        rb.insert_batch([1, 2, 3])

        # Test overwrite without incrementing count
        rb.overwrite_latest(99, increment_count=False)
        assert len(rb) == 3  # Count unchanged
        assert list(rb.unwrapped()) == [1, 2, 99]

        # Test overwrite with incrementing count
        rb.overwrite_latest(100, increment_count=True)
        # The capacity is 4 (rounded from 3), so we can have 4 elements
        unwrapped = rb.unwrapped()
        assert len(unwrapped) == 4
        assert unwrapped == [1, 2, 99, 100]

    def test_consume_operations(self):
        """Test consume and consume_all operations."""
        rb = GenericRingBuffer(4)
        rb.insert_batch(["a", "b", "c"])

        # Test single consume (should return newest)
        consumed = rb.consume()
        assert consumed == "c"
        assert len(rb) == 2
        assert list(rb.unwrapped()) == ["a", "b"]

        # Test consume_all
        all_consumed = rb.consume_all()
        assert all_consumed == ["a", "b"]
        assert rb.is_empty()

    def test_indexing_operations(self):
        """Test array-like indexing."""
        rb = GenericRingBuffer(5)
        data = ["x", "y", "z", "w"]
        rb.insert_batch(data)

        # Test positive indexing
        assert rb[0] == "x"
        assert rb[1] == "y"
        assert rb[3] == "w"

        # Test negative indexing
        assert rb[-1] == "w"
        assert rb[-2] == "z"

        # Test out of bounds
        with pytest.raises(IndexError):
            _ = rb[10]
        with pytest.raises(IndexError):
            _ = rb[-10]

    def test_contains_operations(self):
        """Test membership testing."""
        rb = GenericRingBuffer(4)
        rb.insert_batch([1, "hello", [1, 2], {"key": "value"}])

        assert 1 in rb
        assert "hello" in rb
        assert [1, 2] in rb
        assert {"key": "value"} in rb
        assert "missing" not in rb
        assert 999 not in rb

    def test_clear_operations(self):
        """Test clear functionality."""
        rb = GenericRingBuffer(3)
        rb.insert_batch(["a", "b", "c"])
        assert len(rb) == 3

        rb.clear()
        assert rb.is_empty()
        assert len(rb) == 0

        # Test that we can use it normally after clear
        rb.insert("new_item")
        assert len(rb) == 1
        assert rb.peekleft() == "new_item"


class TestGenericRingBufferEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_buffer_operations(self):
        """Test operations on empty buffer."""
        rb = GenericRingBuffer(5)

        # These should raise errors on empty buffer
        with pytest.raises((IndexError, ValueError)):
            rb.consume()

        with pytest.raises(IndexError):
            _ = rb[0]

        # peekleft and peekright may return None or raise errors on empty buffer
        try:
            result = rb.peekleft()
            # If it doesn't raise, it should return None or similar
            assert result is None or result == 0
        except (IndexError, ValueError):
            pass  # This is also acceptable behavior

    def test_single_element_buffer(self):
        """Test buffer with capacity of 1."""
        rb = GenericRingBuffer(1)

        rb.insert("single")
        assert rb.is_full()
        assert len(rb) == 1
        assert rb.peekleft() == "single"
        assert rb.peekright() == "single"

        # Inserting another should overwrite
        rb.insert("new")
        assert len(rb) == 1
        assert rb.peekleft() == "new"

    def test_object_reference_handling(self):
        """Test that objects are stored by reference."""
        rb = GenericRingBuffer(3)

        # Test mutable objects
        mutable_list = [1, 2, 3]
        mutable_dict = {"key": "value"}

        rb.insert(mutable_list)
        rb.insert(mutable_dict)

        # Modify original objects
        mutable_list.append(4)
        mutable_dict["new_key"] = "new_value"

        # Should reflect changes (stored by reference)
        retrieved_list = rb.peekleft()
        retrieved_dict = rb.peekright()

        assert retrieved_list == [1, 2, 3, 4]
        assert retrieved_dict == {"key": "value", "new_key": "new_value"}
        assert retrieved_list is mutable_list
        assert retrieved_dict is mutable_dict

    def test_none_and_falsy_values(self):
        """Test handling of None and falsy values."""
        rb = GenericRingBuffer(5)

        falsy_values = [None, False, 0, "", [], {}]
        rb.insert_batch(falsy_values)

        unwrapped = rb.unwrapped()
        assert len(unwrapped) == 6
        assert unwrapped == falsy_values

        # Test contains with falsy values
        assert None in rb
        assert False in rb
        assert 0 in rb


class TestGenericRingBufferAsyncFunctionality:
    """Test async functionality comprehensively."""

    @pytest.mark.asyncio
    async def test_async_consume_basic(self):
        """Test basic async consume functionality."""
        rb = GenericRingBuffer(5, disable_async=False)

        async def waiter():
            return await rb.aconsume()

        # Start waiting before inserting
        task = asyncio.create_task(waiter())
        await asyncio.sleep(0.01)  # Let task start waiting

        # Insert item - should wake up the waiter
        rb.insert("test_item")
        result = await asyncio.wait_for(task, timeout=1.0)
        assert result == "test_item"

    @pytest.mark.asyncio
    async def test_async_consume_sequential(self):
        """Test sequential async consume operations."""
        rb = GenericRingBuffer(10, disable_async=False)

        # Pre-populate with data
        rb.insert_batch(["item_1", "item_2", "item_3"])

        # Test that aconsume works with existing data
        result1 = await rb.aconsume()
        result2 = await rb.aconsume()
        result3 = await rb.aconsume()

        # Should consume in LIFO order (newest first)
        assert result1 == "item_3"
        assert result2 == "item_2"
        assert result3 == "item_1"

        assert rb.is_empty()

    @pytest.mark.asyncio
    async def test_async_consume_iterable(self):
        """Test async consume iterable functionality."""
        rb = GenericRingBuffer(5, disable_async=False)

        collected = []

        async def producer():
            for i in range(5):
                await asyncio.sleep(0.01)
                rb.insert(f"item_{i}")

        async def consumer():
            async for item in rb.aconsume_iterable():
                collected.append(item)
                if len(collected) == 5:
                    break

        # Run producer and consumer concurrently
        producer_task = asyncio.create_task(producer())
        consumer_task = asyncio.create_task(consumer())

        await asyncio.wait_for(
            asyncio.gather(producer_task, consumer_task), timeout=5.0
        )

        expected = ["item_0", "item_1", "item_2", "item_3", "item_4"]
        assert collected == expected

    @pytest.mark.asyncio
    async def test_async_disabled_mode(self):
        """Test that async functions are disabled when disable_async=True."""
        rb = GenericRingBuffer(5, disable_async=True)

        # Async methods should raise errors when disabled
        with pytest.raises(RuntimeError, match="Async operations are disabled"):
            await rb.aconsume()

        with pytest.raises(RuntimeError, match="Async operations are disabled"):
            async for item in rb.aconsume_iterable():
                pass

    @pytest.mark.asyncio
    async def test_async_timeout_behavior(self):
        """Test async timeout behavior."""
        rb = GenericRingBuffer(5, disable_async=False)

        # Test that aconsume times out when no items are available
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(rb.aconsume(), timeout=0.1)

    @pytest.mark.asyncio
    async def test_async_consume_with_existing_data(self):
        """Test async consume when data already exists."""
        rb = GenericRingBuffer(5, disable_async=False)

        # Pre-populate with data
        rb.insert_batch(["existing_1", "existing_2"])

        # aconsume should immediately return without waiting
        result = await asyncio.wait_for(rb.aconsume(), timeout=0.1)
        assert result == "existing_2"  # Should get the newest item

        # Buffer should now have one less item
        assert len(rb) == 1
        assert rb.peekleft() == "existing_1"


class TestGenericRingBufferDataTypes:
    """Test GenericRingBuffer with various data types."""

    def test_primitive_types(self):
        """Test with primitive data types."""
        rb = GenericRingBuffer(5)

        primitives = [1, 3.14, "string", True, None]
        rb.insert_batch(primitives)

        unwrapped = rb.unwrapped()
        assert unwrapped == primitives

    def test_complex_objects(self):
        """Test with complex objects."""
        rb = GenericRingBuffer(5)

        class CustomClass:
            def __init__(self, value):
                self.value = value

            def __eq__(self, other):
                return isinstance(other, CustomClass) and self.value == other.value

        complex_objects = [
            {"nested": {"dict": "value"}},
            [[1, 2], [3, 4]],
            CustomClass("test"),
            lambda x: x * 2,
            {"function": lambda: "closure"},
        ]

        rb.insert_batch(complex_objects)
        unwrapped = rb.unwrapped()

        assert len(unwrapped) == 5
        assert unwrapped[0] == {"nested": {"dict": "value"}}
        assert unwrapped[1] == [[1, 2], [3, 4]]
        assert unwrapped[2] == CustomClass("test")
        assert callable(unwrapped[3])
        assert callable(unwrapped[4]["function"])

    def test_large_objects(self):
        """Test with large objects."""
        rb = GenericRingBuffer(3)

        # Create large objects
        large_list = list(range(1000))
        large_dict = {f"key_{i}": f"value_{i}" for i in range(100)}
        large_string = "x" * 10000

        rb.insert_batch([large_list, large_dict, large_string])

        unwrapped = rb.unwrapped()
        assert len(unwrapped) == 3
        assert unwrapped[0] == large_list
        assert unwrapped[1] == large_dict
        assert unwrapped[2] == large_string


class TestGenericRingBufferPerformance:
    """Test performance-related functionality."""

    def test_disable_async_option(self):
        """Test async disable option."""
        rb_async = GenericRingBuffer(5, disable_async=False)
        # Just test that it was created successfully
        assert rb_async is not None

        rb_no_async = GenericRingBuffer(5, disable_async=True)
        # Test that async methods raise errors (tested elsewhere)
        assert rb_no_async is not None

    def test_memory_efficiency_with_overwrites(self):
        """Test memory efficiency with many overwrites."""
        rb = GenericRingBuffer(100)

        # Fill with data
        initial_data = [f"item_{i}" for i in range(50)]
        rb.insert_batch(initial_data)
        assert len(rb) == 50

        # Perform many overwrites
        for i in range(10):
            rb.overwrite_latest(f"overwrite_{i}", increment_count=False)

        # Should still have same count with last overwrite
        assert len(rb) == 50
        unwrapped = rb.unwrapped()
        assert unwrapped[-1] == "overwrite_9"
