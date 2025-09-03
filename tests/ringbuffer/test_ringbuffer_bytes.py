"""Tests for BytesRingBuffer implementation."""

import asyncio

import pytest

from mm_toolbox.ringbuffer.bytes import BytesRingBuffer


class TestBytesRingBufferBasics:
    """Test basic BytesRingBuffer functionality."""

    def test_initialization(self):
        """Test BytesRingBuffer initialization."""
        rb = BytesRingBuffer(5)
        assert rb.is_empty()
        assert not rb.is_full()
        assert len(rb) == 0

    def test_capacity_validation(self):
        """Test capacity validation."""
        with pytest.raises(ValueError):
            BytesRingBuffer(0)
        with pytest.raises(ValueError):
            BytesRingBuffer(-1)

    def test_power_of_2_capacity_rounding(self):
        """Test that capacity gets rounded to power of 2."""
        test_cases = [(3, 4), (5, 8), (10, 16), (16, 16), (17, 32)]

        for requested, expected in test_cases:
            rb = BytesRingBuffer(requested)
            # Test indirectly by filling beyond requested capacity
            data = [f"item_{i}".encode() for i in range(expected + 2)]
            rb.insert_batch(data)
            assert len(rb.unwrapped()) == expected

    def test_bytes_operations(self):
        """Test operations with various bytes objects."""
        rb = BytesRingBuffer(5)

        # Test different bytes formats
        test_data = [
            b"simple",
            b"unicode_string",
            b"\x00\x01\x02\x03",  # binary data
            b"",  # empty bytes
            "emoji_🚀".encode(),
        ]

        for data in test_data:
            rb.insert(data)

        assert len(rb) == 5
        unwrapped = rb.unwrapped()
        assert unwrapped == test_data

    def test_batch_operations(self):
        """Test batch insert operations."""
        rb = BytesRingBuffer(4)

        batch_data = [b"a", b"b", b"c"]
        rb.insert_batch(batch_data)
        assert rb.unwrapped() == batch_data

        # Test batch with overflow
        large_batch = [f"item_{i}".encode() for i in range(10)]
        rb.clear()
        rb.insert_batch(large_batch)
        # Should keep last 4 elements (capacity = 4)
        expected = large_batch[-4:]
        assert rb.unwrapped() == expected

    def test_peek_and_consume_operations(self):
        """Test peek and consume operations."""
        rb = BytesRingBuffer(4)
        rb.insert_batch([b"first", b"second", b"third"])

        # Test peek operations
        assert rb.peekleft() == b"first"
        assert rb.peekright() == b"third"

        # Test consume (should return newest)
        consumed = rb.consume()
        assert consumed == b"third"
        assert len(rb) == 2

        # Test consume_all
        remaining = rb.consume_all()
        assert remaining == [b"first", b"second"]
        assert rb.is_empty()

    def test_indexing_and_contains(self):
        """Test indexing and contains operations."""
        rb = BytesRingBuffer(5)
        data = [b"alpha", b"beta", b"gamma"]
        rb.insert_batch(data)

        # Test indexing
        assert rb[0] == b"alpha"
        assert rb[1] == b"beta"
        assert rb[-1] == b"gamma"

        # Test contains
        assert b"alpha" in rb
        assert b"beta" in rb
        assert b"gamma" in rb
        assert b"missing" not in rb

    def test_overwrite_operations(self):
        """Test overwrite_latest functionality."""
        rb = BytesRingBuffer(3)
        rb.insert_batch([b"a", b"b", b"c"])

        # Test overwrite without incrementing count
        rb.overwrite_latest(b"new", increment_count=False)
        assert len(rb) == 3
        unwrapped = rb.unwrapped()
        assert unwrapped == [b"a", b"b", b"new"]

        # Test overwrite with incrementing count
        rb.overwrite_latest(b"newer", increment_count=True)
        # Capacity is 4, so we can have 4 elements
        unwrapped = rb.unwrapped()
        assert len(unwrapped) == 4
        assert unwrapped == [b"a", b"b", b"new", b"newer"]


class TestBytesRingBufferSpecialCases:
    """Test bytes-specific functionality."""

    def test_empty_bytes_handling(self):
        """Test handling of empty bytes objects."""
        rb = BytesRingBuffer(3)

        empty_data = [b"", b"non_empty", b""]
        rb.insert_batch(empty_data)

        unwrapped = rb.unwrapped()
        assert unwrapped == empty_data
        assert b"" in rb

    def test_large_bytes_objects(self):
        """Test with large bytes objects."""
        rb = BytesRingBuffer(3)

        large_bytes = [b"x" * 1000, b"y" * 5000, "large_unicode_" * 100 + "🚀" * 50]
        large_bytes[2] = large_bytes[2].encode("utf-8")

        rb.insert_batch(large_bytes)
        unwrapped = rb.unwrapped()
        assert len(unwrapped) == 3
        assert unwrapped[0] == b"x" * 1000
        assert unwrapped[1] == b"y" * 5000
        assert len(unwrapped[2]) > 1000  # Large unicode string

    def test_binary_data_handling(self):
        """Test handling of binary data."""
        rb = BytesRingBuffer(5)

        binary_data = [
            bytes([0, 1, 2, 3, 4]),
            bytes([255, 254, 253]),
            b"\x80\x81\x82",
            bytes(range(256))[:50],  # First 50 bytes
        ]

        rb.insert_batch(binary_data)
        unwrapped = rb.unwrapped()

        assert len(unwrapped) == 4
        for original, retrieved in zip(binary_data, unwrapped, strict=False):
            assert original == retrieved

    def test_string_encoding_compatibility(self):
        """Test compatibility with string encoding."""
        rb = BytesRingBuffer(5)

        # Test various encodings
        test_strings = ["hello", "café", "🚀🌟", "中文", "العربية"]

        for string in test_strings:
            encoded = string.encode("utf-8")
            rb.insert(encoded)

        unwrapped = rb.unwrapped()
        assert len(unwrapped) == 5

        # Verify we can decode back
        for original, stored in zip(test_strings, unwrapped, strict=False):
            assert stored.decode("utf-8") == original


class TestBytesRingBufferAsyncFunctionality:
    """Test async functionality for BytesRingBuffer."""

    @pytest.mark.asyncio
    async def test_async_consume_basic(self):
        """Test basic async consume functionality."""
        rb = BytesRingBuffer(5, disable_async=False)

        async def waiter():
            return await rb.aconsume()

        # Start waiting before inserting
        task = asyncio.create_task(waiter())
        await asyncio.sleep(0.01)

        # Insert item - should wake up the waiter
        rb.insert(b"async_test")
        result = await asyncio.wait_for(task, timeout=1.0)
        assert result == b"async_test"

    @pytest.mark.asyncio
    async def test_async_consume_iterable(self):
        """Test async consume iterable functionality."""
        rb = BytesRingBuffer(5, disable_async=False)

        collected = []

        async def producer():
            for i in range(3):
                await asyncio.sleep(0.01)
                rb.insert(f"bytes_{i}".encode())

        async def consumer():
            async for item in rb.aconsume_iterable():
                collected.append(item)
                if len(collected) == 3:
                    break

        # Run producer and consumer concurrently
        producer_task = asyncio.create_task(producer())
        consumer_task = asyncio.create_task(consumer())

        await asyncio.wait_for(
            asyncio.gather(producer_task, consumer_task), timeout=3.0
        )

        expected = [b"bytes_0", b"bytes_1", b"bytes_2"]
        assert collected == expected

    @pytest.mark.asyncio
    async def test_async_disabled_mode(self):
        """Test that async functions are disabled when disable_async=True."""
        rb = BytesRingBuffer(5, disable_async=True)

        with pytest.raises(RuntimeError, match="Async operations are disabled"):
            await rb.aconsume()

    @pytest.mark.asyncio
    async def test_async_consume_with_existing_data(self):
        """Test async consume when data already exists."""
        rb = BytesRingBuffer(5, disable_async=False)

        # Pre-populate with data
        rb.insert_batch([b"existing_1", b"existing_2"])

        # aconsume should immediately return without waiting
        result = await asyncio.wait_for(rb.aconsume(), timeout=0.1)
        assert result == b"existing_2"  # Should get the newest item


class TestBytesRingBufferEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_buffer_operations(self):
        """Test operations on empty buffer."""
        rb = BytesRingBuffer(5)

        with pytest.raises((IndexError, ValueError)):
            rb.consume()

        with pytest.raises(IndexError):
            _ = rb[0]

    def test_single_element_buffer(self):
        """Test buffer with capacity of 1."""
        rb = BytesRingBuffer(1)

        rb.insert(b"single")
        assert len(rb) == 1
        assert rb.peekleft() == b"single"
        assert rb.peekright() == b"single"

        # Inserting another should overwrite
        rb.insert(b"new")
        assert len(rb) == 1
        assert rb.peekleft() == b"new"

    def test_type_validation(self):
        """Test that non-bytes objects are rejected."""
        rb = BytesRingBuffer(3)

        # These should raise errors (bytes only)
        with pytest.raises((TypeError, ValueError)):
            rb.insert("string_not_bytes")

        with pytest.raises((TypeError, ValueError)):
            rb.insert(123)

        with pytest.raises((TypeError, ValueError)):
            rb.insert([1, 2, 3])
