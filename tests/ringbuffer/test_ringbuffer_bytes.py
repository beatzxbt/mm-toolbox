"""Tests for BytesRingBuffer implementation."""

import asyncio

import pytest

from mm_toolbox.ringbuffer.bytes import BytesRingBuffer, BytesRingBufferFast


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

        # Test consume (should return oldest)
        consumed = rb.consume()
        assert consumed == b"first"
        assert len(rb) == 2

        # Test consume_all
        remaining = rb.consume_all()
        assert remaining == [b"second", b"third"]
        assert rb.is_empty()

    def test_contains_operations(self):
        """Test contains operations."""
        rb = BytesRingBuffer(5)
        data = [b"alpha", b"beta", b"gamma"]
        rb.insert_batch(data)

        # Test contains
        assert b"alpha" in rb
        assert b"beta" in rb
        assert b"gamma" in rb
        assert b"missing" not in rb


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


class TestBytesRingBufferFast:
    """Test BytesRingBufferFast behavior."""

    def test_consume_is_fifo(self):
        """Consume should remove the oldest item first."""
        rb = BytesRingBufferFast(max_capacity=4, disable_async=True)
        rb.insert_batch([b"first", b"second", b"third"])

        assert rb.consume() == b"first"
        assert rb.consume() == b"second"
        assert rb.consume() == b"third"

    def test_oversize_items_raise(self):
        """Oversized inserts should raise instead of truncating."""
        rb = BytesRingBufferFast(
            max_capacity=4,
            expected_item_size=4,
            buffer_percent=0.0,
            disable_async=True,
        )

        with pytest.raises(ValueError, match="exceeds slot size"):
            rb.insert(b"12345")

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
        assert result == b"existing_1"  # Should get the oldest item

    @pytest.mark.asyncio
    async def test_fast_async_consume_with_existing_data_fifo(self):
        """Fast async consume should return oldest item first."""
        rb = BytesRingBufferFast(max_capacity=4, disable_async=False)
        rb.insert_batch([b"existing_1", b"existing_2"])

        result = await asyncio.wait_for(rb.aconsume(), timeout=0.1)
        assert result == b"existing_1"


class TestBytesRingBufferInsertChar:
    """Test insert_char on BytesRingBuffer and BytesRingBufferFast."""

    def test_insert_char_roundtrip_bytes_ringbuffer(self):
        rb = BytesRingBuffer(4)
        data = b"hello"
        rb.insert_char(data, len(data))
        assert rb.consume() == b"hello"

    def test_insert_char_empty_bytes_ringbuffer(self):
        rb = BytesRingBuffer(4)
        rb.insert_char(b"", 0)
        assert rb.consume() == b""

    def test_insert_char_large_data_bytes_ringbuffer(self):
        rb = BytesRingBuffer(4)
        data = b"x" * 1000
        rb.insert_char(data, len(data))
        assert rb.consume() == data

    def test_insert_char_roundtrip_fast(self):
        rb = BytesRingBufferFast(4, expected_item_size=128, disable_async=True)
        data = b"hello"
        rb.insert_char(data, len(data))
        assert rb.consume() == b"hello"

    def test_insert_char_empty_fast(self):
        rb = BytesRingBufferFast(4, expected_item_size=16, disable_async=True)
        rb.insert_char(b"", 0)
        assert rb.consume() == b""

    def test_insert_char_large_data_fast(self):
        rb = BytesRingBufferFast(4, expected_item_size=1024, disable_async=True)
        data = b"x" * 1000
        rb.insert_char(data, len(data))
        assert rb.consume() == data


class TestBytesRingBufferConsumeInto:
    """Test consume_into on BytesRingBuffer and BytesRingBufferFast."""

    def test_consume_into_basic_bytes_ringbuffer(self):
        rb = BytesRingBuffer(4)
        rb.insert(b"hello")
        dst = bytearray(10)
        n = rb.consume_into(dst)
        assert n == 5
        assert dst[:5] == b"hello"

    def test_consume_into_buffer_too_small_bytes_ringbuffer(self):
        rb = BytesRingBuffer(4)
        rb.insert(b"hello")
        dst = bytearray(3)
        with pytest.raises(ValueError, match="too small"):
            rb.consume_into(dst)

    def test_consume_into_basic_fast(self):
        rb = BytesRingBufferFast(4, expected_item_size=128, disable_async=True)
        rb.insert(b"hello")
        dst = bytearray(10)
        n = rb.consume_into(dst)
        assert n == 5
        assert dst[:5] == b"hello"

    def test_consume_into_buffer_too_small_fast(self):
        rb = BytesRingBufferFast(4, expected_item_size=128, disable_async=True)
        rb.insert(b"hello")
        dst = bytearray(3)
        with pytest.raises(ValueError, match="too small"):
            rb.consume_into(dst)


class TestBytesRingBufferConsumeAllInto:
    """Test consume_all_into on BytesRingBuffer and BytesRingBufferFast."""

    def test_consume_all_into_basic_bytes_ringbuffer(self):
        rb = BytesRingBuffer(4)
        rb.insert_batch([b"hello", b"world"])
        buffers = [bytearray(10), bytearray(10)]
        n = rb.consume_all_into(buffers)
        assert n == 2
        assert buffers[0][:5] == b"hello"
        assert buffers[1][:5] == b"world"

    def test_consume_all_into_buffer_too_small_bytes_ringbuffer(self):
        rb = BytesRingBuffer(4)
        rb.insert_batch([b"hello", b"world"])
        buffers = [bytearray(10), bytearray(3)]
        with pytest.raises(ValueError, match="too small"):
            rb.consume_all_into(buffers)

    def test_consume_all_into_more_buffers_than_items_bytes_ringbuffer(self):
        rb = BytesRingBuffer(4)
        rb.insert_batch([b"hello"])
        buffers = [bytearray(10), bytearray(10), bytearray(10)]
        n = rb.consume_all_into(buffers)
        assert n == 1
        assert buffers[0][:5] == b"hello"

    def test_consume_all_into_basic_fast(self):
        rb = BytesRingBufferFast(4, expected_item_size=128, disable_async=True)
        rb.insert_batch([b"hello", b"world"])
        buffers = [bytearray(10), bytearray(10)]
        n = rb.consume_all_into(buffers)
        assert n == 2
        assert buffers[0][:5] == b"hello"
        assert buffers[1][:5] == b"world"

    def test_consume_all_into_buffer_too_small_fast(self):
        rb = BytesRingBufferFast(4, expected_item_size=128, disable_async=True)
        rb.insert_batch([b"hello", b"world"])
        buffers = [bytearray(10), bytearray(3)]
        with pytest.raises(ValueError, match="too small"):
            rb.consume_all_into(buffers)

    def test_consume_all_into_more_buffers_than_items_fast(self):
        rb = BytesRingBufferFast(4, expected_item_size=128, disable_async=True)
        rb.insert_batch([b"hello"])
        buffers = [bytearray(10), bytearray(10), bytearray(10)]
        n = rb.consume_all_into(buffers)
        assert n == 1
        assert buffers[0][:5] == b"hello"


class TestBytesRingBufferEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_buffer_operations(self):
        """Test operations on empty buffer."""
        rb = BytesRingBuffer(5)

        with pytest.raises((IndexError, ValueError)):
            rb.consume()

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

    def test_insert_returns_bool(self):
        """Test that insert returns True."""
        rb = BytesRingBuffer(4)
        assert rb.insert(b"hello") is True
        assert rb.insert_batch([b"a", b"b"]) is True
        assert len(rb) == 3


class TestBytesRingBufferTimestamps:
    """Test timestamp tracking on monolithic bytes ringbuffers."""

    def test_latest_insert_time_ns_initial(self):
        rb = BytesRingBuffer(4)
        assert rb.latest_insert_time_ns == 0

    def test_latest_insert_time_ns_updated_on_insert(self):
        rb = BytesRingBuffer(4)
        rb.insert(b"a")
        t1 = rb.latest_insert_time_ns
        assert t1 > 0
        import time

        time.sleep(0.001)
        rb.insert(b"b")
        t2 = rb.latest_insert_time_ns
        assert t2 > t1

    def test_latest_insert_time_ns_updated_on_insert_batch(self):
        rb = BytesRingBuffer(4)
        rb.insert_batch([b"a", b"b"])
        assert rb.latest_insert_time_ns > 0

    def test_latest_consume_time_ns_initial(self):
        rb = BytesRingBuffer(4)
        assert rb.latest_consume_time_ns == 0

    def test_latest_consume_time_ns_updated_on_consume(self):
        rb = BytesRingBuffer(4)
        rb.insert_batch([b"a", b"b"])
        t1 = rb.latest_insert_time_ns
        rb.consume()
        t2 = rb.latest_consume_time_ns
        assert t2 > 0
        assert t2 >= t1


class TestBytesRingBufferFastTimestamps:
    """Test timestamp tracking on fast bytes ringbuffers."""

    def test_latest_insert_time_ns_initial(self):
        rb = BytesRingBufferFast(4)
        assert rb.latest_insert_time_ns == 0

    def test_latest_insert_time_ns_updated_on_insert(self):
        rb = BytesRingBufferFast(4)
        rb.insert(b"a")
        t1 = rb.latest_insert_time_ns
        assert t1 > 0
        import time

        time.sleep(0.001)
        rb.insert(b"b")
        t2 = rb.latest_insert_time_ns
        assert t2 > t1

    def test_latest_consume_time_ns_initial(self):
        rb = BytesRingBufferFast(4)
        assert rb.latest_consume_time_ns == 0

    def test_latest_consume_time_ns_updated_on_consume(self):
        rb = BytesRingBufferFast(4)
        rb.insert_batch([b"a", b"b"])
        t1 = rb.latest_insert_time_ns
        rb.consume()
        t2 = rb.latest_consume_time_ns
        assert t2 > 0
        assert t2 >= t1
