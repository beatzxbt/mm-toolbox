"""Tests for IPC RingBuffer implementation."""

import asyncio
import os
import random
import string

import pytest

from mm_toolbox.ringbuffer.ipc import (
    IPCRingBufferConfig,
    IPCRingBufferConsumer,
    IPCRingBufferProducer,
)


def _random_ipc_path() -> str:
    """Generate unique IPC path to avoid test collisions."""
    suffix = "".join(random.choice(string.ascii_lowercase) for _ in range(8))
    return f"ipc:///tmp/mmtoolbox_ipc_{os.getpid()}_{suffix}"


class TestIPCRingBufferConfig:
    """Test IPCRingBufferConfig validation comprehensively."""

    def test_valid_config_creation(self):
        """Test creating valid configurations."""
        cfg = IPCRingBufferConfig(
            path=_random_ipc_path(), backlog=1024, num_producers=1, num_consumers=1
        )
        assert cfg.backlog == 1024
        assert cfg.num_producers == 1
        assert cfg.num_consumers == 1
        assert cfg.path.startswith("ipc:///tmp/mmtoolbox_ipc_")

    def test_path_validation(self):
        """Test path validation."""
        valid_path = _random_ipc_path()
        cfg = IPCRingBufferConfig(
            path=valid_path, backlog=1, num_producers=1, num_consumers=1
        )
        assert cfg.path == valid_path

        # Invalid paths should raise ValueError
        invalid_paths = ["tcp://invalid", "http://invalid", "invalid_path", ""]
        for invalid_path in invalid_paths:
            with pytest.raises(ValueError):
                IPCRingBufferConfig(
                    path=invalid_path, backlog=1, num_producers=1, num_consumers=1
                )

        # None path should raise AttributeError
        with pytest.raises(AttributeError):
            IPCRingBufferConfig(path=None, backlog=1, num_producers=1, num_consumers=1)

    def test_backlog_validation(self):
        """Test backlog validation."""
        valid_path = _random_ipc_path()

        # Valid backlogs
        for backlog in [1, 10, 1024]:
            cfg = IPCRingBufferConfig(
                path=valid_path, backlog=backlog, num_producers=1, num_consumers=1
            )
            assert cfg.backlog == backlog

        # Invalid backlogs
        for invalid_backlog in [0, -1]:
            with pytest.raises(ValueError):
                IPCRingBufferConfig(
                    path=valid_path,
                    backlog=invalid_backlog,
                    num_producers=1,
                    num_consumers=1,
                )

    def test_producer_consumer_count_validation(self):
        """Test producer and consumer count validation."""
        valid_path = _random_ipc_path()

        # Valid SPSC configuration
        cfg = IPCRingBufferConfig(
            path=valid_path, backlog=1024, num_producers=1, num_consumers=1
        )
        assert cfg.num_producers == 1
        assert cfg.num_consumers == 1

        # Invalid: zero or negative counts
        invalid_configs = [(0, 1), (1, 0), (-1, 1), (1, -1)]
        for num_prod, num_cons in invalid_configs:
            with pytest.raises(ValueError):
                IPCRingBufferConfig(
                    path=valid_path,
                    backlog=1024,
                    num_producers=num_prod,
                    num_consumers=num_cons,
                )

        # Invalid: MPMC not supported
        with pytest.raises(ValueError, match="MPMC is not supported"):
            IPCRingBufferConfig(
                path=valid_path, backlog=1024, num_producers=2, num_consumers=2
            )


class TestIPCRingBufferBasicOperations:
    """Test basic IPC ringbuffer operations."""

    def test_producer_consumer_creation(self):
        """Test creating producer and consumer."""
        cfg = IPCRingBufferConfig(
            path=_random_ipc_path(), backlog=1024, num_producers=1, num_consumers=1
        )
        producer = IPCRingBufferProducer(cfg)
        consumer = IPCRingBufferConsumer(cfg)

        try:
            assert producer is not None
            assert consumer is not None
        finally:
            producer.stop()
            consumer.stop()

    def test_single_message_roundtrip(self):
        """Test sending and receiving a single message."""
        cfg = IPCRingBufferConfig(
            path=_random_ipc_path(), backlog=1024, num_producers=1, num_consumers=1
        )
        producer = IPCRingBufferProducer(cfg)
        consumer = IPCRingBufferConsumer(cfg)

        try:
            test_message = b"hello_world"
            producer.insert(test_message)
            received = consumer.consume()
            assert received == test_message
        finally:
            producer.stop()
            consumer.stop()

    def test_multiple_messages_roundtrip(self):
        """Test sending and receiving multiple messages."""
        cfg = IPCRingBufferConfig(
            path=_random_ipc_path(), backlog=1024, num_producers=1, num_consumers=1
        )
        producer = IPCRingBufferProducer(cfg)
        consumer = IPCRingBufferConsumer(cfg)

        try:
            messages = [b"msg1", b"msg2", b"msg3"]
            for msg in messages:
                producer.insert(msg)

            received = []
            for _ in messages:
                received.append(consumer.consume())

            assert received == messages
        finally:
            producer.stop()
            consumer.stop()

    def test_batch_operations(self):
        """Test batch insert and consume operations."""
        cfg = IPCRingBufferConfig(
            path=_random_ipc_path(), backlog=1024, num_producers=1, num_consumers=1
        )
        producer = IPCRingBufferProducer(cfg)
        consumer = IPCRingBufferConsumer(cfg)

        try:
            batch_messages = [b"batch1", b"batch2", b"batch3"]
            producer.insert_batch(batch_messages)

            received = consumer.consume_all()
            if not received:
                received = [consumer.consume() for _ in batch_messages]

            assert received == batch_messages
        finally:
            producer.stop()
            consumer.stop()

    def test_packed_operations(self):
        """Test packed insert and consume operations."""
        cfg = IPCRingBufferConfig(
            path=_random_ipc_path(), backlog=1024, num_producers=1, num_consumers=1
        )
        producer = IPCRingBufferProducer(cfg)
        consumer = IPCRingBufferConsumer(cfg)

        try:
            packed_data = [b"pack1", b"pack2", b"pack3"]
            producer.insert_packed(packed_data)
            received_packed = consumer.consume_packed()
            assert received_packed == packed_data
        finally:
            producer.stop()
            consumer.stop()


class TestIPCRingBufferDataTypes:
    """Test various data types and edge cases."""

    def test_empty_messages(self):
        """Test handling of empty byte messages."""
        cfg = IPCRingBufferConfig(
            path=_random_ipc_path(), backlog=1024, num_producers=1, num_consumers=1
        )
        producer = IPCRingBufferProducer(cfg)
        consumer = IPCRingBufferConsumer(cfg)

        try:
            # Test empty bytes
            producer.insert(b"")
            received = consumer.consume()
            assert received == b""
        finally:
            producer.stop()
            consumer.stop()

    def test_large_messages(self):
        """Test handling of reasonably large messages."""
        cfg = IPCRingBufferConfig(
            path=_random_ipc_path(), backlog=1024, num_producers=1, num_consumers=1
        )
        producer = IPCRingBufferProducer(cfg)
        consumer = IPCRingBufferConsumer(cfg)

        try:
            # Test moderately large message (not too big to avoid slowness)
            large_msg = b"x" * 1024  # 1KB
            producer.insert(large_msg)
            received = consumer.consume()
            assert received == large_msg
            assert len(received) == len(large_msg)
        finally:
            producer.stop()
            consumer.stop()

    def test_binary_data(self):
        """Test handling of binary data."""
        cfg = IPCRingBufferConfig(
            path=_random_ipc_path(), backlog=1024, num_producers=1, num_consumers=1
        )
        producer = IPCRingBufferProducer(cfg)
        consumer = IPCRingBufferConsumer(cfg)

        try:
            # Test binary data
            binary_data = [
                bytes([0, 1, 2, 3, 4]),
                bytes([255, 254, 253]),
                b"\x00\x01\x80\xff",
            ]

            for binary_msg in binary_data:
                producer.insert(binary_msg)
                received = consumer.consume()
                assert received == binary_msg
        finally:
            producer.stop()
            consumer.stop()

    def test_unicode_encoded_data(self):
        """Test handling of unicode strings encoded as bytes."""
        cfg = IPCRingBufferConfig(
            path=_random_ipc_path(), backlog=1024, num_producers=1, num_consumers=1
        )
        producer = IPCRingBufferProducer(cfg)
        consumer = IPCRingBufferConsumer(cfg)

        try:
            unicode_strings = ["hello", "café", "🚀"]  # Reduced set for speed

            for unicode_str in unicode_strings:
                encoded = unicode_str.encode("utf-8")
                producer.insert(encoded)
                received = consumer.consume()
                assert received == encoded
                assert received.decode("utf-8") == unicode_str
        finally:
            producer.stop()
            consumer.stop()


class TestIPCRingBufferErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_message_types(self):
        """Test that invalid message types are rejected."""
        cfg = IPCRingBufferConfig(
            path=_random_ipc_path(), backlog=1024, num_producers=1, num_consumers=1
        )
        producer = IPCRingBufferProducer(cfg)

        try:
            # Test invalid types (should only accept bytes)
            invalid_messages = ["string_not_bytes", 123, [1, 2, 3]]

            for invalid_msg in invalid_messages:
                with pytest.raises((TypeError, ValueError)):
                    producer.insert(invalid_msg)
        finally:
            producer.stop()

    def test_consumer_without_producer(self):
        """Test consumer behavior when no producer is running."""
        cfg = IPCRingBufferConfig(
            path=_random_ipc_path(), backlog=1024, num_producers=1, num_consumers=1
        )
        consumer = IPCRingBufferConsumer(cfg)

        try:
            # Consuming from empty should handle gracefully
            result = consumer.consume_all()
            assert result == [] or result is None
        finally:
            consumer.stop()

    def test_double_stop_safety(self):
        """Test that calling stop() multiple times is safe."""
        cfg = IPCRingBufferConfig(
            path=_random_ipc_path(), backlog=1024, num_producers=1, num_consumers=1
        )
        producer = IPCRingBufferProducer(cfg)
        consumer = IPCRingBufferConsumer(cfg)

        # Multiple stops should not crash
        producer.stop()
        producer.stop()  # Should be safe

        consumer.stop()
        consumer.stop()  # Should be safe


class TestIPCRingBufferAsyncOperations:
    """Test async operations (simplified for speed)."""

    @pytest.mark.asyncio
    async def test_async_single_message(self):
        """Test async single message operations."""
        cfg = IPCRingBufferConfig(
            path=_random_ipc_path(), backlog=1024, num_producers=1, num_consumers=1
        )

        producer = IPCRingBufferProducer(cfg)
        consumer = IPCRingBufferConsumer(cfg)

        try:
            producer.insert(b"async_test")
            result = await asyncio.wait_for(consumer.aconsume(), timeout=1.0)
            assert result == b"async_test"
        finally:
            producer.stop()
            consumer.stop()

    @pytest.mark.asyncio
    async def test_async_packed_operations(self):
        """Test async packed operations."""
        cfg = IPCRingBufferConfig(
            path=_random_ipc_path(), backlog=1024, num_producers=1, num_consumers=1
        )

        producer = IPCRingBufferProducer(cfg)
        consumer = IPCRingBufferConsumer(cfg)

        try:
            packed_data = [b"async_pack1", b"async_pack2"]
            producer.insert_packed(packed_data)

            result = await asyncio.wait_for(consumer.aconsume_packed(), timeout=1.0)
            assert result == packed_data
        finally:
            producer.stop()
            consumer.stop()

    @pytest.mark.asyncio
    async def test_async_timeout_behavior(self):
        """Test async timeout behavior."""
        cfg = IPCRingBufferConfig(
            path=_random_ipc_path(), backlog=1024, num_producers=1, num_consumers=1
        )

        consumer = IPCRingBufferConsumer(cfg)

        try:
            # Test that aconsume times out when no data available
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(consumer.aconsume(), timeout=0.1)
        finally:
            consumer.stop()
