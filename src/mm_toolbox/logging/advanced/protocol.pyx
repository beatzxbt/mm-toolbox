"""Simple binary protocol implementation for logging."""

from libc.stdlib cimport malloc, free, realloc
from libc.string cimport memcpy
from libc.stdint cimport (
    uint8_t as u8, 
    uint16_t as u16, 
    uint32_t as u32, 
    uint64_t as u64
)
from cpython.bytes cimport PyBytes_FromStringAndSize

from mm_toolbox.logging.advanced.protocol cimport MessageType, InternalMessage


cdef inline InternalMessage create_internal_message(MessageType type, u64 timestamp_ns, u32 len, unsigned char* data) noexcept nogil:
    """Create an internal message struct.

    Args:
        type (MessageType): Message classification.
        timestamp_ns (u64): Nanosecond timestamp.
        len (u32): Byte length of the payload.
        data (unsigned char*): Pointer to the payload data.

    Returns:
        InternalMessage: Populated message struct.
    """
    cdef InternalMessage message
    message.type = type
    message.timestamp_ns = timestamp_ns
    message.len = len
    message.data = data
    return message

cdef bytes internal_message_to_bytes(InternalMessage message):
    """Serialize an InternalMessage to bytes.

    Args:
        message (InternalMessage): Message to serialize.

    Returns:
        bytes: Serialized binary payload.
    """
    cdef BinaryWriter writer = BinaryWriter()
    writer.write_u8(<u8>message.type)
    writer.write_u64(message.timestamp_ns)
    writer.write_u32(message.len)
    writer.write_chars(message.data, message.len)
    return writer.finalize()

cdef InternalMessage bytes_to_internal_message(bytes message):
    """Parse bytes into an InternalMessage with owned data.

    Args:
        message (bytes): Serialized message payload.

    Returns:
        InternalMessage: Parsed message with heap-owned data.

    """
    cdef:
        BinaryReader    reader = BinaryReader(message)
        MessageType     msg_type = <MessageType>reader.read_u8()
        u64             timestamp_ns = reader.read_u64()
        u32             data_len = reader.read_u32()
        unsigned char*  data = NULL
        unsigned char*  source = NULL
    if data_len > 0:
        source = reader.read_chars(data_len)
        data = <unsigned char*>malloc(data_len * sizeof(unsigned char))
        if not data:
            raise MemoryError("Failed to allocate memory for InternalMessage data")
        memcpy(data, source, data_len)
    return create_internal_message(msg_type, timestamp_ns, data_len, data)

cdef void free_internal_message_data(unsigned char* data) noexcept nogil:
    """Release heap memory allocated for InternalMessage.data.

    Args:
        data (unsigned char*): Data pointer to free.
    """
    if data != NULL:
        free(data)

cdef class BinaryWriter:
    """Fast, type-safe binary serializer."""
    
    def __cinit__(self, u32 initial_capacity = 1024):
        """Initialize the binary serializer with a backing buffer.

        Args:
            initial_capacity (u32): Starting buffer size in bytes.

        Raises:
            MemoryError: If the backing buffer cannot be allocated.
        """
        self._capacity = initial_capacity
        self._buffer = <unsigned char*>malloc(initial_capacity * sizeof(unsigned char))
        if not self._buffer:
            raise MemoryError("Failed to allocate memory for BinaryWriter")
        self._pos = 0
    
    def __dealloc__(self):
        free(self._buffer)
    
    cdef void _ensure_capacity(self, u32 needed):
        """Grow the internal buffer to accommodate ``needed`` additional bytes.

        Args:
            needed (u32): Number of bytes required.

        Raises:
            MemoryError: If reallocation fails.
            ValueError: If capacity would overflow.
        """
        cdef u32 new_size
        cdef unsigned char* new_buffer
        
        # Check for overflow in _pos + needed
        if needed > <u32>(0xFFFFFFFF) - self._pos:
            raise ValueError("Buffer size would overflow u32")
        
        if self._pos + needed > self._capacity:
            new_size = max(self._capacity * 2, self._pos + needed)
            # Check for overflow in new_size calculation
            if new_size < self._capacity or new_size < needed:
                raise ValueError("Buffer capacity overflow")
            new_buffer = <unsigned char*>realloc(
                self._buffer, new_size * sizeof(unsigned char)
            )
            if not new_buffer:
                raise MemoryError("Failed to reallocate memory for BinaryWriter")
            self._buffer = new_buffer
            self._capacity = new_size
    
    cdef inline u32 length(self) nogil:
        """Return the number of bytes written so far.

        Returns:
            u32: Current write position (length).
        """
        return self._pos
    
    cdef void write_u8(self, u8 value):
        """Write an unsigned 8-bit integer.

        Args:
            value (u8): Value to write.
        """
        self._ensure_capacity(1)
        (<u8*>&self._buffer[self._pos])[0] = value
        self._pos += 1
    
    cdef void write_u16(self, u16 value):
        """Write an unsigned 16-bit integer.

        Args:
            value (u16): Value to write.
        """
        self._ensure_capacity(2)
        (<u16*>&self._buffer[self._pos])[0] = value
        self._pos += 2
    
    cdef void write_u32(self, u32 value):
        """Write an unsigned 32-bit integer.

        Args:
            value (u32): Value to write.
        """
        self._ensure_capacity(4)
        (<u32*>&self._buffer[self._pos])[0] = value
        self._pos += 4
    
    cdef void write_u64(self, u64 value):
        """Write an unsigned 64-bit integer.

        Args:
            value (u64): Value to write.
        """
        self._ensure_capacity(8)
        (<u64*>&self._buffer[self._pos])[0] = value
        self._pos += 8
    
    cdef void write_bytes(self, bytes data):
        """Write a Python bytes object.

        Args:
            data (bytes): Payload to write.
        """
        cdef u32 length = len(data)
        self._ensure_capacity(length)
        cdef const unsigned char[:] data_view = data    # type: ignore
        memcpy(&self._buffer[self._pos], &data_view[0], length)
        self._pos += length

    cdef void write_chars(self, unsigned char* data, u32 length):
        """Write raw bytes from a C pointer.

        Args:
            data (unsigned char*): Source pointer.
            length (u32): Number of bytes to copy.
        """
        self._ensure_capacity(length)
        memcpy(&self._buffer[self._pos], data, length)
        self._pos += length
    
    cdef bytes finalize(self):
        """Return the serialized bytes and reset the write position.

        Returns:
            bytes: Snapshot of the buffer contents.
        """
        cdef bytes result = PyBytes_FromStringAndSize(<char*>self._buffer, self._pos)
        self._pos = 0
        return result

    cdef (unsigned char*, u32) finalize_to_chars(self) nogil:
        """Return the buffer pointer and length, and reset the write position.

        Returns:
            tuple(unsigned char*, u32): Buffer pointer and current length.
        """
        cdef unsigned char* ptr = self._buffer
        cdef u32 len = self._pos
        self._pos = 0
        return (ptr, len)
    
    cdef inline void reset(self) nogil:
        """Reset the write position without returning data."""
        self._pos = 0


cdef class BinaryReader:
    """Fast, type-safe binary deserializer."""
    
    def __cinit__(self, bytes buffer):
        """Initialize the binary deserializer.

        Args:
            buffer (bytes): Byte string to read from.
        """
        self._buffer = buffer
        self._buf_view = buffer
        self._len = len(buffer)
        self._pos = 0
    
    cdef u8 read_u8(self):
        """Read an unsigned 8-bit integer.

        Returns:
            u8: The read value.

        Raises:
            ValueError: On buffer underrun.
        """
        cdef u8 value
        if self._pos + 1 > self._len:
            raise ValueError("Buffer underrun reading u8")
        value = <u8>self._buf_view[self._pos]
        self._pos += 1
        return value

    cdef u16 read_u16(self):
        """Read an unsigned 16-bit integer.

        Returns:
            u16: The read value.

        Raises:
            ValueError: On buffer underrun.
        """
        cdef u16 value
        if self._pos + 2 > self._len:
            raise ValueError("Buffer underrun reading u16")
        memcpy(&value, &self._buf_view[self._pos], sizeof(u16))
        self._pos += 2
        return value
    
    cdef u32 read_u32(self):
        """Read an unsigned 32-bit integer.

        Returns:
            u32: The read value.

        Raises:
            ValueError: On buffer underrun.
        """
        cdef u32 value
        if self._pos + 4 > self._len:
            raise ValueError("Buffer underrun reading u32")
        memcpy(&value, &self._buf_view[self._pos], sizeof(u32))
        self._pos += 4
        return value
    
    cdef u64 read_u64(self):
        """Read an unsigned 64-bit integer.

        Returns:
            u64: The read value.

        Raises:
            ValueError: On buffer underrun.
        """
        cdef u64 value
        if self._pos + 8 > self._len:
            raise ValueError("Buffer underrun reading u64")
        memcpy(&value, &self._buf_view[self._pos], sizeof(u64))
        self._pos += 8
        return value
    
    cdef bytes read_bytes(self, u32 length):
        """Read a slice of bytes.

        Args:
            length (u32): Number of bytes to read.

        Returns:
            bytes: The read slice.

        Raises:
            ValueError: On buffer underrun.
        """
        cdef bytes result
        if self._pos + length > self._len:
            raise ValueError("Buffer underrun reading bytes")
        result = self._buffer[self._pos:self._pos + length]
        self._pos += length
        return result

    cdef unsigned char* read_chars(self, u32 length):
        """Read a raw pointer into the buffer.

        Args:
            length (u32): Number of bytes to expose.

        Returns:
            unsigned char*: Pointer at the current read position.

        Raises:
            ValueError: On buffer underrun.
        """
        cdef unsigned char* result
        if self._pos + length > self._len:
            raise ValueError("Buffer underrun reading bytes")
        result = <unsigned char*>&self._buf_view[self._pos]
        self._pos += length
        return result
