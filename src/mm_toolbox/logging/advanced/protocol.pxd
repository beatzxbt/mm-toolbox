"""Binary protocol definitions for advanced logging."""

from libc.stdint cimport (
    uint8_t as u8, 
    uint16_t as u16, 
    uint32_t as u32, 
    uint64_t as u64
)

cdef enum MessageType:
    LOG
    HEARTBEAT # Reserved
    DATA # Reserved

cdef struct InternalMessage:
    MessageType type
    u64         timestamp_ns
    u32         len
    unsigned char* data

cdef InternalMessage create_internal_message(MessageType type, u64 timestamp_ns, u32 len, unsigned char* data) noexcept nogil
cdef bytes internal_message_to_bytes(InternalMessage message)
cdef InternalMessage bytes_to_internal_message(bytes message)

cdef class BinaryWriter:
    cdef:
        unsigned char* _buffer
        u32 _capacity
        u32 _pos
    
    cdef void   _ensure_capacity(self, u32 needed)
    cdef inline u32 length(self) nogil
    cdef void   write_u8(self, u8 value)
    cdef void   write_u16(self, u16 value)
    cdef void   write_u32(self, u32 value)
    cdef void   write_u64(self, u64 value)
    cdef void   write_bytes(self, bytes data)
    cdef void   write_chars(self, unsigned char* data, u32 length)
    cdef bytes  finalize(self)
    cdef (unsigned char*, u32) finalize_to_chars(self) nogil
    cdef inline void   reset(self) nogil

cdef class BinaryReader:
    cdef: 
        bytes _buffer
        const unsigned char[:] _buf_view
        u32 _pos
        u32 _len
    
    cdef u8     read_u8(self)
    cdef u16    read_u16(self)
    cdef u32    read_u32(self)
    cdef u64    read_u64(self)
    cdef bytes  read_bytes(self, u32 length)
    cdef unsigned char* read_chars(self, u32 length)
