# worker_logger.pyx
import os
import time
import threading

from libc.stdint cimport (
    uint8_t as u8, 
    uint32_t as u32, 
    uint64_t as u64,
)

from mm_toolbox.time.time cimport time_ns
from mm_toolbox.ringbuffer.ipc import IPCRingBufferProducer, IPCRingBufferConfig

from mm_toolbox.logging.advanced.log cimport CLogLevel
from mm_toolbox.logging.advanced.protocol cimport (
    BinaryWriter, 
    MessageType, 
    InternalMessage, 
    create_internal_message, 
    internal_message_to_bytes
)

from mm_toolbox.logging.advanced.config cimport LoggerConfig


cdef class WorkerLogger:
    """A lightweight worker logger that sends log messages to the master logger."""

    def __cinit__(
        self, 
        LoggerConfig config=None, 
        str name=None,
    ):
        self._config = config if config else LoggerConfig() 

        self._name = name if name else f"WORKER{os.getpid()}"
        self._name = self._name.encode('utf-8')
        self._len_name = len(self._name)
        self._name_as_chars = <unsigned char*>self._name
        
        self._num_pending_logs = 0
        self._batch_writer = BinaryWriter(initial_capacity=1*1024*1024)  # 1MB baseline

        self._transport = IPCRingBufferProducer(
            IPCRingBufferConfig(
                path=self._config.path,
                backlog=10000,  
                num_producers=2,  # >1 workers to indicate MPSC
                num_consumers=1   # Single master
            )
        )
        
        self._is_running = True
        
        self._timed_operations_thread = threading.Thread(
            target=self._timed_operations,
            daemon=True
        )
        self._timed_operations_thread.start()

        self.info(f"WorkerLogger started; name: {self._name.decode()}")

    cpdef void _timed_operations(self):
        """Background processing loop."""
        while self._is_running:
            time.sleep(self._config.flush_interval_s)
            if self._num_pending_logs > 0:
                self._flush_logs()
            
    cdef void _flush_logs(self):
        """Flush pending logs."""
        if self._num_pending_logs == 0:
            return
        
        cdef BinaryWriter data_writer = BinaryWriter(8 + self._len_name + self._batch_writer.length())
        data_writer.write_u32(self._len_name)
        data_writer.write_chars(self._name_as_chars, self._len_name)
        data_writer.write_u32(self._num_pending_logs)
        data_writer.write_chars(self._batch_writer._buffer, self._batch_writer.length())
        
        cdef unsigned char* data_ptr
        cdef u32 data_len
        data_ptr, data_len = data_writer.finalize_to_chars()

        cdef InternalMessage internal_message = create_internal_message(
            MessageType.LOG, 
            time_ns(),
            data_len, 
            data_ptr
        )
        self._transport.insert(internal_message_to_bytes(internal_message))

        self._batch_writer.reset()
        self._num_pending_logs = 0
    
    cdef void _add_log_to_batch(self, CLogLevel clevel, u32 message_len, unsigned char* message):
        """Add a log to the batch."""
        cdef u64 time_now_ns = time_ns()
        self._batch_writer.write_u64(time_now_ns)
        
        self._batch_writer.write_u32(self._len_name)
        self._batch_writer.write_chars(self._name_as_chars, self._len_name)
        
        self._batch_writer.write_u8(<u8>clevel)
        
        self._batch_writer.write_u32(message_len)
        self._batch_writer.write_chars(message, message_len)
        
        self._num_pending_logs += 1

    cpdef void trace(self, str msg_str=None, bytes msg_bytes=b""):
        """Send a trace-level log message."""
        if self._is_running and self._config.base_level <= CLogLevel.TRACE:
            message = msg_str.encode('utf-8') if msg_str else msg_bytes
            self._add_log_to_batch(CLogLevel.TRACE, len(message), <unsigned char*>message)

    cpdef void debug(self, str msg_str=None, bytes msg_bytes=b""):
        """Send a debug-level log message."""
        if self._is_running and self._config.base_level <= CLogLevel.DEBUG:
            message = msg_str.encode('utf-8') if msg_str else msg_bytes
            self._add_log_to_batch(CLogLevel.DEBUG, len(message), <unsigned char*>message)
    
    cpdef void info(self, str msg_str=None, bytes msg_bytes=b""):
        """Send an info-level log message."""
        if self._is_running and self._config.base_level <= CLogLevel.INFO:
            message = msg_str.encode('utf-8') if msg_str else msg_bytes
            self._add_log_to_batch(CLogLevel.INFO, len(message), <unsigned char*>message)
    
    cpdef void warning(self, str msg_str=None, bytes msg_bytes=b""):
        """Send a warning-level log message."""
        if self._is_running and self._config.base_level <= CLogLevel.WARNING:
            message = msg_str.encode('utf-8') if msg_str else msg_bytes
            self._add_log_to_batch(CLogLevel.WARNING, len(message), <unsigned char*>message)
    
    cpdef void error(self, str msg_str=None, bytes msg_bytes=b""):
        """Send an error-level log message."""
        if self._is_running and self._config.base_level <= CLogLevel.ERROR:
            message = msg_str.encode('utf-8') if msg_str else msg_bytes
            self._add_log_to_batch(CLogLevel.ERROR, len(message), <unsigned char*>message)

    cpdef void shutdown(self):
        """Shutdownwith proper cleanup."""
        if not self._is_running:
            return
        
        self.warning(f"Shutting down worker logger; name: {self._name.decode()}")
        self._is_running = False
        
        # Final flush, having kept self._is_running = True until this 
        # point ensures that no more logs will be added into the batch.
        self._flush_logs()
        
        # Wait for thread and cleanup
        self._timed_operations_thread.join()
        self._transport.stop()

    cpdef bint is_running(self):
        """
        Check if the logger is running.
        """
        return self._is_running
    
    cpdef str get_name(self):
        """
        Get the name of the logger.
        """
        return self._name.decode()

    cpdef object get_config(self):
        """
        Get the configuration of the logger.
        """
        return self._config