import unittest
import os
import asyncio
from io import StringIO
from contextlib import redirect_stdout
from mm_toolbox.logging.standard import (
    Logger,
    LogLevel,
    LoggerConfig,
    FileLogHandler,
)


class TestLoggerConfig(unittest.TestCase):
    def test_valid_config(self):
        config = LoggerConfig(
            base_level=LogLevel.WARNING, 
            buffer_capacity=20, 
            buffer_timeout_s=15, 
            do_stout=True, 
            str_format="%(asctime)s - %(levelname)s - %(message)s"
        )
        
    def test_config_values(self):
        """Test that config values are correctly set and retrieved."""
        config = LoggerConfig(
            base_level=LogLevel.DEBUG,
            buffer_capacity=100,
            buffer_timeout_s=30,
            do_stout=False,
            str_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        self.assertEqual(config.base_level, LogLevel.DEBUG)
        self.assertEqual(config.buffer_capacity, 100)
        self.assertEqual(config.buffer_timeout_s, 30)
        self.assertFalse(config.do_stout)
        self.assertEqual(config.str_format, "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        
    def test_config_defaults(self):
        """Test that default config values are set correctly."""
        config = LoggerConfig()
        
        self.assertEqual(config.base_level, LogLevel.INFO)
        self.assertEqual(config.buffer_capacity, 100)
        self.assertEqual(config.buffer_timeout, 5.0)
        self.assertTrue(config.do_stout)
        self.assertEqual(config.str_format, "%(asctime)s [%(levelname)s] %(name)s - %(message)s")
        
    def test_invalid_buffer_capacity(self):
        """Test that invalid buffer capacity raises ValueError."""
        with self.assertRaises(ValueError):
            LoggerConfig(buffer_capacity=-10)
            
    def test_invalid_buffer_timeout(self):
        """Test that invalid buffer timeout raises ValueError."""
        with self.assertRaises(ValueError):
            LoggerConfig(buffer_timeout_s=-5.0)
            
    def test_invalid_str_format(self):
        """Test that invalid string format raises ValueError."""
        with self.assertRaises(ValueError):
            LoggerConfig(str_format="")  # Empty string
            
    def test_config_validation_success(self):
        """Test that a valid config passes validation."""
        LoggerConfig(
            base_level=LogLevel.ERROR,
            buffer_capacity=500,
            buffer_timeout_s=10.0
        )


class TestLogger(unittest.TestCase):
    def setUp(self):
        """
        Set up a logger instance for testing.
        """
        self.logger_config = LoggerConfig(
            base_level=LogLevel.INFO,
            do_stout=True,
            str_format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            buffer_capacity=10,
            buffer_timeout=5.0
        )
        self.file_config = FileLogHandler(
            filepath="test_logs.txt",
            create=True
        )
        self.logger = Logger(
            config=self.logger_config, 
            handlers=[self.file_config],
            name="TestLogger"
        )
        self.loop = asyncio.get_event_loop()

    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.file_config.filepath):
            os.remove(self.file_config.filepath)
        
        # Ensure logger is shut down
        if hasattr(self, 'logger') and self.logger._is_running:
            self.loop.run_until_complete(self.logger.shutdown())

    def read_log_file(self):
        """Helper to read the log file contents."""
        with open(self.file_config.filepath, "r") as file:
            return file.read()

    def test_initialization(self):
        """Test logger initialization with explicit config."""
        self.logger = Logger(
            config=self.logger_config, 
            handlers=[self.file_config],
            name="TestLogger"
        )
        # Validate all the attrs are set correctly
        self.assertEqual(self.logger._config, self.logger_config)
        self.assertEqual(self.logger._handlers, [self.file_config])
        self.assertEqual(self.logger._buffer_size, 0)
        self.assertEqual(self.logger._name, "TestLogger")
        self.assertIsInstance(self.logger._buffer_start_time, float)
        self.assertIsInstance(self.logger._msg_queue, asyncio.Queue)
        self.assertIsInstance(self.logger._ev_loop, asyncio.AbstractEventLoop)
        self.assertTrue(self.logger._is_running)

    def test_default_initialization(self):
        """Test logger initialization with default config."""
        default_logger = Logger()
        self.assertIsNotNone(default_logger._config)
        self.assertEqual(default_logger._name, "")
        self.assertEqual(default_logger._handlers, [])
        self.assertTrue(default_logger._is_running)
        self.loop.run_until_complete(default_logger.shutdown())

    def test_failed_initialization(self):
        """Test initialization with invalid handler."""
        class InvalidHandler:
            pass

        # Handlers that don't inherit from BaseLogHandler will 
        # raise a TypeError on initialization. 
        with self.assertRaises(TypeError):
            Logger(
                config=self.logger_config, 
                handlers=[InvalidHandler()]
            )

    def test_set_log_level(self):
        """Test setting log level at runtime."""
        # Start with INFO level
        self.assertEqual(self.logger._config.base_level, LogLevel.INFO)
        
        # Set to DEBUG level
        self.logger.set_log_level(LogLevel.DEBUG)
        self.assertEqual(self.logger._config.base_level, LogLevel.DEBUG)
        
        # Debug message should appear
        self.logger.debug("Debug after level change")
        
        # Set to WARNING level
        self.logger.set_log_level(LogLevel.WARNING)
        self.assertEqual(self.logger._config.base_level, LogLevel.WARNING)
        
        # Debug message should not be processed
        self.logger.debug("Debug after setting to WARNING")
        
        self.loop.run_until_complete(self.logger.shutdown())
        
        log_content = self.read_log_file()
        self.assertIn("Debug after level change", log_content)
        self.assertNotIn("Debug after setting to WARNING", log_content)

    def test_set_format(self):
        """Test changing format string at runtime."""
        original_format = self.logger._config.str_format
        new_format = "%(asctime)s [TEST-%(levelname)s] %(message)s"
        
        # Log a message with original format
        self.logger.info("Message with original format")
        
        # Change format and log another message
        self.logger.set_format(new_format)
        self.assertEqual(self.logger._config.str_format, new_format)
        self.logger.info("Message with new format")
        
        self.loop.run_until_complete(self.logger.shutdown())
        
        log_content = self.read_log_file()
        # First message should have the original format
        self.assertIn("Message with original format", log_content)
        # Second message should have TEST- prefix in the level part
        self.assertIn("TEST-INFO", log_content)

    def test_trace_level(self):
        """Test trace level logging."""
        # Set log level to TRACE, currently at INFO
        self.logger.set_log_level(LogLevel.TRACE)
        self.logger.trace("This is a trace message")
        self.loop.run_until_complete(self.logger.shutdown())

        log_content = self.read_log_file()
        self.assertIn("This is a trace message", log_content)
        self.assertIn("TRACE", log_content)

    def test_debug(self):
        """Test debug level logging."""
        # Set log level to DEBUG, currently at INFO
        self.logger.set_log_level(LogLevel.DEBUG)
        self.logger.debug("This is a debug message")
        self.loop.run_until_complete(self.logger.shutdown())

        log_content = self.read_log_file()
        self.assertIn("This is a debug message", log_content)
        self.assertIn("DEBUG", log_content)

    def test_info(self):
        """Test info level logging."""
        self.logger.info("This is an info message")
        self.loop.run_until_complete(self.logger.shutdown())

        log_content = self.read_log_file()
        self.assertIn("This is an info message", log_content)
        self.assertIn("INFO", log_content)

    def test_warning(self):
        """Test warning level logging."""
        self.logger.warning("This is a warning message")
        self.loop.run_until_complete(self.logger.shutdown())

        log_content = self.read_log_file()
        self.assertIn("This is a warning message", log_content)
        self.assertIn("WARNING", log_content)

    def test_error(self):
        """Test error level logging."""
        self.logger.error("This is an error message")
        self.loop.run_until_complete(self.logger.shutdown())

        log_content = self.read_log_file()
        self.assertIn("This is an error message", log_content)
        self.assertIn("ERROR", log_content)

    def test_critical(self):
        """Test critical level logging."""
        self.logger.critical("This is a critical message")
        self.loop.run_until_complete(self.logger.shutdown())

        log_content = self.read_log_file()
        self.assertIn("This is a critical message", log_content)
        self.assertIn("CRITICAL", log_content)

    def test_with_buffering(self):
        """Test buffer behavior when buffer is filled."""
        # Fill the buffer but don't exceed capacity
        for i in range(1, 10):
            self.logger.info(f"Buffered message {i}")

        # Messages should not yet be flushed since the buffer isn't full
        log_content = self.read_log_file()
        self.assertEqual(log_content, "")

        # Add more messages to trigger buffer flush
        self.logger.info("Buffered message 10")
        self.logger.info("Buffered message 11")
        
        # Allow time for async operations
        self.loop.run_until_complete(asyncio.sleep(0.1))

        log_content = self.read_log_file()
        self.assertIn("Buffered message 1", log_content)
        self.assertIn("Buffered message 10", log_content)
        
        self.loop.run_until_complete(self.logger.shutdown())

    def test_error_and_critical_bypass_buffer(self):
        """Test that ERROR and CRITICAL messages bypass the buffer."""
        # Add some INFO messages to the buffer
        for i in range(1, 5):
            self.logger.info(f"Buffered message {i}")

        # Add an ERROR message which should flush the buffer
        self.logger.error("This is an error message")

        # Allow time for async operations
        self.loop.run_until_complete(asyncio.sleep(0.1))

        log_content = self.read_log_file()
        self.assertIn("This is an error message", log_content)
        self.assertIn("Buffered message 1", log_content)

        # Add a CRITICAL message which should also flush the buffer
        self.logger.critical("This is a critical message")

        # Allow time for async operations
        self.loop.run_until_complete(asyncio.sleep(0.1))

        log_content = self.read_log_file()
        self.assertIn("This is a critical message", log_content)

        self.loop.run_until_complete(self.logger.shutdown())

    def test_with_time_flush(self):
        """Test buffer flush based on timeout."""
        self.logger.info("Message 1")
        self.logger.warning("Message 2")

        # Wait enough time to ensure the buffer flushes due to timeout
        self.loop.run_until_complete(asyncio.sleep(1.5))

        # Add another message after timeout
        self.logger.warning("Message 3")
        
        # Allow time for async operations
        self.loop.run_until_complete(asyncio.sleep(1.0))

        log_content = self.read_log_file()
        self.assertIn("Message 1", log_content)
        self.assertIn("Message 2", log_content)

        self.loop.run_until_complete(self.logger.shutdown())

    def test_shutdown(self):
        """Test logger shutdown behavior."""
        self.logger.info("Message before shutdown")
        self.loop.run_until_complete(self.logger.shutdown())

        log_content = self.read_log_file()
        self.assertIn("Message before shutdown", log_content)
        
        # Verify logger is no longer running
        self.assertFalse(self.logger._is_running)

    def test_no_below_base_level(self):
        """Test that messages below base level are not logged."""
        self.logger.set_log_level(LogLevel.ERROR)

        self.logger.debug("This should not be logged")
        self.logger.info("This should not be logged")
        self.logger.warning("This should not be logged")
        
        # Add a message that should be logged
        self.logger.error("This should be logged")

        self.loop.run_until_complete(self.logger.shutdown())
        log_content = self.read_log_file()
        self.assertNotIn("This should not be logged", log_content)
        self.assertIn("This should be logged", log_content)

    def test_stdout(self):
        """Test stdout output when do_stout is True."""
        with StringIO() as buf, redirect_stdout(buf):
            self.logger.info("This should appear in stdout")
            self.logger.warning("Another stdout message")
            
            # Allow time for async operations
            self.loop.run_until_complete(asyncio.sleep(0.1))
            
            output = buf.getvalue()
            
        self.assertIn("This should appear in stdout", output)
        self.assertIn("Another stdout message", output)
        
        self.loop.run_until_complete(self.logger.shutdown())

    def test_no_stdout(self):
        """Test no stdout output when do_stout is False."""
        # Create a new logger with stdout disabled
        no_stdout_config = LoggerConfig(do_stout=False)
        no_stdout_logger = Logger(config=no_stdout_config, handlers=[self.file_config])
        
        with StringIO() as buf, redirect_stdout(buf):
            no_stdout_logger.info("This should not appear in stdout")
            
            # Allow time for async operations
            self.loop.run_until_complete(asyncio.sleep(0.1))
            
            output = buf.getvalue()
            
        self.assertEqual("", output)
        
        self.loop.run_until_complete(no_stdout_logger.shutdown())

    def test_set_format(self):
        """Test changing format string at runtime."""
        original_format = self.logger._config.str_format
        new_format = "%(asctime)s | %(levelname)s | %(message)s"
        
        self.logger.set_format(new_format)
        self.assertEqual(self.logger._config.str_format, new_format)
        
        self.logger.info("Message with new format")
        self.loop.run_until_complete(self.logger.shutdown())
        
        log_content = self.read_log_file()
        # Check that the message is logged with the new format
        self.assertIn(" | INFO | Message with new format", log_content)

    def test_multiple_handlers(self):
        """Test logger with multiple handlers."""
        second_file_handler = FileLogHandler(
            filepath="test_logs_2.txt",
            create=True
        )
        
        multi_handler_logger = Logger(
            config=self.logger_config,
            handlers=[self.file_config, second_file_handler]
        )
        
        multi_handler_logger.info("Message for multiple handlers")
        self.loop.run_until_complete(multi_handler_logger.shutdown())
        
        # Check first log file
        log_content = self.read_log_file()
        self.assertIn("Message for multiple handlers", log_content)
        
        # Check second log file
        with open(second_file_handler.filepath, "r") as file:
            second_log_content = file.read()
        self.assertIn("Message for multiple handlers", second_log_content)
        
        # Clean up
        if os.path.exists(second_file_handler.filepath):
            os.remove(second_file_handler.filepath)

    def test_logger_with_name(self):
        """Test logger with a specific name."""
        named_logger = Logger(
            config=self.logger_config,
            name="SpecialLogger",
            handlers=[self.file_config]
        )
        
        named_logger.info("Message from named logger")
        self.loop.run_until_complete(named_logger.shutdown())
        
        log_content = self.read_log_file()
        self.assertIn("SpecialLogger", log_content)
        self.assertIn("Message from named logger", log_content)

    def test_stress_test_with_message_burst(self):
        """Test logger with a large burst of messages to stress test it."""
        # Create a logger with a larger buffer capacity for stress testing
        stress_config = LoggerConfig(
            base_level=LogLevel.DEBUG,
            do_stout=False,  # Disable stdout to speed up test
            buffer_capacity=100,  # Larger buffer
            buffer_timeout=1.0,
        )
        
        stress_file_handler = FileLogHandler(
            filepath="stress_test_logs.txt",
            create=True
        )
        
        stress_logger = Logger(
            config=stress_config,
            name="StressTestLogger",
            handlers=[stress_file_handler],
        )
        
        # Send a large burst of messages
        num_messages = 10_000
        for i in range(num_messages):
            stress_logger.info(f"Stress test message {i}")
        
        # Ensure all messages are flushed
        self.loop.run_until_complete(stress_logger.shutdown())
        
        # Read the log file and count messages
        with open(stress_file_handler.filepath, "r") as file:
            log_content = file.read()
        
        # Count the number of logged messages
        message_count = 0
        for i in range(num_messages):
            if f"Stress test message {i}" in log_content:
                message_count += 1
        
        # Check that all messages were logged
        self.assertEqual(message_count, num_messages, 
                         f"Expected {num_messages} messages, but found {message_count}")
        
        # Clean up
        if os.path.exists(stress_file_handler.filepath):
            os.remove(stress_file_handler.filepath)

if __name__ == "__main__":
    unittest.main()
