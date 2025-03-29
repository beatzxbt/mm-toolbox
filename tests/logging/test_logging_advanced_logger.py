import os
import asyncio
import unittest
import msgspec
import multiprocessing
import time

from mm_toolbox.time.time import time_ns
from mm_toolbox.logging.advanced import (
    LogLevel,
    LoggerConfig, 
    MasterLogger,
    WorkerLogger,
    FileLogHandler,
)


class TestLoggerConfig(unittest.TestCase):
    """Test the LoggerConfig validation."""
    
    def test_valid_config(self):
        """Test that a valid config initializes correctly."""
        config = LoggerConfig(
            base_level=LogLevel.INFO,
            do_stout=True,
            str_format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            path="tcp://127.0.0.1:5555",
            log_timeout_s=2.0,
            data_timeout_s=5.0
        )
        self.assertEqual(config.path, "tcp://127.0.0.1:5555")
        self.assertTrue(config.do_stout)
        self.assertEqual(config.str_format, "%(asctime)s [%(levelname)s] %(name)s - %(message)s")
        self.assertEqual(config.base_level, LogLevel.INFO)
        self.assertEqual(config.log_timeout_s, 2.0)
        self.assertEqual(config.data_timeout_s, 5.0)
    
    def test_default_config(self):
        """Test that default config values are set correctly."""
        config = LoggerConfig()
        self.assertEqual(config.path, "ipc:///tmp/hft_logger")
        self.assertFalse(config.do_stout)
        self.assertEqual(config.str_format, "%(asctime)s [%(levelname)s] %(name)s - %(message)s")
        self.assertEqual(config.base_level, LogLevel.INFO)
        self.assertEqual(config.log_timeout_s, 2.0)
        self.assertEqual(config.data_timeout_s, 5.0)
    
    def test_invalid_str_format(self):
        """Test that invalid string format raises ValueError."""
        with self.assertRaises(ValueError):
            LoggerConfig(str_format="No message placeholder")  # Missing %(message)s
    def test_format_string(self):
        """Test that the format string correctly formats log messages."""
        config = LoggerConfig(str_format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
        
        # Create a dictionary with the format values
        format_values = {
            "asctime": "2023-01-01 12:00:00",
            "levelname": "INFO",
            "name": "test_logger",
            "message": "Test message",
        }
        
        # Format the string directly
        formatted = config.str_format % format_values
        
        # Check that the format is correct
        self.assertEqual(formatted, "2023-01-01 12:00:00 [INFO] test_logger - Test message")

    def test_invalid_base_level(self):
        """Test that invalid base level raises ValueError."""
        with self.assertRaises(ValueError):
            LoggerConfig(base_level=999)  # Invalid log level
    
    def test_valid_base_levels(self):
        """Test that valid base levels are accepted."""
        config_trace = LoggerConfig(base_level=LogLevel.TRACE)
        self.assertEqual(config_trace.base_level, LogLevel.TRACE)
        
        config_debug = LoggerConfig(base_level=LogLevel.DEBUG)
        self.assertEqual(config_debug.base_level, LogLevel.DEBUG)
        
        config_info = LoggerConfig(base_level=LogLevel.INFO)
        self.assertEqual(config_info.base_level, LogLevel.INFO)
        
        config_warning = LoggerConfig(base_level=LogLevel.WARNING)
        self.assertEqual(config_warning.base_level, LogLevel.WARNING)
        
        config_error = LoggerConfig(base_level=LogLevel.ERROR)
        self.assertEqual(config_error.base_level, LogLevel.ERROR)
        
        config_critical = LoggerConfig(base_level=LogLevel.CRITICAL)
        self.assertEqual(config_critical.base_level, LogLevel.CRITICAL)
    
    def test_default_base_level(self):
        """Test that the default base level is INFO."""
        config = LoggerConfig()
        self.assertEqual(config.base_level, LogLevel.INFO)
        
    def test_invalid_timeouts(self):
        """Test that invalid timeout values raise ValueError."""
        with self.assertRaises(ValueError):
            LoggerConfig(log_timeout_s=0.0)
            
        with self.assertRaises(ValueError):
            LoggerConfig(log_timeout_s=-1.0)
            
        with self.assertRaises(ValueError):
            LoggerConfig(data_timeout_s=0.0)
            
        with self.assertRaises(ValueError):
            LoggerConfig(data_timeout_s=-1.0)
            
    def test_valid_timeouts(self):
        """Test that valid timeout values are accepted."""
        config = LoggerConfig(log_timeout_s=1.5, data_timeout_s=3.0)
        self.assertEqual(config.log_timeout_s, 1.5)
        self.assertEqual(config.data_timeout_s, 3.0)


class TestAdvancedLogger(unittest.TestCase):
    def setUp(self):
        """Set up the test environment with master-worker loggers."""
        self.ev_loop = asyncio.get_event_loop()
        self.logger_config = LoggerConfig()
        self.file_path = "test_worker_logs.txt"
        self.file_handler = FileLogHandler(
            filepath=self.file_path,
            create=True
        )
        self.master_logger = MasterLogger(
            config=self.logger_config,
            name="test_master",
            log_handlers=[self.file_handler],
            data_handlers=[self.file_handler]
        )
        self.worker_logger = WorkerLogger(
            config=self.logger_config,
            name="test_worker",
        )   

    def tearDown(self):
        """Clean up after tests by removing log files."""
        self.worker_logger.shutdown()
        self.master_logger.shutdown()
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

    def read_log_file(self):
        """Helper method to read the log file contents."""
        with open(self.file_path, "r") as file:
            return file.read()

    def test_worker_initialization(self):
        """Test that the worker logger initializes correctly."""
        self.assertIsNotNone(self.worker_logger)

        # NOTE: Some of the methods would be hidden from the  
        # user as they're internal cython variables. For the sake 
        # of testing, we'll just check that the worker logger is 
        # initialized without raising any exceptions.
        self.assertEqual(self.worker_logger.get_name(), "test_worker")
        self.assertTrue(self.worker_logger.is_running())
        self.assertEqual(self.worker_logger.get_config(), self.logger_config)

    def test_master_initialization(self):
        """Test that the master logger initializes correctly."""
        self.assertIsNotNone(self.master_logger)

        # NOTE: Some of the methods would be hidden from the  
        # user as they're internal cython variables. For the sake 
        # of testing, we'll just check that the worker logger is 
        # initialized without raising any exceptions.
        self.assertEqual(self.master_logger.get_name(), "test_master")
        self.assertTrue(self.master_logger.is_running())
        self.assertEqual(self.master_logger.get_config(), self.logger_config)
    
    def test_change_log_level(self):
        """Test changing the log level at runtime for both master and worker loggers."""
        # Initial LogLevel is INFO (default in setup)
        # Change LogLevel to TRACE and check the config
        self.worker_logger.set_log_level(LogLevel.TRACE)
        updated_worker_config = self.worker_logger.get_config()
        self.assertEqual(updated_worker_config.base_level, LogLevel.TRACE)

        # Repeat change for master logger
        self.master_logger.set_log_level(LogLevel.TRACE)
        updated_master_config = self.master_logger.get_config()
        self.assertEqual(updated_master_config.base_level, LogLevel.TRACE)
        
    def test_change_format_string(self):
        """Test changing the format string at runtime for both master and worker loggers."""
        new_format = "%(asctime)s - %(levelname)s - %(message)s"

        # Initial format is set to '%(asctime)s [%(levelname)s] %(name)s - %(message)s'
        # Change format to new_format and check the config
        self.worker_logger.set_format(new_format)
        updated_worker_config = self.worker_logger.get_config()
        self.assertEqual(updated_worker_config.str_format, new_format)

        # Repeat change for master logger
        self.master_logger.set_format(new_format)
        updated_master_config = self.master_logger.get_config()
        self.assertEqual(updated_master_config.str_format, new_format)

    def test_log_methods(self):
        """
        Test all log methods using the standard logger configuration.
        
        This test verifies that all logging levels work correctly and 
        that the log file is created and populated correctly.
        """
        # Set LogLevel to TRACE to ensure all messages are logged
        self.worker_logger.set_log_level(LogLevel.TRACE)
        self.master_logger.set_log_level(LogLevel.TRACE)

        # Test all log methods
        self.worker_logger.trace("Trace message")
        self.worker_logger.debug("Debug message")
        self.worker_logger.info("Info message")
        self.worker_logger.warning("Warning message")
        self.worker_logger.error("Error message")
        self.worker_logger.critical("Critical message")
        
        # Give some time for messages to be processed
        time.sleep(0.1)
        
        # Flush all messages by shutting down and restarting
        self.worker_logger.shutdown()
        
        # Give some time for messages to be sent to the master logger
        time.sleep(0.1)

        self.master_logger.shutdown()
        
        # Read the log file
        log_content = self.read_log_file()
        
        # Check that all messages are in the log file
        self.assertIn("Info message", log_content)
        self.assertIn("Warning message", log_content)
        self.assertIn("Error message", log_content)
        self.assertIn("Critical message", log_content)
        
        # Check log level strings are present
        self.assertIn("INFO", log_content)
        self.assertIn("WARNING", log_content)
        self.assertIn("ERROR", log_content)
        self.assertIn("CRITICAL", log_content)
        
    def test_data_log_method(self):
        """Test the data method with valid and invalid inputs."""
        class TestData(msgspec.Struct):
            mid_px: float
            sz: float
            px: float
            is_buy: bool
        
        # Test with valid data
        valid_data = TestData(mid_px=100.0, sz=1.0, px=100.0, is_buy=True)
        try:
            self.worker_logger.data(valid_data)
        except Exception as e:
            self.fail(f"data() raised exception with valid input: {e}")
        
        # Test with valid, but unsafe data
        try:
            self.worker_logger.data(valid_data, unsafe=True)
        except Exception as e:
            self.fail(f"data() raised exception with valid input (unsafe=True): {e}")
        
        # Test with invalid data
        invalid_data = {"value": 42, "name": "test_tcp"}
        with self.assertRaises(TypeError):
            self.worker_logger.data(invalid_data)
        
        # Give some time for messages to be processed
        time.sleep(0.1)
        
        # Flush all messages by shutting down and restarting
        self.worker_logger.shutdown()
        
        # Give some time for messages to be sent to the master logger
        time.sleep(0.1)

        self.master_logger.shutdown()

        # Read the log file
        msg_content = self.read_log_file()

        # Check that all sent data is in the file
        self.assertIn("{'mid_px': 100.0, 'sz': 1.0, 'px': 100.0, 'is_buy': True}", msg_content)
        self.assertIn("test_worker", msg_content)
    
    def test_multiprocess_logging(self):
        """Test logging from multiple processes to ensure all logs are received."""
        # Configure IPC transport for multiprocess communication
        self.logger_config.path = f"ipc:///tmp/test_logger_multiprocess_{os.getpid()}"
        
        # Create master logger
        master_logger = MasterLogger(
            config=self.logger_config,
            name="test_master_multiprocess",
            log_handlers=[self.file_handler],
            data_handlers=[self.file_handler]
        )
        
        # Function to run in each worker process
        def worker_process(worker_id):
            worker_logger = WorkerLogger(
                config=self.logger_config,
                name=f"worker_{worker_id}"
            )
            
            # Send logs at different levels
            worker_logger.trace(f"Trace message from worker {worker_id}")
            worker_logger.debug(f"Debug message from worker {worker_id}")
            worker_logger.info(f"Info message from worker {worker_id}")
            worker_logger.warning(f"Warning message from worker {worker_id}")
            worker_logger.error(f"Error message from worker {worker_id}")
            worker_logger.critical(f"Critical message from worker {worker_id}")
            
            # Create and send data
            class TestData(msgspec.Struct):
                worker_id: int
                message: str
            
            data = TestData(worker_id=worker_id, message=f"Data from worker {worker_id}")
            worker_logger.data(data)
            
            # Shutdown worker logger
            worker_logger.shutdown()
        
        # Spawn 25 worker processes
        processes = []
        for i in range(25):
            p = multiprocessing.Process(target=worker_process, args=(i,))
            processes.append(p)
            p.start()
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
        
        # Give some time for all messages to be processed
        time.sleep(1)
        
        # Shutdown master logger
        master_logger.shutdown()
        
        # Read the log file
        msg_content = self.read_log_file()
        
        # Verify that logs from all workers are present
        for i in range(25):
            self.assertIn(f"worker_{i}", msg_content, f"Missing logs from worker {i}")
            self.assertIn(f"Info message from worker {i}", msg_content)
            self.assertIn(f"Error message from worker {i}", msg_content)
            self.assertIn(f"Data from worker {i}", msg_content)
        
        # Verify that all log levels are represented
        self.assertIn("Trace message", msg_content)
        self.assertIn("Debug message", msg_content)
        self.assertIn("Info message", msg_content)
        self.assertIn("Warning message", msg_content)
        self.assertIn("Error message", msg_content)
        self.assertIn("Critical message", msg_content)
    
    def test_multiprocess_data(self):
        """Test data sending from multiple processes to ensure all data is received."""
        # Configure IPC transport for multiprocess communication
        self.logger_config.path = f"ipc:///tmp/test_logger_multiprocess_data_{os.getpid()}"
        
        # Create master logger
        master_logger = MasterLogger(
            config=self.logger_config,
            name="test_master_multiprocess_data",
            log_handlers=[self.file_handler],
            data_handlers=[self.file_handler]
        )
        
        # Function to run in each worker process
        def worker_process(worker_id):
            worker_logger = WorkerLogger(
                config=self.logger_config,
                name=f"data_worker_{worker_id}"
            )
            
            # Create and send multiple data objects
            class TestData(msgspec.Struct):
                worker_id: int
                value: int
                message: str
            
            # Send multiple data objects with different values
            for i in range(5):
                data = TestData(
                    worker_id=worker_id, 
                    value=i*10, 
                    message=f"Data packet {i} from worker {worker_id}"
                )
                worker_logger.data(data)
            
            # Shutdown worker logger
            worker_logger.shutdown()
        
        # Spawn 25 worker processes
        processes = []
        for i in range(25):
            p = multiprocessing.Process(target=worker_process, args=(i,))
            processes.append(p)
            p.start()
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
        
        # Give some time for all messages to be processed
        time.sleep(1)
        
        # Shutdown master logger
        master_logger.shutdown()
        
        # Read the log file
        msg_content = self.read_log_file()
        
        # Verify that data from all workers is present
        for i in range(25):
            self.assertIn(f"data_worker_{i}", msg_content, f"Missing data from worker {i}")
            # Check for at least one data packet from each worker
            self.assertIn(f"Data packet", msg_content)
            self.assertIn(f"worker {i}", msg_content)
        
        # Verify that different data values are represented
        for i in range(5):
            value = i*10
            self.assertIn(f"value={value}", msg_content, f"Missing data with value {value}")
    
    def test_multiprocess_high_burst_rate(self):
        """Test that the logger can handle a high burst rate of messages without dropping any."""
        BURST_COUNT = 10_000

        # Configure and start master logger
        config = LoggerConfig(
            base_level=LogLevel.TRACE,
            path=f"ipc:///tmp/high_burst_test",
        )
        
        master_logger = MasterLogger(
            config=config,
            name="test_master_high_burst",
            log_handlers=[self.file_handler],
            data_handlers=[self.file_handler]
        )
        
        # Define worker process function that sends a burst of messages
        def worker_process(worker_id):
            # Create worker logger
            worker_logger = WorkerLogger(
                config=config,
                name=f"burst_worker_{worker_id}"
            )
            
            # Send a burst of log messages at different levels
            for i in range(BURST_COUNT):
                level = i % 6
                if level == LogLevel.TRACE:
                    worker_logger.trace(f"TRACE message {i} from worker {worker_id}; seq={i}")
                elif level == LogLevel.DEBUG:
                    worker_logger.debug(f"DEBUG message {i} from worker {worker_id}; seq={i}")
                elif level == LogLevel.INFO:
                    worker_logger.info(f"INFO message {i} from worker {worker_id}; seq={i}")
                elif level == LogLevel.WARNING:
                    worker_logger.warning(f"WARNING message {i} from worker {worker_id}; seq={i}")
                elif level == LogLevel.ERROR:
                    worker_logger.error(f"ERROR message {i} from worker {worker_id}; seq={i}")
                elif level == LogLevel.CRITICAL:
                    worker_logger.critical(f"CRITICAL message {i} from worker {worker_id}; seq={i}")
            
            # Create and send data objects in burst
            class BurstData(msgspec.Struct):
                worker_id: int
                seq: int
                time: int
            
            # Send burst of data objects
            for i in range(BURST_COUNT):
                data = BurstData(
                    worker_id=worker_id,
                    seq=i,
                    time=time_ns()
                )
                worker_logger.data(data)
            
            # Shutdown worker logger
            worker_logger.shutdown()
        
        # Spawn multiple worker processes to create high load
        processes = []
        num_workers = 25
        for i in range(num_workers):
            p = multiprocessing.Process(target=worker_process, args=(i,))
            processes.append(p)
            p.start()
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
        
        # Give some time for all messages to be processed
        time.sleep(5)
        
        # Shutdown master logger
        master_logger.shutdown()
        
        # Read the log file
        log_content = self.read_log_file()
        
        # Verify that messages from all workers are present with their sequence numbers
        for worker_id in range(num_workers):
            # Check for every log message sequence number
            for seq in range(BURST_COUNT):
                self.assertIn(f"; seq={seq}", log_content, 
                             f"Missing log sequence {seq} from worker {worker_id}")
            
            # Check for every data message sequence number
            for seq in range(BURST_COUNT):  
                self.assertIn(f"worker_id={worker_id}, seq={seq}", log_content, 
                             f"Missing data sequence {seq} from worker {worker_id}")
        
        # Verify that different log levels are represented
        for level in ["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            self.assertIn(level, log_content, f"Missing {level} log messages")
        
        # Verify that logging after shutdown doesn't raise exceptions
        try:
            master_logger.info("Message after shutdown")
        except Exception as e:
            self.fail(f"Logging after shutdown raised exception: {e}")

    def test_multiple_shutdowns(self):
        """Test that multiple shutdowns don't cause issues."""
        self.worker_logger.shutdown()
        self.worker_logger.shutdown()  # Should be a no-op
        self.assertFalse(self.worker_logger.is_running())