import unittest
import os
import asyncio
from io import StringIO
from contextlib import redirect_stdout
from mm_toolbox.logging import Logger, LoggerConfig, FileLogConfig, DiscordLogConfig, TelegramLogConfig


class TestLoggerConfig(unittest.TestCase):
    def test_valid_config(self):
        config = LoggerConfig(base_level="INFO", stout=True, max_buffer_size=20, max_buffer_age=15)
        config.validate()

    def test_invalid_log_level(self):
        config = LoggerConfig(base_level="INVALID")
        with self.assertRaises(ValueError):
            config.validate()

    def test_invalid_buffer_size(self):
        config = LoggerConfig(max_buffer_size=0)
        with self.assertRaises(ValueError):
            config.validate()

    def test_invalid_buffer_age(self):
        config = LoggerConfig(max_buffer_age=0)
        with self.assertRaises(ValueError):
            config.validate()


class TestFileLogConfig(unittest.TestCase):
    def test_valid_config(self):
        config = FileLogConfig(filepath="logs.txt", buffer_size=10, flush_interval=5)
        config.validate()  

    def test_invalid_filepath(self):
        config = FileLogConfig(filepath="invalid_log")
        with self.assertRaises(ValueError):
            config.validate()

    def test_invalid_buffer_size(self):
        config = FileLogConfig(buffer_size=0)
        with self.assertRaises(ValueError):
            config.validate()

    def test_invalid_flush_interval(self):
        config = FileLogConfig(flush_interval=0)
        with self.assertRaises(ValueError):
            config.validate()


class TestDiscordLogConfig(unittest.TestCase):
    def test_valid_config(self):
        config = DiscordLogConfig(webhook="https://discord.com/api/webhooks/12345/abcdefg")
        config.validate()  

    def test_invalid_webhook(self):
        config = DiscordLogConfig(webhook="invalid_webhook")
        with self.assertRaises(ValueError):
            config.validate()


class TestTelegramLogConfig(unittest.TestCase):
    def test_valid_config(self):
        config = TelegramLogConfig(bot_token="4839574812:AAFD39kkdpWt3ywyRZergyOLMaJhac60qc", chat_id="123456")
        config.validate()  

    def test_invalid_bot_token(self):
        config = TelegramLogConfig(bot_token="invalid_bot_token")
        with self.assertRaises(ValueError):
            config.validate()

    def test_invalid_chat_id(self):
        config = TelegramLogConfig(chat_id="invalid_chat_id")
        with self.assertRaises(ValueError):
            config.validate()


class TestLogger(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.get_event_loop()
        self.logger_config = LoggerConfig(base_level="DEBUG")
        self.file_config = FileLogConfig(filepath="test_logs.txt")
        self.logger = Logger(logger_config=self.logger_config, file_config=self.file_config)

    def tearDown(self):
        if os.path.exists(self.file_config.filepath):
            os.remove(self.file_config.filepath)

    def read_log_file(self):
        with open(self.file_config.filepath, 'r') as file:
            return file.read()

    def test_debug_logging(self):
        self.logger.debug("This is a debug message")
        self.loop.run_until_complete(self.logger.shutdown())

        log_content = self.read_log_file()
        self.assertIn("This is a debug message", log_content)

    def test_info_logging(self):
        self.logger.info("This is an info message")
        self.loop.run_until_complete(self.logger.shutdown())

        log_content = self.read_log_file()
        self.assertIn("This is an info message", log_content)

    def test_warning_logging(self):
        self.logger.warning("This is a warning message")
        self.loop.run_until_complete(self.logger.shutdown())

        log_content = self.read_log_file()
        self.assertIn("This is a warning message", log_content)

    def test_error_logging(self):
        self.logger.error("This is an error message")
        self.loop.run_until_complete(self.logger.shutdown())

        log_content = self.read_log_file()
        self.assertIn("This is an error message", log_content)

    def test_critical_logging(self):
        self.logger.critical("This is a critical message")
        self.loop.run_until_complete(self.logger.shutdown())

        log_content = self.read_log_file()
        self.assertIn("This is a critical message", log_content)

    def test_logging_with_buffering(self):
        for i in range(1, 11):
            self.logger.debug(f"Buffered message {i}")

        # Messages should not yet be flushed since the buffer is full
        log_content = self.read_log_file()
        self.assertEqual(log_content, "")

        # Add one more message to trigger buffer flush
        self.logger.debug("Buffered message 11")
        self.loop.run_until_complete(self.logger.shutdown())

        log_content = self.read_log_file()
        self.assertIn("Buffered message 1", log_content)
        self.assertIn("Buffered message 11", log_content)

    def test_error_and_critical_bypass_buffer(self):
        for i in range(1, 5):
            self.logger.info(f"Buffered message {i}")

        # Add an ERROR message which should bypass the buffer
        self.logger.error("This is an error message")

        log_content = self.read_log_file()
        self.assertIn("This is an error message", log_content)
        self.assertIn("Buffered message 1", log_content)

        # Add a CRITICAL message which should also bypass the buffer
        self.logger.critical("This is a critical message")

        log_content = self.read_log_file()
        self.assertIn("This is a critical message", log_content)
        self.assertNotIn("Buffered message 1", log_content)

        # Now shutdown the logger to flush the buffer
        self.loop.run_until_complete(self.logger.shutdown())

        log_content = self.read_log_file()
        self.assertIn("Buffered message 1", log_content)

    def test_logging_with_time_flush(self):
        logger_config = LoggerConfig(base_level="DEBUG", stout=False, max_buffer_size=10, max_buffer_age=1)
        logger = Logger(logger_config=logger_config, file_config=self.file_config)

        logger.debug("Message 1")
        logger.info("Message 2")

        # Wait enough time to ensure the buffer flushes
        self.loop.run_until_complete(asyncio.sleep(1.5))
        self.loop.run_until_complete(logger.shutdown())

        log_content = self.read_log_file()
        self.assertIn("Message 1", log_content)
        self.assertIn("Message 2", log_content)

    def test_logging_shutdown(self):
        logger = Logger(logger_config=self.logger_config, file_config=self.file_config)

        logger.debug("Message before shutdown")
        self.loop.run_until_complete(logger.shutdown())

        log_content = self.read_log_file()
        self.assertIn("Message before shutdown", log_content)

    def test_no_logging_below_base_level(self):
        logger_config = LoggerConfig(base_level="ERROR", stout=False)
        logger = Logger(logger_config=logger_config, file_config=self.file_config)

        logger.debug("This should not be logged")
        logger.info("This should not be logged")
        logger.warning("This should not be logged")

        self.loop.run_until_complete(logger.shutdown())
        log_content = self.read_log_file()
        self.assertEqual(log_content, "")

    def test_stdout_logging(self):
        logger_config = LoggerConfig(base_level="DEBUG", stout=True)
        logger = Logger(logger_config=logger_config, file_config=self.file_config)

        with StringIO() as buf, redirect_stdout(buf):
            logger.debug("This should appear in stdout")
            logger.info("Another stdout message")
            self.loop.run_until_complete(logger.shutdown())

            output = buf.getvalue()

        self.assertIn("This should appear in stdout", output)
        self.assertIn("Another stdout message", output)

if __name__ == "__main__":
    unittest.main()
