import unittest
import socket
import zmq
import threading
import time
import asyncio
from unittest.mock import patch, MagicMock

from mm_toolbox.logging.utils.zmq import ZmqConnection, AsyncZmqConnection
from mm_toolbox.logging.utils.system import _get_system_info


class TestZmqConnection(unittest.TestCase):
    """Test the synchronous ZmqConnection class."""

    def setUp(self):
        self.test_port = 5556
        self.test_path = f"tcp://127.0.0.1:{self.test_port}"

    def tearDown(self):
        # Clean up any connections that might be lingering
        pass

    def test_initialization(self):
        """Test that ZmqConnection initializes with correct parameters."""
        conn = ZmqConnection(
            socket_type=zmq.PUB,
            path=self.test_path,
            subscribe_filter="test",
            warmup_timeout=0.1,
            bind=True,
            snd_hwm=200,
            rcv_hwm=300,
        )

        self.assertEqual(conn.socket_type, zmq.PUB)
        self.assertEqual(conn.path, self.test_path)
        self.assertEqual(conn.subscribe_filter, "test")
        self.assertEqual(conn.warmup_timeout, 0.1)
        self.assertTrue(conn.bind)
        self.assertEqual(conn.snd_hwm, 200)
        self.assertEqual(conn.rcv_hwm, 300)
        self.assertFalse(conn._is_started)

    def test_start_and_stop(self):
        """Test starting and stopping the connection."""
        # Use a random port to avoid "Address already in use" errors
        s = socket.socket()
        s.bind(("", 0))
        port = s.getsockname()[1]
        s.close()

        test_path = f"tcp://127.0.0.1:{port}"

        conn = ZmqConnection(socket_type=zmq.PUB, path=test_path, bind=True)

        # Start the connection
        conn.start()
        self.assertTrue(conn._is_started)
        self.assertIsNotNone(conn._socket)

        # Stop the connection
        conn.stop()
        self.assertFalse(conn._is_started)

    def test_pub_sub_communication(self):
        """Test basic publish-subscribe communication."""
        # Use a random port to avoid "Address already in use" errors
        s = socket.socket()
        s.bind(("", 0))
        port = s.getsockname()[1]
        s.close()

        test_path = f"tcp://127.0.0.1:{port}"

        # Create publisher
        pub = ZmqConnection(socket_type=zmq.PUB, path=test_path, bind=True)
        pub.start()

        # Create subscriber
        sub = ZmqConnection(
            socket_type=zmq.SUB, path=test_path, subscribe_filter="", bind=False
        )
        sub.start()

        # Allow time for connection to establish
        time.sleep(0.5)

        # Test message
        test_message = b"Hello ZMQ"

        # Set up a flag to check if message was received
        message_received = threading.Event()
        received_data = []

        def on_message(msg: bytes):
            received_data.append(msg)
            message_received.set()

        # Start listening for messages
        sub.listen(on_message)

        # Send a message
        pub.send(test_message)

        # Wait for message to be received (with timeout)
        message_received.wait(timeout=2.0)

        # Stop connections
        sub.cancel_listening()
        sub.stop()
        pub.stop()

        # Check if message was received correctly
        self.assertTrue(message_received.is_set())
        self.assertEqual(received_data[0], test_message)

    def test_send_without_start(self):
        """Test that sending without starting raises an exception."""
        conn = ZmqConnection(socket_type=zmq.PUB, path=self.test_path)

        with self.assertRaises(RuntimeError):
            conn.send(b"test")

    def test_recv_without_start(self):
        """Test that receiving without starting raises an exception."""
        conn = ZmqConnection(socket_type=zmq.SUB, path=self.test_path)

        with self.assertRaises(RuntimeError):
            conn.recv()

    def test_listen_without_start(self):
        """Test that listening without starting raises an exception."""
        conn = ZmqConnection(socket_type=zmq.SUB, path=self.test_path)

        with self.assertRaises(RuntimeError):
            conn.listen(lambda msg: None)

    def test_bind_error_handling(self):
        """Test that binding errors are properly handled and reported."""
        # Create a ZmqConnection with an invalid protocol
        conn = ZmqConnection(socket_type=zmq.PUB, path="invalid://protocol", bind=True)

        # Attempt to start should raise a RuntimeError
        with self.assertRaises(RuntimeError) as context:
            conn.start()

        # Verify error message contains the right information
        error_msg = str(context.exception)
        self.assertIn("binding to", error_msg)
        self.assertIn("invalid://protocol", error_msg)
        self.assertIn(str(zmq.PUB), error_msg)

        # Verify socket is cleaned up
        self.assertIsNone(conn._socket)
        self.assertFalse(conn._is_started)

    def test_connect_error_handling(self):
        """Test that connection errors are properly handled and reported."""
        # Non-routable IP address that should cause connection issues
        conn = ZmqConnection(
            socket_type=zmq.SUB,
            path="tcp://192.0.2.1:12345",  # Reserved TEST-NET-1 address, shouldn't be routable
            bind=False,
        )

        # Start should not raise immediately for TCP connections
        # but we're testing the error handling mechanism
        try:
            conn.start()
            # For some transport types, errors are detected immediately during connect()
            # For others, they might appear later during send/recv operations
        except RuntimeError as e:
            # If an exception was raised, check that it contains the right info
            error_msg = str(e)
            self.assertIn("connecting to", error_msg)
            self.assertIn("tcp://192.0.2.1:12345", error_msg)
            self.assertIn(str(zmq.SUB), error_msg)

        # Clean up resources
        conn.stop()


class TestAsyncZmqConnection(unittest.TestCase):
    """Test the asynchronous ZmqConnection class."""

    def setUp(self):
        self.test_port = 5557
        self.test_path = f"tcp://127.0.0.1:{self.test_port}"

    def test_initialization(self):
        """Test that AsyncZmqConnection initializes with correct parameters."""
        conn = AsyncZmqConnection(
            socket_type=zmq.PUB,
            path=self.test_path,
            subscribe_filter="test",
            warmup_timeout=0.1,
            bind=True,
            snd_hwm=200,
            rcv_hwm=300,
        )

        self.assertEqual(conn.socket_type, zmq.PUB)
        self.assertEqual(conn.path, self.test_path)
        self.assertEqual(conn.subscribe_filter, "test")
        self.assertEqual(conn.warmup_timeout, 0.1)
        self.assertTrue(conn.bind)
        self.assertEqual(conn.snd_hwm, 200)
        self.assertEqual(conn.rcv_hwm, 300)
        self.assertFalse(conn._is_started)

    def test_async_start_and_stop(self):
        """Test starting and stopping the async connection."""
        # Use a random port to avoid "Address already in use" errors
        s = socket.socket()
        s.bind(("", 0))
        port = s.getsockname()[1]
        s.close()

        test_path = f"tcp://127.0.0.1:{port}"

        conn = AsyncZmqConnection(socket_type=zmq.PUB, path=test_path, bind=True)

        async def test_coro():
            # Start the connection
            await conn.start()
            self.assertTrue(conn._is_started)
            self.assertIsNotNone(conn._socket)

            # Stop the connection
            conn.stop()
            self.assertFalse(conn._is_started)

        # Run the test coroutine
        asyncio.run(test_coro())

    def test_async_pub_sub_communication(self):
        """Test basic async publish-subscribe communication."""
        # Use a random port to avoid "Address already in use" errors
        s = socket.socket()
        s.bind(("", 0))
        port = s.getsockname()[1]
        s.close()

        test_path = f"tcp://127.0.0.1:{port}"

        async def test_coro():
            # Create publisher
            pub = AsyncZmqConnection(socket_type=zmq.PUB, path=test_path, bind=True)
            await pub.start()

            # Create subscriber
            sub = AsyncZmqConnection(
                socket_type=zmq.SUB, path=test_path, subscribe_filter="", bind=False
            )
            await sub.start()

            # Allow time for connection to establish
            await asyncio.sleep(0.5)

            # Test message
            test_message = b"Hello Async ZMQ"

            # Set up a future to check if message was received
            received_future = asyncio.Future()

            async def on_message(msg: bytes):
                received_future.set_result(msg)

            # Start listening for messages
            sub.listen(on_message)

            # Send a message
            await pub.send(test_message)

            # Wait for message to be received (with timeout)
            try:
                received_data = await asyncio.wait_for(received_future, timeout=2.0)
            except asyncio.TimeoutError:
                self.fail("Timeout waiting for message")

            # Stop connections
            sub.stop()
            pub.stop()

            # Check if message was received correctly
            self.assertEqual(received_data, test_message)

        # Run the test coroutine
        asyncio.run(test_coro())

    def test_async_send_without_start(self):
        """Test that sending without starting raises an exception."""
        conn = AsyncZmqConnection(socket_type=zmq.PUB, path=self.test_path)

        async def test_coro():
            with self.assertRaises(RuntimeError):
                await conn.send(b"test")

        # Run the test coroutine
        asyncio.run(test_coro())

    def test_async_recv_without_start(self):
        """Test that receiving without starting raises an exception."""
        conn = AsyncZmqConnection(socket_type=zmq.SUB, path=self.test_path)

        async def test_coro():
            with self.assertRaises(RuntimeError):
                await conn.recv()

        # Run the test coroutine
        asyncio.run(test_coro())

    def test_async_listen_without_start(self):
        """Test that listening without starting raises an exception."""
        conn = AsyncZmqConnection(socket_type=zmq.SUB, path=self.test_path)

        async def test_coro():
            with self.assertRaises(RuntimeError):
                await conn.listen(lambda msg: None)

        # Run the test coroutine
        asyncio.run(test_coro())

    def test_async_bind_error_handling(self):
        """Test that binding errors in AsyncZmqConnection are properly handled."""
        # Create an AsyncZmqConnection with an invalid protocol
        conn = AsyncZmqConnection(
            socket_type=zmq.PUB, path="invalid://protocol", bind=True
        )

        # Define a test coroutine
        async def test_coro():
            with self.assertRaises(RuntimeError) as context:
                await conn.start()

            # Verify error message contains the right information
            error_msg = str(context.exception)
            self.assertIn("binding to", error_msg)
            self.assertIn("invalid://protocol", error_msg)
            self.assertIn(str(zmq.PUB), error_msg)

            # Verify socket is cleaned up
            self.assertIsNone(conn._socket)
            self.assertFalse(conn._is_started)

        # Run the test coroutine
        asyncio.run(test_coro())

    def test_async_connect_error_handling(self):
        """Test that connection errors in AsyncZmqConnection are properly handled."""
        # Non-routable IP address that should cause connection issues
        conn = AsyncZmqConnection(
            socket_type=zmq.SUB,
            path="tcp://192.0.2.1:12345",  # Reserved TEST-NET-1 address, shouldn't be routable
            bind=False,
        )

        # Define a test coroutine
        async def test_coro():
            try:
                await conn.start()
                # Some ZMQ transports don't raise immediately on connect
            except RuntimeError as e:
                # If an exception was raised, check that it contains the right info
                error_msg = str(e)
                self.assertIn("connecting to", error_msg)
                self.assertIn("tcp://192.0.2.1:12345", error_msg)
                self.assertIn(str(zmq.SUB), error_msg)
            finally:
                # Clean up resources
                conn.stop()

        # Run the test coroutine
        asyncio.run(test_coro())


class TestSystemUtils(unittest.TestCase):
    """Test the system utility functions."""

    def test_get_system_info_default(self):
        """Test getting system info with default parameters."""
        info = _get_system_info()

        # Default should include network info
        self.assertIn("hostname", info)
        self.assertIn("ip-address", info)
        self.assertIn("mac-address", info)

        # Default should not include machine info
        self.assertNotIn("architecture", info)
        self.assertNotIn("processor", info)

        # Default should not include OS info
        self.assertNotIn("platform-version", info)

    def test_get_system_info_all(self):
        """Test getting all system info."""
        info = _get_system_info(machine=True, network=True, op_sys=True)

        # Should include network info
        self.assertIn("hostname", info)
        self.assertIn("ip-address", info)
        self.assertIn("mac-address", info)

        # Should include machine info
        self.assertIn("architecture", info)
        self.assertIn("processor", info)
        self.assertIn("pid", info)
        self.assertIn("ram", info)

        # Should include OS info
        self.assertIn("platform-version", info)

    def test_get_system_info_machine_only(self):
        """Test getting only machine info."""
        info = _get_system_info(machine=True, network=False, op_sys=False)

        # Should include machine info
        self.assertIn("architecture", info)
        self.assertIn("processor", info)
        self.assertIn("pid", info)
        self.assertIn("ram", info)

        # Should not include network info
        self.assertNotIn("hostname", info)
        self.assertNotIn("ip-address", info)

        # Should not include OS info
        self.assertNotIn("platform-version", info)

    @patch("platform.machine")
    @patch("platform.processor")
    @patch("os.getpid")
    @patch("psutil.virtual_memory")
    def test_machine_info_values(self, mock_vm, mock_pid, mock_processor, mock_machine):
        """Test that machine info values are correctly retrieved."""
        # Set up mocks
        mock_machine.return_value = "x86_64"
        mock_processor.return_value = "Intel(R) Core(TM) i7"
        mock_pid.return_value = 12345

        # Mock virtual memory
        mock_vm_obj = MagicMock()
        mock_vm_obj.total = 16 * 1024 * 1024 * 1024  # 16 GB
        mock_vm.return_value = mock_vm_obj

        info = _get_system_info(machine=True, network=False, op_sys=False)

        self.assertEqual(info["architecture"], "x86_64")
        self.assertEqual(info["processor"], "Intel(R) Core(TM) i7")
        self.assertEqual(info["pid"], "12345")
        self.assertEqual(info["ram"], "16")

    @patch("socket.gethostname")
    @patch("socket.gethostbyname")
    @patch("uuid.getnode")
    def test_network_info_values(
        self, mock_getnode, mock_gethostbyname, mock_gethostname
    ):
        """Test that network info values are correctly retrieved."""
        # Set up mocks
        mock_gethostname.return_value = "test-host"
        mock_gethostbyname.return_value = "192.168.1.100"
        mock_getnode.return_value = 0x0123456789AB  # MAC address as integer

        info = _get_system_info(machine=False, network=True, op_sys=False)

        self.assertEqual(info["hostname"], "test-host")
        self.assertEqual(info["ip-address"], "192.168.1.100")
        self.assertEqual(info["mac-address"], "01:23:45:67:89:ab")
