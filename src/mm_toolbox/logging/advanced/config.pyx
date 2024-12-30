cdef class LoggerConfig:
    def __init__(self, str transport="ipc", str path="ipc:///logger_queue.ipc"):
        self.transport = transport.lower()

        match self.transport:
            case "ipc":
                # Must start with "ipc://"
                if not path.startswith("ipc://"):
                    raise ValueError(f"Invalid IPC path '{path}'. Must start with 'ipc://'.")
                self.path = path

            case "tcp":
                # Must start with "tcp://"
                if not path.startswith("tcp://"):
                    raise ValueError(f"Invalid TCP path '{path}'. Must start with 'tcp://'.")
                
                # Example: tcp://127.0.0.1:5556[/optional/stuff]
                tcp_part = path[len("tcp://"):]
                
                # We expect something like "host:port"
                slash_split = tcp_part.split("/", 1)
                host_port = slash_split[0]
                
                if ":" not in host_port:
                    raise ValueError(f"TCP path '{path}' has no port specified.")
                
                host, port_str = host_port.split(":", 1)
                try:
                    port = int(port_str)
                except ValueError:
                    raise ValueError(f"Invalid port in TCP path '{path}'. Must be an integer.")
                
                if not (1 <= port <= 65535):
                    raise ValueError(f"Port must be between 1 and 65535, got {port}.")
                
                self.path = path

            case _:
                raise ValueError(f"Invalid transport; expected ['ipc', 'tcp'] but got '{self.transport}'")
