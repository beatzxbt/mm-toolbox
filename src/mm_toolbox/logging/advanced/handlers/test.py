import msgspec
from .base import LogHandler
from mm_toolbox.time import time_s

class TestLogHandler(LogHandler):
    def __init__(self):
        super().__init__()

    def push(self, buffer):
        for line in buffer.data:
            print(msgspec.structs.asdict(line))

class TestBenchmarkLogHandler(LogHandler):
    def __init__(self):
        super().__init__()
        self.start_time = 0
        self.last_batch_time = 0
        self.events_received = 0

    def push(self, buffer):
        if self.last_batch_time == 0:
            self.last_batch_time = buffer.time
            self.start_time = time_s()

        avg_time_per_msg = (buffer.time - self.last_batch_time) / buffer.size
        pid = buffer.system["pid"]
        architecture = buffer.system["architecture"]
        ip_address = buffer.system["ip-address"]

        print(f"Batch Details:")
        print(f"  - Size: {buffer.size}")
        print(f"  - Avg. Time per Message: {avg_time_per_msg:.2f} ns")
        print(f"System Details:")
        print(f"  - PID: {pid}")
        print(f"  - Architecture: {architecture}")
        print(f"  - IP Address: {ip_address}")
        print("-" * 40)

        self.last_batch_time = buffer.time

