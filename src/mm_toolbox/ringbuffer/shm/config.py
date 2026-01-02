"""Configuration helpers for shared-memory ring buffers."""

from msgspec import Struct


class ShmRingBufferConfig(Struct):
    """Configuration for shared-memory bytes ring buffer."""

    path: str
    capacity_bytes: int
    create: bool = True
    unlink_on_close: bool = False
    spin_wait: int = 1024

    def __post_init__(self) -> None:
        """Validate shared-memory configuration."""
        if not self.path:
            raise ValueError("path must be non-empty")
        if self.capacity_bytes <= 0:
            raise ValueError("capacity_bytes must be > 0")
        if self.spin_wait <= 0:
            raise ValueError("spin_wait must be > 0")
        if self.unlink_on_close and not self.create:
            raise ValueError("unlink_on_close requires create=True")

    @classmethod
    def default(cls) -> "ShmRingBufferConfig":
        """Create a default shared-memory configuration."""
        return cls(
            path="/tmp/shm_ring.bin",
            capacity_bytes=1 << 16,
            create=True,
            unlink_on_close=False,
            spin_wait=1024,
        )

    def producer_kwargs(self) -> dict[str, object]:
        """Return kwargs suitable for SharedBytesRingBufferProducer."""
        return {
            "path": self.path,
            "capacity_bytes": self.capacity_bytes,
            "create": self.create,
            "unlink_on_close": self.unlink_on_close,
            "spin_wait": self.spin_wait,
        }

    def consumer_kwargs(self) -> dict[str, object]:
        """Return kwargs suitable for SharedBytesRingBufferConsumer."""
        return {
            "path": self.path,
            "spin_wait": self.spin_wait,
        }
