# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

"""Common base for shared-memory ring buffers."""

import os
from libc.stdint cimport uint64_t as u64
from libc.stddef cimport size_t
from libc.errno cimport errno

from .header cimport ShmHeader, ShmMpscGlobalHeader, ShmSubRingHeader
from .memory cimport align_up, pow2_at_least

# Constants
cdef u64 _MAGIC = 0x53484252  # 'SHBR'
cdef u64 _MPSC_MAGIC = 0x53484D50  # 'SHMP'
cdef size_t _HEADER_ALIGN = 64
cdef size_t _HEADER_SIZE = align_up(sizeof(ShmHeader), _HEADER_ALIGN)
cdef size_t _MPSC_GLOBAL_HEADER_SIZE = align_up(sizeof(ShmMpscGlobalHeader), _HEADER_ALIGN)
cdef size_t _SUB_RING_HEADER_SIZE = align_up(sizeof(ShmSubRingHeader), _HEADER_ALIGN)


cdef class _ShmRingBase:
    """Common lifecycle for shared ring buffers."""

    def __cinit__(self) -> None:
        if os.name != "posix":
            raise OSError("Shared memory ringbuffer is only supported on POSIX platforms")
        self._base = NULL
        self._map_len = 0
        self._fd = -1
        self._owner = False
        self._unlink_on_close = False
        self._spin_wait = 1024
        self._path_py = None
        self._path = NULL

    cdef inline void _close_map(self):
        """Unmap and close backing file."""
        if self._base != NULL:
            munmap(self._base, self._map_len)
            self._base = NULL
            self._map_len = 0
        if self._fd >= 0:
            close(self._fd)
            self._fd = -1

    def __dealloc__(self):
        self._close_map()

    cpdef void close(self):
        self._close_map()
        if self._owner and self._unlink_on_close and self._path_py is not None:
            try:
                os.unlink(self._path_py)
            except Exception:
                pass
        self._owner = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
