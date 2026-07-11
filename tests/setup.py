"""Setup script for compiling Cython native test modules.

This is separate from the main library setup.py to keep test build
configuration isolated. Handles all Cython/C native tests across the
entire test suite.

Run with:
    python tests/setup.py build_ext --inplace
Or:
    make build-test
"""

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext
from Cython.Build import cythonize
import numpy as np
import os
import sys

# Add src to path so we can import the library
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Compiler flags (match main library for consistency)
EXTRA_COMPILE_ARGS = ["-O3", "-march=native"]
EXTRA_LINK_ARGS = []

ORDERBOOK_ADVANCED_INCLUDE_DIRS = [
    np.get_include(),
    "../src/mm_toolbox/orderbook/advanced",
    "../src/mm_toolbox/orderbook/advanced/c",
    "../src/mm_toolbox/orderbook/advanced/enum",
    "../src/mm_toolbox/orderbook/advanced/level",
    "../src/mm_toolbox/orderbook/advanced/ladder",
    "../src",
    ".",  # Current dir (tests/) for engine_helpers.pxd
    "orderbook/advanced/engine",
]

orderbook_advanced_extensions = [
    Extension(
        name="cython_test_level",
        sources=["orderbook/advanced/engine/test_level.pyx"],
        include_dirs=ORDERBOOK_ADVANCED_INCLUDE_DIRS,
        extra_compile_args=EXTRA_COMPILE_ARGS,
        extra_link_args=EXTRA_LINK_ARGS,
    ),
    Extension(
        name="cython_test_ladder",
        sources=["orderbook/advanced/engine/test_ladder.pyx"],
        include_dirs=ORDERBOOK_ADVANCED_INCLUDE_DIRS,
        extra_compile_args=EXTRA_COMPILE_ARGS,
        extra_link_args=EXTRA_LINK_ARGS,
    ),
    Extension(
        name="cython_test_core",
        sources=["orderbook/advanced/engine/test_core.pyx"],
        include_dirs=ORDERBOOK_ADVANCED_INCLUDE_DIRS,
        extra_compile_args=EXTRA_COMPILE_ARGS,
        extra_link_args=EXTRA_LINK_ARGS,
    ),
    Extension(
        name="cython_test_wrapper",
        sources=["orderbook/advanced/wrapper_cython/test_wrapper.pyx"],
        include_dirs=ORDERBOOK_ADVANCED_INCLUDE_DIRS,
        extra_compile_args=EXTRA_COMPILE_ARGS,
        extra_link_args=EXTRA_LINK_ARGS,
    ),
]

WEBSOCKET_INCLUDE_DIRS = [
    np.get_include(),
    "../src",
    ".",
]

websocket_extensions = [
    Extension(
        name="cython_test_websocket_connection",
        sources=["websocket/connection/test_native.pyx"],
        include_dirs=WEBSOCKET_INCLUDE_DIRS,
        extra_compile_args=EXTRA_COMPILE_ARGS,
        extra_link_args=EXTRA_LINK_ARGS,
    ),
]

RINGBUFFER_INCLUDE_DIRS = [
    np.get_include(),
    "../src",
    ".",
    "ringbuffer",
]

ringbuffer_extensions = [
    Extension(
        name="cython_test_memory",
        sources=["ringbuffer/test_memory.pyx"],
        include_dirs=RINGBUFFER_INCLUDE_DIRS,
        extra_compile_args=EXTRA_COMPILE_ARGS,
        extra_link_args=EXTRA_LINK_ARGS,
    ),
]

all_extensions: list[Extension] = []
all_extensions.extend(orderbook_advanced_extensions)
all_extensions.extend(websocket_extensions)
all_extensions.extend(ringbuffer_extensions)

# Collect all .pyx source files from extensions before cythonization
pyx_files: list[str] = []
for ext in all_extensions:
    for src in ext.sources:
        if src.endswith(".pyx"):
            pyx_files.append(src)


class build_ext(_build_ext):
    """Custom build_ext to remove generated .c files after build."""

    def run(self):
        """Run the build_ext command and clean up generated C files."""
        super().run()
        # Remove all generated .c files from cythonized .pyx sources
        for pyx_file in pyx_files:
            c_file = pyx_file.replace(".pyx", ".c")
            if os.path.exists(c_file):
                try:
                    os.remove(c_file)
                    print(f"Cleaned up: {c_file}")
                except Exception as e:
                    print(f"Warning: Could not remove {c_file}: {e}")


setup(
    name="mm_toolbox_tests",
    cmdclass={"build_ext": build_ext},
    ext_modules=cythonize(
        all_extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
    ),
)
