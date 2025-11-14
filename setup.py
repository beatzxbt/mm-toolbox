import contextlib
import os
import platform
import sys

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext as _build_ext

"""
# Usage instructions
#
# To build
#   'python setup.py build_ext --inplace'
#   'python setup.py build_ext --inplace --parallel 4'
#
# To install the package
#   'pip install .'
"""

COMPILER_DIRECTIVES = {
    "language_level": 3,
    "boundscheck": False,
    "wraparound": False,
    "embedsignature": True,
    "embedsignature.format": "python",
    "cdivision": True,
    "cpow": True,
}
USE_OFAST = os.environ.get("MM_TOOLBOX_OFAST", "0").lower() in {"1", "true", "yes"}
USE_LTO = os.environ.get("MM_TOOLBOX_LTO", "0").lower() in {"1", "true", "yes"}
USE_FAST_MATH = os.environ.get("MM_TOOLBOX_FAST_MATH", "0").lower() in {"1", "true", "yes"}

EXTRA_COMPILE_ARGS = [
    "-march=native",
    "-Ofast" if USE_OFAST else "-O3",
    "-ffast-math" if USE_FAST_MATH else "",
    "-Wno-sign-compare",
    "-Wno-unused-function",
    "-Wno-unreachable-code",
]
EXTRA_COMPILE_ARGS = [arg for arg in EXTRA_COMPILE_ARGS if arg]

EXTRA_LINK_ARGS = ["-Wl,-w"] if sys.platform == "darwin" else []
if USE_LTO:
    EXTRA_COMPILE_ARGS.append("-flto")
    EXTRA_LINK_ARGS.append("-flto")


def get_extension(name, sources, include_dirs=None):
    """Helper to create extension with consistent settings."""
    return Extension(
        name=name,
        sources=sources,
        include_dirs=include_dirs if include_dirs else ["src"],
        extra_compile_args=EXTRA_COMPILE_ARGS,
        extra_link_args=EXTRA_LINK_ARGS,
    )


def get_build_dir():
    # Example: build/cython.linux-x86_64-cpython-313
    plat = platform.system().lower()
    machine = platform.machine().lower()
    impl = platform.python_implementation().lower()
    # e.g. cpython-313
    py_tag = (
        f"{impl}-"
        f"{sys.version_info.major}"
        f"{sys.version_info.minor}"
        f"{sys.version_info.micro}"
    )
    return f"build/cython.{plat}-{machine}-{py_tag}"


#   * Hierarchy of build dependencies *
#
#   - Rounding, Time, Ringbuffer, Orderbook (depends on -)
#
#   - Logger, Websocket (depends on Time)
#   - Candles (depends on Ringbuffer)
#   - Moving Average (depends on Ringbuffer & Time)
def get_rounding_extensions():
    """Get list of rounding extensions for compilation."""
    return [
        get_extension(
            name="mm_toolbox.rounding.rounder",
            sources=["src/mm_toolbox/rounding/rounder.pyx"],
            include_dirs=[np.get_include(), "src"],
        ),
    ]


def get_time_extensions():
    """Get list of time extensions for compilation."""
    return [
        get_extension(
            name="mm_toolbox.time.time",
            sources=[
                "src/mm_toolbox/time/ctime_impl.c",
                "src/mm_toolbox/time/time.pyx",
            ],
            include_dirs=["src/mm_toolbox/time", "src"],
        ),
    ]


def get_ringbuffer_extensions():
    """Get list of ringbuffer extensions for compilation."""
    return [
        get_extension(
            name="mm_toolbox.ringbuffer.generic",
            sources=["src/mm_toolbox/ringbuffer/generic.pyx"],
            include_dirs=[np.get_include(), "src/mm_toolbox/ringbuffer", "src"],
        ),
        get_extension(
            name="mm_toolbox.ringbuffer.numeric",
            sources=["src/mm_toolbox/ringbuffer/numeric.pyx"],
            include_dirs=[np.get_include(), "src/mm_toolbox/ringbuffer", "src"],
        ),
        get_extension(
            name="mm_toolbox.ringbuffer.bytes",
            sources=["src/mm_toolbox/ringbuffer/bytes.pyx"],
            include_dirs=[np.get_include(), "src/mm_toolbox/ringbuffer", "src"],
        ),
    ]


def get_orderbook_extensions():
    """Get list of orderbook extensions for compilation."""
    return [
        get_extension(
            name="mm_toolbox.orderbook.corderbook.level",
            sources=["src/mm_toolbox/orderbook/corderbook/level.pyx"],
            include_dirs=[
                np.get_include(),
                "src/mm_toolbox/orderbook/corderbook",
                "src",
            ],
        ),
        get_extension(
            name="mm_toolbox.orderbook.corderbook.side",
            sources=["src/mm_toolbox/orderbook/corderbook/side.pyx"],
            include_dirs=[
                np.get_include(),
                "src/mm_toolbox/orderbook/corderbook",
                "src",
            ],
        ),
        get_extension(
            name="mm_toolbox.orderbook.corderbook.corderbook",
            sources=["src/mm_toolbox/orderbook/corderbook/corderbook.pyx"],
            include_dirs=[
                np.get_include(),
                "src/mm_toolbox/orderbook/corderbook",
                "src",
            ],
        ),
    ]


def get_logging_extensions():
    """Get list of logging extensions for compilation."""
    return [
        get_extension(
            name="mm_toolbox.logging.advanced.config",
            sources=["src/mm_toolbox/logging/advanced/config.pyx"],
            include_dirs=["src/mm_toolbox/logging/advanced", "src"],
        ),
        get_extension(
            name="mm_toolbox.logging.advanced.protocol",
            sources=["src/mm_toolbox/logging/advanced/protocol.pyx"],
            include_dirs=["src/mm_toolbox/logging/advanced", "src"],
        ),
        get_extension(
            name="mm_toolbox.logging.advanced.master",
            sources=["src/mm_toolbox/logging/advanced/master.pyx"],
            include_dirs=["src/mm_toolbox/logging/advanced", "src"],
        ),
        get_extension(
            name="mm_toolbox.logging.advanced.worker",
            sources=["src/mm_toolbox/logging/advanced/worker.pyx"],
            include_dirs=["src/mm_toolbox/logging/advanced", "src"],
        ),
    ]


def get_moving_average_extensions():
    """Get list of moving average extensions for compilation."""
    return [
        get_extension(
            name="mm_toolbox.moving_average.base",
            sources=["src/mm_toolbox/moving_average/base.pyx"],
            include_dirs=[np.get_include(), "src/mm_toolbox/moving_average", "src"],
        ),
        get_extension(
            name="mm_toolbox.moving_average.ema",
            sources=["src/mm_toolbox/moving_average/ema.pyx"],
            include_dirs=[np.get_include(), "src/mm_toolbox/moving_average", "src"],
        ),
        get_extension(
            name="mm_toolbox.moving_average.sma",
            sources=["src/mm_toolbox/moving_average/sma.pyx"],
            include_dirs=[np.get_include(), "src/mm_toolbox/moving_average", "src"],
        ),
        get_extension(
            name="mm_toolbox.moving_average.tema",
            sources=["src/mm_toolbox/moving_average/tema.pyx"],
            include_dirs=[np.get_include(), "src/mm_toolbox/moving_average", "src"],
        ),
        get_extension(
            name="mm_toolbox.moving_average.wma",
            sources=["src/mm_toolbox/moving_average/wma.pyx"],
            include_dirs=[np.get_include(), "src/mm_toolbox/moving_average", "src"],
        ),
    ]


def get_candles_extensions():
    """Get list of candles extensions for compilation."""
    return [
        get_extension(
            name="mm_toolbox.candles.base",
            sources=["src/mm_toolbox/candles/base.pyx"],
            include_dirs=[np.get_include(), "src"],
        ),
        get_extension(
            name="mm_toolbox.candles.multi",
            sources=["src/mm_toolbox/candles/multi.pyx"],
            include_dirs=[np.get_include(), "src"],
        ),
        get_extension(
            name="mm_toolbox.candles.time",
            sources=["src/mm_toolbox/candles/time.pyx"],
            include_dirs=[np.get_include(), "src"],
        ),
        get_extension(
            name="mm_toolbox.candles.tick",
            sources=["src/mm_toolbox/candles/tick.pyx"],
            include_dirs=[np.get_include(), "src"],
        ),
        get_extension(
            name="mm_toolbox.candles.volume",
            sources=["src/mm_toolbox/candles/volume.pyx"],
            include_dirs=[np.get_include(), "src"],
        ),
        get_extension(
            name="mm_toolbox.candles.price",
            sources=["src/mm_toolbox/candles/price.pyx"],
            include_dirs=[np.get_include(), "src"],
        ),
    ]


def get_websocket_extensions():
    """Get list of websocket extensions for compilation."""
    return [
        get_extension(
            name="mm_toolbox.websocket.connection",
            sources=["src/mm_toolbox/websocket/connection.pyx"],
            include_dirs=[np.get_include(), "src"],
        ),
    ]


# def get_misc_extensions():
#     """Get list of misc extensions for compilation."""
#     return [
#         get_extension(
#             name="mm_toolbox.misc.limiter.core",
#             sources=["src/mm_toolbox/misc/limiter/core.pyx"],
#             include_dirs=["src/mm_toolbox/misc/limiter", "src"],
#         ),
#     ]


# def get_parsers_extensions():
#     """Get list of parsers extensions for compilation."""
#     return [
#         get_extension(
#             name="mm_toolbox.misc.parsers.json.fastjson",
#             sources=["src/mm_toolbox/misc/parsers/json/fastjson.pyx"],
#             include_dirs=["src"],
#         ),
#         get_extension(
#             name="mm_toolbox.misc.parsers.crypto.binance._bbo_cache",
#             sources=["src/mm_toolbox/misc/parsers/crypto/binance/_bbo_cache.pyx"],
#             include_dirs=["src"],
#         ),
#         get_extension(
#             name="mm_toolbox.misc.parsers.crypto.binance.binance_tob_parser",
#             sources=["src/mm_toolbox/misc/parsers/crypto/binance/binance_tob_parser.pyx"],
#             include_dirs=["src"],
#         ),
#     ]


module_list: list[Extension] = []
module_list.extend(get_rounding_extensions())
module_list.extend(get_time_extensions())
module_list.extend(get_ringbuffer_extensions())
# module_list.extend(get_orderbook_extensions())
module_list.extend(get_moving_average_extensions())
module_list.extend(get_candles_extensions())
module_list.extend(get_logging_extensions())
module_list.extend(get_websocket_extensions())
# # module_list.extend(get_misc_extensions())
# module_list.extend(get_parsers_extensions())


class BuildExt(_build_ext):
    """Custom build_ext to remove generated .c files after build."""

    def run(self):
        """Run the build_ext command and clean up generated C files."""
        super().run()
        # Remove all generated .c files from cythonized .pyx sources
        for ext in self.extensions:
            for src in ext.sources:
                if src.endswith(".pyx"):
                    c_file = src.replace(".pyx", ".c")
                    if os.path.exists(c_file):
                        with contextlib.suppress(Exception):
                            os.remove(c_file)


setup(
    cmdclass={"build_ext": BuildExt},
    name="mm_toolbox",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={
        "mm_toolbox": [
            "**/*.pxd",
            "**/*.pyi",
        ]
    },
    ext_modules=cythonize(
        module_list=module_list,
        compiler_directives=COMPILER_DIRECTIVES,
        build_dir=get_build_dir(),
    ),
)
