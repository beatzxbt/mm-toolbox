import numpy as np
from Cython.Build import cythonize

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

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
    'language_level': 3,
    'boundscheck': False,
    'wraparound': False,
    'embedsignature': True,
    'embedsignature.format': 'python',
    'cdivision': True,
    'cpow': True,
}
EXTRA_COMPILE_ARGS = ['-march=native', '-O3']

def get_extension(name, sources, include_dirs=None):
    """Helper to create extension with consistent settings"""
    return Extension(
        name=name,
        sources=sources,
        include_dirs=['src'] if not include_dirs else include_dirs,
        extra_compile_args=EXTRA_COMPILE_ARGS,
    )

#   * Hierarchy of build dependencies *
#   
#   - Rounding (depends on -)
#   - Time (depends on -)
#   - Ringbuffer (depends on -)
#   - Orderbook (depends on -)
#
#   - Logger (depends on Time)
#   - Candles (depends on Ringbuffer)
#   - Moving Average (depends on Ringbuffer & Time)
#   - Websocket (depends on Logger & Time)
def get_rounding_extensions():
    return [
        get_extension(
            name="mm_toolbox.rounding.rounder",
            sources=["src/mm_toolbox/rounding/rounder.pyx"],
            include_dirs=[np.get_include(), 'src'],
        ),
    ]
def get_time_extensions():
    return [
        get_extension(
            name="mm_toolbox.time.time",
            sources=[
                "src/mm_toolbox/time/time.pyx",
                "src/mm_toolbox/time/ctime_impl.c"
            ],
            include_dirs=["src/mm_toolbox/time", 'src'],
        ),
    ]
def get_ringbuffer_extensions():
    return [
        get_extension(
            name="mm_toolbox.ringbuffer.onedim",
            sources=["src/mm_toolbox/ringbuffer/onedim.pyx"],
            include_dirs=[np.get_include(), "src/mm_toolbox/ringbuffer", 'src'],
        ),
        get_extension(
            name="mm_toolbox.ringbuffer.twodim",
            sources=["src/mm_toolbox/ringbuffer/twodim.pyx"],
            include_dirs=[np.get_include(), "src/mm_toolbox/ringbuffer", 'src'],
        ),
        get_extension(
            name="mm_toolbox.ringbuffer.multi",
            sources=["src/mm_toolbox/ringbuffer/multi.pyx"],
            include_dirs=[np.get_include(), "src/mm_toolbox/ringbuffer", 'src'],
        ),
    ]

def get_orderbook_extensions():
    return [
        get_extension(
            name="mm_toolbox.orderbook.orderbook",
            sources=["src/mm_toolbox/orderbook/orderbook.pyx"],
            include_dirs=[np.get_include(), "src/mm_toolbox/orderbook", 'src'],
        ),
    ]

def get_logging_extensions():
    return [
        get_extension(
            name="mm_toolbox.logging.advanced.structs",
            sources=["src/mm_toolbox/logging/advanced/structs.pyx"],
            include_dirs=["src/mm_toolbox/logging/advanced", 'src'],
        ),
        get_extension(
            name="mm_toolbox.logging.advanced.config",
            sources=["src/mm_toolbox/logging/advanced/config.pyx"],
            include_dirs=["src/mm_toolbox/logging/advanced", 'src'],
        ),
        get_extension(
            name="mm_toolbox.logging.advanced.master",
            sources=["src/mm_toolbox/logging/advanced/master.pyx"],
            include_dirs=["src/mm_toolbox/logging/advanced", 'src'],
        ),
        get_extension(
            name="mm_toolbox.logging.advanced.worker",
            sources=["src/mm_toolbox/logging/advanced/worker.pyx"],
            include_dirs=["src/mm_toolbox/logging/advanced", 'src'],
        ),
    ]

def get_moving_average_extensions():
    return [
        get_extension(
            name="mm_toolbox.moving_average.base",
            sources=["src/mm_toolbox/moving_average/base.pyx"],
            include_dirs=[np.get_include(), "src/mm_toolbox/moving_average", 'src'],
        ),
        get_extension(
            name="mm_toolbox.moving_average.ema",
            sources=["src/mm_toolbox/moving_average/ema.pyx"],
            include_dirs=[np.get_include(), "src/mm_toolbox/moving_average", 'src'],
        ),
        get_extension(
            name="mm_toolbox.moving_average.sma",
            sources=["src/mm_toolbox/moving_average/sma.pyx"],
            include_dirs=[np.get_include(), "src/mm_toolbox/moving_average", 'src'],
        ),
        get_extension(
            name="mm_toolbox.moving_average.tema",
            sources=["src/mm_toolbox/moving_average/tema.pyx"],
            include_dirs=[np.get_include(), "src/mm_toolbox/moving_average", 'src'],
        ),
        get_extension(
            name="mm_toolbox.moving_average.wma",
            sources=["src/mm_toolbox/moving_average/wma.pyx"],
            include_dirs=[np.get_include(), "src/mm_toolbox/moving_average", 'src'],
        ),
    ]

def get_candles_extensions():
    return [
        get_extension(
            name="mm_toolbox.candles.base",
            sources=["src/mm_toolbox/candles/base.pyx"],
            include_dirs=[np.get_include(), 'src'],
        ),
        get_extension(
            name="mm_toolbox.candles.multi",
            sources=["src/mm_toolbox/candles/multi.pyx"],
            include_dirs=[np.get_include(), 'src'],
        ),
        get_extension(
            name="mm_toolbox.candles.time",
            sources=["src/mm_toolbox/candles/time.pyx"],
            include_dirs=[np.get_include(), 'src'],
        ),
        get_extension(
            name="mm_toolbox.candles.tick",
            sources=["src/mm_toolbox/candles/tick.pyx"],
            include_dirs=[np.get_include(), 'src'],
        ),
        get_extension(
            name="mm_toolbox.candles.volume",
            sources=["src/mm_toolbox/candles/volume.pyx"],
            include_dirs=[np.get_include(), 'src'],
        ),
        get_extension(
            name="mm_toolbox.candles.price",
            sources=["src/mm_toolbox/candles/price.pyx"],
            include_dirs=[np.get_include(), 'src'],
        ),
    ]

def get_websocket_extensions():
    return [
        get_extension(
            name="mm_toolbox.websocket.raw",
            sources=["src/mm_toolbox/websocket/raw.pyx"],
        ),
        get_extension(
            name="mm_toolbox.websocket.single",
            sources=["src/mm_toolbox/websocket/single.pyx"],
        ),
        get_extension(
            name="mm_toolbox.websocket.pool",
            sources=["src/mm_toolbox/websocket/pool.pyx"],
        ),
    ]

module_list: list[Extension] = []
module_list.extend(get_rounding_extensions())
module_list.extend(get_time_extensions())
module_list.extend(get_ringbuffer_extensions())
module_list.extend(get_orderbook_extensions())
# module_list.extend(get_moving_average_extensions())
# module_list.extend(get_candles_extensions())
# module_list.extend(get_logging_extensions())
# module_list.extend(get_websocket_extensions())

setup(
    cmdclass={'build_ext': build_ext},
    name="mm_toolbox",
    packages=find_packages(where='src'),
    package_dir={"": 'src'},
    ext_modules=cythonize(
        module_list=module_list,
        compiler_directives=COMPILER_DIRECTIVES,
    ),
)