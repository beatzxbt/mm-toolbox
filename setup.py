import numpy as np
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
from setuptools.command.build_ext import build_ext

# Usage instructions
# 
# To build with optimizations:
#   python setup.py build_ext --inplace --optimize
# 
# To build with optimizations and parallel compilation:
#   python setup.py build_ext --inplace --optimize --parallel 4
# 
# To install the package with optimizations:
#   pip install . --install-option="--optimize"

OPTIMIZE = False

# Custom build_ext class to add the --optimize option
class OptimizedBuildExt(build_ext):
    user_options = build_ext.user_options + [
        ('optimize', None, 'Enable CPU-specific optimizations (SIMD, OpenMP)'),
    ]
    
    def initialize_options(self):
        super().initialize_options()
        self.optimize = None
    
    def finalize_options(self):
        super().finalize_options()
        global OPTIMIZE
        if self.optimize is not None:
            OPTIMIZE = True
            print("⚡ Building with CPU optimizations ⚡")
        else:
            print("Building without CPU optimizations. Use --optimize for better performance.")
            
# Helper function to create extensions with consistent settings
def get_extension(name, sources, include_dirs=None):
    """Helper to create extension with consistent settings"""
    if include_dirs is None:
        include_dirs = ["src"]
    if np and 'np' in locals():
        include_dirs.append(np.get_include())
    
    ext_kwargs = {}
    if OPTIMIZE:
        ext_kwargs.update({
            'extra_compile_args': ['-march=native', '-O3', '-fopenmp'],
            'extra_link_args': ['-fopenmp'],
        })
    
    return Extension(
        name=name,
        sources=sources,
        include_dirs=include_dirs,
        **ext_kwargs,
    )

#   * Hierarchy of build dependencies *
#
#   - Rounding (depends on nothing)
#   - Time (depends on nothing)
#   - Ringbuffer (depends on nothing)
#   - Orderbook (depends on nothing)
#
#   - Logger (depends on Time)
#   - Moving Average (depends on Ringbuffer)
#   - Candles (depends on Ringbuffer)
#
#   - Websocket (depends on Logger, Time)

def get_rounding_extensions():
    return [
        get_extension(
            name="mm_toolbox.rounding.round",
            sources=["src/mm_toolbox/rounding/round.pyx"],
            include_dirs=[np.get_include(), "src"],
        ),
    ]

def get_time_extensions():
    return [
        get_extension(
            name="mm_toolbox.time.time",
            sources=["src/mm_toolbox/time/time.pyx"],
            include_dirs=["src"],
        ),
    ]

def get_ringbuffer_extensions():
    return [
        get_extension(
            name="mm_toolbox.ringbuffer.onedim",
            sources=["src/mm_toolbox/ringbuffer/onedim.pyx"],
            include_dirs=[np.get_include(), "src"],
        ),
        get_extension(
            name="mm_toolbox.ringbuffer.twodim",
            sources=["src/mm_toolbox/ringbuffer/twodim.pyx"],
            include_dirs=[np.get_include(), "src"],
        ),
        get_extension(
            name="mm_toolbox.ringbuffer.multi",
            sources=["src/mm_toolbox/ringbuffer/multi.pyx"],
            include_dirs=[np.get_include(), "src"],
        ),
    ]

def get_orderbook_extensions():
    return [
        get_extension(
            name="mm_toolbox.orderbook.orderbook",
            sources=["src/mm_toolbox/orderbook/orderbook.pyx"],
            include_dirs=[np.get_include(), "src"],
        ),
    ]

def get_logging_extensions():
    return [
        get_extension(
            name="mm_toolbox.logging.advanced.structs",
            sources=["src/mm_toolbox/logging/advanced/structs.pyx"],
            include_dirs=[np.get_include(), "src"],
        ),
        get_extension(
            name="mm_toolbox.logging.advanced.config",
            sources=["src/mm_toolbox/logging/advanced/config.pyx"],
            include_dirs=[np.get_include(), "src"],
        ),
        get_extension(
            name="mm_toolbox.logging.advanced.master",
            sources=["src/mm_toolbox/logging/advanced/master.pyx"],
            include_dirs=[np.get_include(), "src"],
        ),
        get_extension(
            name="mm_toolbox.logging.advanced.worker",
            sources=["src/mm_toolbox/logging/advanced/worker.pyx"],
            include_dirs=[np.get_include(), "src"],
        ),
    ]

def get_moving_average_extensions():
    return [
        get_extension(
            name="mm_toolbox.moving_average.base",
            sources=["src/mm_toolbox/moving_average/base.pyx"],
            include_dirs=[np.get_include(), "src"],
        ),
        get_extension(
            name="mm_toolbox.moving_average.ema",
            sources=["src/mm_toolbox/moving_average/ema.pyx"],
            include_dirs=[np.get_include(), "src"],
        ),
        get_extension(
            name="mm_toolbox.moving_average.tema",
            sources=["src/mm_toolbox/moving_average/tema.pyx"],
            include_dirs=[np.get_include(), "src"],
        ),
        get_extension(
            name="mm_toolbox.moving_average.sma",
            sources=["src/mm_toolbox/moving_average/sma.pyx"],
            include_dirs=[np.get_include(), "src"],
        ),
        get_extension(
            name="mm_toolbox.moving_average.wma",
            sources=["src/mm_toolbox/moving_average/wma.pyx"],
            include_dirs=[np.get_include(), "src"],
        ),
    ]

def get_candles_extensions():
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
    return [
        get_extension(
            name="mm_toolbox.websocket.raw",
            sources=["src/mm_toolbox/websocket/raw.pyx"],
            include_dirs=[np.get_include(), "src"],
        ),
        get_extension(
            name="mm_toolbox.websocket.single",
            sources=["src/mm_toolbox/websocket/single.pyx"],
            include_dirs=[np.get_include(), "src"]
        ),
        get_extension(
            name="mm_toolbox.websocket.pool",
            sources=["src/mm_toolbox/websocket/pool.pyx"],
            include_dirs=[np.get_include(), "src"],
        ),
        get_extension(
            name="mm_toolbox.websocket.tools",
            sources=["src/mm_toolbox/websocket/tools.pyx"],
            include_dirs=[np.get_include(), "src"],
        ),
    ]


module_list: list[Extension] = []
module_list.extend(get_rounding_extensions())
module_list.extend(get_time_extensions())
module_list.extend(get_ringbuffer_extensions())
module_list.extend(get_logging_extensions())
module_list.extend(get_moving_average_extensions())
module_list.extend(get_candles_extensions())
module_list.extend(get_orderbook_extensions())
# module_list.extend(get_websocket_extensions())

# Use the below command to build the extensions in parallel
# 'python setup.py build_ext --inplace --parallel 4'
setup(
    name="mm_toolbox",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=cythonize(
        module_list=module_list,
        build_dir="build",
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "nonecheck": False,
        },
    ),
    cmdclass={'build_ext': OptimizedBuildExt},
)