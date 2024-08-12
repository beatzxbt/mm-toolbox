import sys
import os

# Get the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Add the project root directory to the Python path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# -------------------------------------------- #

import timeit
import numpy as np
from mm_toolbox.ringbuffer.ringbuffer import RingBufferF64, RingBufferI64

def benchmark_ringbuffer(buffer_class, capacity, iters, repeat):
    setup_code = f"""
from mm_toolbox.ringbuffer.ringbuffer import {buffer_class.__name__}
buffer = {buffer_class.__name__}({capacity})
    """
    statements = {
        'appendright': 'buffer.appendright(1)',
        'appendleft': 'buffer.appendleft(1)',
        'popright': 'buffer.appendright(1); buffer.popright()',
        'popleft': 'buffer.appendright(1); buffer.popleft()'
    }

    times = {op: np.array(timeit.repeat(stmt=stmt, setup=setup_code, repeat=repeat, number=iters)) * 1e9 / iters
             for op, stmt in statements.items()}

    return times

def display_results(times):
    print(f"{'Operation':<15} | {'Mean (ns)':>12} | {'Std Dev (ns)':>12} | {'50th % (ns)':>12} | {'75th % (ns)':>12} | {'90th % (ns)':>12} | {'99th % (ns)':>12}")
    print('-' * 110)
    for op, time_list in times.items():
        mean = np.mean(time_list)
        std_dev = np.std(time_list)
        p50 = np.percentile(time_list, 50)
        p75 = np.percentile(time_list, 75)
        p90 = np.percentile(time_list, 90)
        p99 = np.percentile(time_list, 99)

        print(f"{op:<15} | {mean:>12.1f} | {std_dev:>12.1f} | {p50:>12.1f} | {p75:>12.1f} | {p90:>12.1f} | {p99:>12.1f}")

if __name__ == '__main__':
    capacity = 1000
    iters = 1_000_000
    runs = 5
    
    print(f"Benchmarking RingBufferF64 | Iters: {iters} | Runs: {runs}")
    times_f64 = benchmark_ringbuffer(RingBufferF64, capacity, iters, runs)
    display_results(times_f64)

    print(f"\n\nBenchmarking RingBufferI64 | Iters: {iters} | Runs: {runs}")
    times_i64 = benchmark_ringbuffer(RingBufferI64, capacity, iters, runs)
    display_results(times_i64)
