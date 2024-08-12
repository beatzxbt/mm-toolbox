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
from mm_toolbox.moving_average.ema import EMA


def display_results(times, label):
    print(f"\n{label}")
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
    window = 5
    alpha = 0.5
    data = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    new_val = 6.0
    iters = 1_000_000
    runs = 5 

    # Benchmark for EMA initialization
    setup_code = f"""
from mm_toolbox.ema.ema import EMA
import numpy as np
window = {window}
alpha = {alpha}
data = np.array({data.tolist()})
ema = EMA(window, alpha, fast=False)
    """
    stmt_initialize = 'ema.initialize(data)'

    time_initialize = np.array(timeit.repeat(stmt=stmt_initialize, setup=setup_code, repeat=runs, number=iters)) * 1e9 / iters

    # Benchmark for EMA update
    setup_code = f"""
from mm_toolbox.ema.ema import EMA
import numpy as np
window = {window}
alpha = {alpha}
data = np.array({data.tolist()})
new_val = {new_val}
ema = EMA(window, alpha, fast=False)
ema.initialize(data)
    """
    stmt_update = 'ema.update(new_val)'

    time_update = np.array(timeit.repeat(stmt=stmt_update, setup=setup_code, repeat=runs, number=iters)) * 1e9 / iters

    # Collect and display results
    times = {
        'initialize': time_initialize,
        'update': time_update,
    }

    display_results(times, f"Benchmarking EMA | Iters: {iters} | Runs: {runs}")

