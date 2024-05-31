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
from mm_toolbox.rounding.rounding import round_ceil, round_floor, round_discrete

def display_results(times, label):
    print(f"\n{label}")
    print(f"{'Operation':<20} | {'Mean (ns)':>12} | {'Std Dev (ns)':>12} | {'50th % (ns)':>12} | {'75th % (ns)':>12} | {'90th % (ns)':>12} | {'99th % (ns)':>12}")
    print('-' * 125)
    for op, time_list in times.items():
        mean = np.mean(time_list)
        std_dev = np.std(time_list)
        p50 = np.percentile(time_list, 50)
        p75 = np.percentile(time_list, 75)
        p90 = np.percentile(time_list, 90)
        p99 = np.percentile(time_list, 99)

        print(f"{op:<20} | {mean:>12.1f} | {std_dev:>12.1f} | {p50:>12.1f} | {p75:>12.1f} | {p90:>12.1f} | {p99:>12.1f}")

if __name__ == '__main__':
    num = np.random.random() * 100
    step_size = 0.1
    array_num = np.random.random(100) * 100
    array_step_size = 0.1
    iters = 1_000_000
    runs = 5

    # Benchmark for round_ceil with single number
    setup_code_num = f"""
from mm_toolbox.rounding.rounding import round_ceil
num = {num}
step_size = {step_size}
    """
    stmt_ceil_num = 'round_ceil(num, step_size)'

    time_ceil_num = np.array(timeit.repeat(stmt=stmt_ceil_num, setup=setup_code_num, repeat=runs, number=iters)) * 1e9 / iters

    # Benchmark for round_ceil with array
    setup_code_array = f"""
from mm_toolbox.rounding.rounding import round_ceil
import numpy as np
array_num = np.array({array_num.tolist()})
array_step_size = {array_step_size}
    """
    stmt_ceil_array = 'round_ceil(array_num, array_step_size)'

    time_ceil_array = np.array(timeit.repeat(stmt=stmt_ceil_array, setup=setup_code_array, repeat=runs, number=iters)) * 1e9 / iters

    # Benchmark for round_floor with single number
    setup_code_num = f"""
from mm_toolbox.rounding.rounding import round_floor
num = {num}
step_size = {step_size}
    """
    stmt_floor_num = 'round_floor(num, step_size)'

    time_floor_num = np.array(timeit.repeat(stmt=stmt_floor_num, setup=setup_code_num, repeat=runs, number=iters)) * 1e9 / iters

    # Benchmark for round_floor with array
    setup_code_array = f"""
from mm_toolbox.rounding.rounding import round_floor
import numpy as np
array_num = np.array({array_num.tolist()})
array_step_size = {array_step_size}
    """
    stmt_floor_array = 'round_floor(array_num, array_step_size)'

    time_floor_array = np.array(timeit.repeat(stmt=stmt_floor_array, setup=setup_code_array, repeat=runs, number=iters)) * 1e9 / iters

    # Benchmark for round_discrete with single number
    setup_code_num = f"""
from mm_toolbox.rounding.rounding import round_discrete
num = {num}
step_size = {step_size}
    """
    stmt_discrete_num = 'round_discrete(num, step_size)'

    time_discrete_num = np.array(timeit.repeat(stmt=stmt_discrete_num, setup=setup_code_num, repeat=runs, number=iters)) * 1e9 / iters

    # Benchmark for round_discrete with array
    setup_code_array = f"""
from mm_toolbox.rounding.rounding import round_discrete
import numpy as np
array_num = np.array({array_num.tolist()})
array_step_size = {array_step_size}
    """
    stmt_discrete_array = 'round_discrete(array_num, array_step_size)'

    time_discrete_array = np.array(timeit.repeat(stmt=stmt_discrete_array, setup=setup_code_array, repeat=runs, number=iters)) * 1e9 / iters

    # Collect and display results
    times = {
        'round_ceil_num': time_ceil_num,
        'round_floor_num': time_floor_num,
        'round_discrete_num': time_discrete_num,
        'round_ceil_array': time_ceil_array,
        'round_floor_array': time_floor_array,
        'round_discrete_array': time_discrete_array,
    }

    display_results(times, f"Benchmarking Rounding Functions | Iters: {iters} | Runs: {runs}")