import sys
import os

# Get the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Add the project root directory to the Python path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# -------------------------------------------- #

# This BTCUSDT data is from Binance USD-M, scraped by me (feel free to use it elsewhere)
# Tick & lot size at the time of testing is 0.1/0.001 respectively
# This benchmark can be ran with other data, just swap keys & data directory
# Can work any sizes, though it's optimized for smaller orderbooks (~50 levels)

import json
import numpy as np
from time import time_ns

from mm_toolbox.orderbook.orderbook import Orderbook

def load_benchmark_data(dir: str):
    with open(dir, 'r') as file:
        for line in file:
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                print(f"Faulty line: {line}")

try:
    print("starting script, loading data & initializing orderbook...")

    data_generator = load_benchmark_data("/Users/beatz/Documents/Github/mm-toolbox/mm_tests/orderbook/BTCUSDT_filtered_out.txt")

    first_update = next(data_generator)
    orderbook = Orderbook(tick_size=0.1, lot_size=0.001, num_levels=50)
    orderbook.ingest_l2_update(
        first_update["ts"],
        np.array(first_update["asks"], dtype=np.float64),
        np.array(first_update["bids"], dtype=np.float64)
    )

    print(f"first update: {first_update}")

    l2_times = []
    trades_times = []
    updates_processed = 0

    # orderbook.display_internal(50)

    print("starting benchmark...")

    for update in data_generator:
        if "trade" in update:
            t1 = time_ns()
            orderbook.ingest_trade_update(
                update["ts"], 
                bool(update["trade"][1]),
                update["trade"][2],
                update["trade"][3]
            )
            t2 = time_ns()
            trades_times.append(t2-t1)
        
        else:
            t1 = time_ns()
            orderbook.ingest_l2_update(
                update["ts"],
                np.array(update["asks"], dtype=np.float64),
                np.array(update["bids"], dtype=np.float64)
            )
            t2 = time_ns()
            l2_times.append(t2-t1)

        updates_processed += 1

    print("completed benchmark, calculating performance...\n\n")

except Exception as e:
    print(f"exception: {e}")
    # print(f"snapshot at point of failure: \n bids: {orderbook.bids}\n asks: {orderbook.asks}\n timestamp: {orderbook.last_updated_timestamp}\n")
    print(f"last known update: {update}")
    print(f"completed updates before failure: {updates_processed}")

finally:
    l2 = np.array(l2_times, dtype=np.int64)
    trades = np.array(trades_times, dtype=np.int64)
    total = np.concatenate([l2, trades], axis=None)

    l2_mean = np.mean(l2)
    l2_std = np.std(l2)
    l2_p50 = np.percentile(l2, 50)
    l2_p75 = np.percentile(l2, 75)
    l2_p90 = np.percentile(l2, 90)
    l2_99 = np.percentile(l2, 99)

    trades_mean = np.mean(trades)
    trades_std = np.std(trades)
    trades_p50 = np.percentile(trades, 50)
    trades_p75 = np.percentile(trades, 75)
    trades_p90 = np.percentile(trades, 90)
    trades_99 = np.percentile(trades, 99)

    total_mean = np.mean(total)
    total_std = np.std(total)
    total_p50 = np.percentile(total, 50)
    total_p75 = np.percentile(total, 75)
    total_p90 = np.percentile(total, 90)
    total_99 = np.percentile(total, 99)

    # Display headers
    print(f"{'Metric':<10} | {'L2':>10} | {'Trades':>10} | {'Total':>10}")
    print('-' * 50)

    # Display data
    print(f"{'Mean':<10} | {l2_mean:>10.1f} | {trades_mean:>10.1f} | {total_mean:>10.1f}")
    print(f"{'Std Dev':<10} | {l2_std:>10.1f} | {trades_std:>10.1f} | {total_std:>10.1f}")
    print(f"{'50th %':<10} | {l2_p50:>10.1f} | {trades_p50:>10.1f} | {total_p50:>10.1f}")
    print(f"{'75th %':<10} | {l2_p75:>10.1f} | {trades_p75:>10.1f} | {total_p75:>10.1f}")
    print(f"{'90th %':<10} | {l2_p90:>10.1f} | {trades_p90:>10.1f} | {total_p90:>10.1f}")
    print(f"{'99th %':<10} | {l2_99:>10.1f} | {trades_99:>10.1f} | {total_99:>10.1f}")