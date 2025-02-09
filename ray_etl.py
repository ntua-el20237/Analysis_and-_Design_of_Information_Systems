import pandas as pd
import ray
import ray.data
import time
import psutil
import os
import json

# Function to split the large dataset into smaller parts
def split_dataset(csv_file, output_file1, output_file2):
    df = pd.read_csv(csv_file)
    mid_index = len(df) // 2
    df.iloc[:mid_index].to_csv(output_file1, index=False)
    df.iloc[mid_index:].to_csv(output_file2, index=False)
    print(f"Dataset split into {output_file1} and {output_file2}")

def ray_etl(csv_file, num_workers, output_json, baseline_runtime=None):
    try:
        ray.init(address="auto")
        print("Connected to an existing Ray cluster.")
    except ConnectionError:
        ray.init(num_cpus=num_workers)
        print(f"Started a new Ray cluster with {num_workers} workers.")

    # Monitor memory and runtime
    process = psutil.Process()
    start_time = time.time()
    start_memory = process.memory_info().rss / (1024 ** 2)

    # Load the CSV file
    load_start = time.time()
    dataset = ray.data.read_csv(csv_file, parallelism=200)
    load_end = time.time()

    # Perform ETL operations
    etl_start = time.time()
    grouped = dataset.groupby("category").mean(["x", "y"]).take_all()
    etl_end = time.time()

    end_time = time.time()
    end_memory = process.memory_info().rss / (1024 ** 2)

    # Metrics calculation
    runtime = end_time - start_time
    memory_usage = end_memory - start_memory
    load_time = load_end - load_start
    etl_time = etl_end - etl_start
    data_size_mb = os.path.getsize(csv_file) / (1024 ** 2)
    data_columns = dataset.schema().names

    # Dataset summary
    data_summary = {
        "mean_x": dataset.mean("x"),
        "mean_y": dataset.mean("y"),
        "min_x": dataset.min("x"),
        "min_y": dataset.min("y"),
        "max_x": dataset.max("x"),
        "max_y": dataset.max("y"),
        "count": dataset.count()
    }

    # Speedup and efficiency
    speedup = baseline_runtime / runtime if baseline_runtime else None
    efficiency = (baseline_runtime / runtime) / num_workers if baseline_runtime else None

    # Results dictionary
    results = {
        "framework": "Ray",
        "csv_file": csv_file,
        "num_workers": num_workers,
        "runtime_seconds": runtime,
        "peak_memory_MB": memory_usage,
        "data_size_rows": dataset.count(),
        "data_size_MB": data_size_mb,
        "data_columns": data_columns,
        "data_summary": data_summary,
        "result_sample": grouped[:5],
        "etl_time_seconds": etl_time,
        "data_load_time_seconds": load_time,
        "speedup": speedup,
        "efficiency": efficiency
    }
    print(f"Results for {num_workers} workers: {results}")

    # Write results to JSON file
    with open(output_json, "a") as f:
        json.dump(results, f)
        f.write("\n")

    # Shutdown Ray
    ray.shutdown()

# Split the large dataset into halves
large_csv = "large_data1.csv"
half1_csv = "half1_data1.csv"
half2_csv = "half2_data1.csv"
split_dataset(large_csv, half1_csv, half2_csv)

# Main script execution
output_json = "ray_results_split.json"

for csv_file in [large_csv, half2_csv]:
    baseline_runtime = None  # Reset baseline for each dataset half
    for workers in [2, 4, 8, 16]:
        ray_etl(csv_file, workers, output_json, baseline_runtime)
        
        # Update baseline_runtime after the first run
        if baseline_runtime is None:
            with open(output_json, "r") as f:
                first_result = json.loads(f.readline())
                baseline_runtime = first_result["runtime_seconds"]

