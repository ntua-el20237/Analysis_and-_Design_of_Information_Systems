import dask.dataframe as dd
import pandas as pd
import time
import psutil
import os
import json



def dask_etl(csv_file, num_workers, output_json, baseline_runtime=None):

    from dask.distributed import Client, LocalCluster
    cluster = LocalCluster(n_workers=num_workers, threads_per_worker=1)
    client = Client(cluster)
    print(f"Started a new Dask cluster with {num_workers} workers.")


    process = psutil.Process()
    start_time = time.time()
    start_memory = process.memory_info().rss / (1024 ** 2)


    load_start = time.time()
    dataset = dd.read_csv(csv_file, blocksize="64MB")
    load_end = time.time()


    etl_start = time.time()
    grouped = dataset.groupby("category")[["x", "y"]].mean().compute()
    etl_end = time.time()

    end_time = time.time()
    end_memory = process.memory_info().rss / (1024 ** 2)


    runtime = end_time - start_time
    memory_usage = end_memory - start_memory
    load_time = load_end - load_start
    etl_time = etl_end - etl_start
    data_size_mb = os.path.getsize(csv_file) / (1024 ** 2)
    data_columns = list(dataset.columns)


    data_summary = {
        "mean_x": dataset["x"].mean().compute(),
        "mean_y": dataset["y"].mean().compute(),
        "min_x": dataset["x"].min().compute(),
        "min_y": dataset["y"].min().compute(),
        "max_x": dataset["x"].max().compute(),
        "max_y": dataset["y"].max().compute(),
        "count": len(dataset)
    }


    speedup = baseline_runtime / runtime if baseline_runtime else None
    efficiency = (baseline_runtime / runtime) / num_workers if baseline_runtime else None


    results = {
        "framework": "Dask",
        "csv_file": csv_file,
        "num_workers": num_workers,
        "runtime_seconds": runtime,
        "peak_memory_MB": memory_usage,
        "data_size_rows": data_summary["count"],
        "data_size_MB": data_size_mb,
        "data_columns": data_columns,
        "data_summary": data_summary,
        "result_sample": grouped.head().to_dict(),  # Convert sample to dict
        "etl_time_seconds": etl_time,
        "data_load_time_seconds": load_time,
        "speedup": speedup,
        "efficiency": efficiency
    }
    print(f"Results for {num_workers} workers: {results}")


    with open(output_json, "a") as f:
        json.dump(results, f)
        f.write("\n")

    # Shutdown Dask
    client.shutdown()

if __name__=="__main__":
 large_csv = "large_data1.csv"
 half1_csv = "half1_data1.csv"
 half2_csv = "half2_data1.csv"



 output_json = "dask_results_split.json"

 for csv_file in [large_csv, half2_csv]:
     baseline_runtime = None  # Reset baseline for each dataset half
     for workers in [2, 4, 8, 16]:
         dask_etl(csv_file, workers, output_json, baseline_runtime)
        
        # Update baseline_runtime after the first run
         if baseline_runtime is None:
             with open(output_json, "r") as f:
                 first_result = json.loads(f.readline())
                 baseline_runtime = first_result["runtime_seconds"]
 
