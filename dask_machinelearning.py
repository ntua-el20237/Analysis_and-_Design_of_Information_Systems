#!/usr/bin/env python3

import dask.dataframe as dd
import pandas as pd
import time
import psutil
import dask_ml
import os
import json
import numpy as np

from dask.distributed import Client, LocalCluster
from dask_ml.model_selection import train_test_split
from dask_ml.linear_model import LogisticRegression
from dask_ml.metrics import accuracy_score


def dask_ml(csv_file, num_workers, output_json, baseline_runtime=None):
    """
    Trains a logistic regression model using Dask-ML on the 'csv_file'.
    Assumes the dataset contains columns: ['id', 'x', 'y', 'category'].
    Logs results (including runtime, memory usage, accuracy, etc.) in JSON format.
    """

    # ----------------------------
    # 1) Start/Connect to Cluster
    # ----------------------------
    print("connect to cluster")
    cluster = LocalCluster(n_workers=num_workers, threads_per_worker=1)
    client = Client(cluster)
    print(f"Started a new Dask cluster with {num_workers} workers for ML job.")

    # ----------------------------
    # 2) Performance Tracking
    # ----------------------------
    print("performance tracking")
    process = psutil.Process()
    start_time = time.time()
    start_memory = process.memory_info().rss / (1024 ** 2)

    # ----------------------------
    # 3) Load Data
    # ----------------------------
    print("load data")
    load_start = time.time()
    df = dd.read_csv(csv_file, blocksize="64MB")
    load_end = time.time()

    # ----------------------------
    # 4) Feature Engineering / Preprocessing
    # ----------------------------
    print("preprocessing")
    df['category'] = df['category'].astype('category')
    df['category'] = df['category'].cat.as_known()
    df['category_code'] = df['category'].cat.codes

    # ----------------------------
    # 5) Train/Test Split
    # ----------------------------
    print("train/test split")
    X = df[['x', 'y']]
    y = df['category_code']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    X_train_array = X_train.to_dask_array(lengths=True)
    y_train_array = y_train.to_dask_array(lengths=True)
    
    X_test_array = X_test.to_dask_array(lengths=True)
    y_test_array = y_test.to_dask_array(lengths=True)
    
    


    # ----------------------------
    # 6) Training
    # ----------------------------
    print("training")
    ml_start = time.time()
    model = LogisticRegression()
    model.fit(X_train_array, y_train_array)
    ml_end = time.time()

    # ----------------------------
    # 7) Inference/Scoring
    # ----------------------------
    print("inference/scoring")
    preds = model.predict(X_test_array)
    accuracy = accuracy_score(y_test_array, preds)

    # ----------------------------
    # 8) Collect Metrics
    # ----------------------------
    print("collect metrics")
    end_time = time.time()
    end_memory = process.memory_info().rss / (1024 ** 2)

    runtime = end_time - start_time
    memory_usage = end_memory - start_memory
    load_time = load_end - load_start
    ml_time = ml_end - ml_start
    data_size_mb = os.path.getsize(csv_file) / (1024 ** 2)
    data_columns = list(df.columns)

    # Speedup & Efficiency
    speedup = baseline_runtime / runtime if baseline_runtime else None
    efficiency = (speedup / num_workers) if speedup else None

    # ----------------------------
    # 9) Prepare Results
    # ----------------------------
    results = {
        "framework": "Dask-ML",
        "csv_file": csv_file,
        "num_workers": num_workers,
        "runtime_seconds": runtime,
        "peak_memory_MB": memory_usage,
        "data_size_MB": data_size_mb,
        "data_columns": data_columns,
        "accuracy": accuracy,
        "ml_time_seconds": ml_time,
        "data_load_time_seconds": load_time,
        "speedup": speedup,
        "efficiency": efficiency
    }
    print(f"ML Results for {num_workers} workers on {csv_file}: {results}")

    # ----------------------------
    # 10) Write to JSON
    # ----------------------------
    with open(output_json, "a") as f:
        json.dump(results, f)
        f.write("\n")

    # ----------------------------
    # 11) Shutdown
    # ----------------------------
    client.shutdown()


if __name__ == "__main__":
    # ----------------------------
    # Example usage in a single script
    # ----------------------------

    # 1) List of CSV files to process
    csv_files = ["large_data1.csv", "half2_data1.csv"]

    # 2) Output JSON file
    output_json = "dask_ml_results.json"

    # 3) Process each CSV file in turn
    for csv_file in csv_files:
        print(f"\n=== Processing {csv_file} with Dask-ML ===")
        baseline_runtime = None  # Reset baseline for this file

        # Try different worker counts
        for workers in [2, 4, 8]:
            dask_ml(csv_file, workers, output_json, baseline_runtime)

            # After the first run for this CSV, update baseline runtime
            if baseline_runtime is None:
                with open(output_json, "r") as f:
                    first_result_line = f.readline().strip()
                    if first_result_line:
                        first_result = json.loads(first_result_line)
                        baseline_runtime = first_result["runtime_seconds"]

