# Filename: ray_ml.py

import ray
import ray.data
import time
import psutil
import os
import json
import sys
import numpy as np
import pandas as pd

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score


def ray_ml_partial_fit(
    csv_file,
    num_workers,
    output_json,
    baseline_runtime=None,
    batch_size=500_000
):
    """
    Incrementally train on all data by streaming it in chunks with Ray,
    using SGDClassifier(loss="log_loss") for partial_fit to mimic logistic regression.

    This function:
      1) Reads a CSV with Ray.
      2) Adds a random column to each batch (pandas format) -> "rand_col".
      3) Splits data into train/test by filtering on rand_col <0.8 vs >=0.8.
      4) Uses partial_fit on ds_train in chunks (batch_size).
      5) Evaluates on ds_test, logs accuracy & performance metrics.

    We're using older Ray-friendly approaches, i.e., no ds.split(..., skip(...)).
    """

    # ----------------------------
    # 1) Start/Connect to Ray
    # ----------------------------
    try:
        ray.init(address="auto")
        print("Connected to an existing Ray cluster.")
    except ConnectionError:
        # Otherwise start a local Ray cluster
        ray.init(num_cpus=num_workers)
        print(f"Started a new Ray runtime with {num_workers} CPUs for ML job.")

    # ----------------------------
    # 2) Performance Tracking
    # ----------------------------
    process = psutil.Process()
    start_time = time.time()
    start_memory = process.memory_info().rss / (1024**2)

    # ----------------------------
    # 3) Load CSV (no row limit)
    # ----------------------------
    load_start = time.time()
    ds = ray.data.read_csv(csv_file)
    load_end = time.time()

    data_size_mb = os.path.getsize(csv_file) / (1024**2)
    data_columns = ds.schema().names

 
    # ----------------------------
    # 5) Add a random float column for train/test split (~80/20)
    #    We'll use map_batches with batch_format="pandas"
    # ----------------------------
    def add_rand_col(df: pd.DataFrame) -> pd.DataFrame:
        # The incoming df has N rows
        N = len(df)
        df["rand_col"] = np.random.rand(N)  # Create float in [0,1)
        return df

    # Important: specify batch_format="pandas" so we always get a Pandas DataFrame
    ds = ds.map_batches(add_rand_col,compute ="actors", batch_format="pandas")

    # ----------------------------
    # 6) Filter rows to create ds_train (~80%) and ds_test (~20%)
    # ----------------------------
    ds_train = ds.filter(lambda row: row["rand_col"] < 0.8)
    ds_test  = ds.filter(lambda row: row["rand_col"] >= 0.8)

    # ----------------------------
    # 7) Partial-Fit Training
    # ----------------------------
    # We'll use SGDClassifier with "log_loss" (like logistic regression).
    sgd_clf = SGDClassifier(loss="log_loss")

    # 7a) Sample a small portion of training data to discover classes & columns
    sample_df = ds_train.limit(10_000).to_pandas()

    # If there's a 'category' column, convert it to a code
    if "category" in sample_df.columns:
        sample_df["category"] = sample_df["category"].astype("category")
        sample_df["category_code"] = sample_df["category"].cat.codes
        target_column = "category_code"
    else:
        # Adapt to your data. We assume "category" or "category_code" is the target.
        # If your target is named differently, adjust accordingly.
        target_column = "category"  # or whatever is your real target

    # Exclude target column + the random col from features
    feature_columns = [
        col for col in sample_df.columns
        if col not in ["category", "category_code", "rand_col"]
    ]

    # If target_column not in sample, bail out
    if target_column not in sample_df.columns:
        print(f"ERROR: target_column '{target_column}' not found in sample data.")
        ray.shutdown()
        return

    # Classes for partial_fit
    classes = sample_df[target_column].unique()

    # 7b) Initialize partial_fit with sample data
    X_sample = sample_df[feature_columns]
    y_sample = sample_df[target_column]
    sgd_clf.partial_fit(X_sample, y_sample, classes=classes)

    # 7c) Iterate over ds_train in chunks
    ml_start = time.time()
    batch_counter = 0
    for batch in ds_train.iter_batches(
        batch_size=batch_size,
        batch_format="pandas"
    ):
        # Convert 'category' -> 'category_code' if needed
        if "category" in batch.columns:
            batch["category"] = batch["category"].astype("category")
            batch["category_code"] = batch["category"].cat.codes

        # Remove the rand_col from features
        if "rand_col" in batch.columns:
            batch.drop(columns=["rand_col"], inplace=True, errors="ignore")

        # Prepare X, y
        if target_column not in batch.columns:
            continue  # can't train if missing

        X_chunk = batch[feature_columns]
        y_chunk = batch[target_column]

        if len(X_chunk) > 0:
            sgd_clf.partial_fit(X_chunk, y_chunk)

        batch_counter += 1
        if batch_counter % 10 == 0:
            print(f"Trained on {batch_counter} chunks so far. This chunk size={len(batch)} rows.")

    ml_end = time.time()

    # ----------------------------
    # 8) Evaluate on the test set
    # ----------------------------
    test_df = ds_test.to_pandas()
    if "category" in test_df.columns:
        test_df["category"] = test_df["category"].astype("category")
        test_df["category_code"] = test_df["category"].cat.codes

    # Remove rand_col from test if present
    if "rand_col" in test_df.columns:
        test_df.drop(columns=["rand_col"], inplace=True, errors="ignore")

    X_test = test_df[feature_columns]
    y_test = test_df[target_column]

    preds = sgd_clf.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    # ----------------------------
    # 9) Collect Metrics
    # ----------------------------
    end_time = time.time()
    end_memory = process.memory_info().rss / (1024**2)

    runtime = end_time - start_time
    memory_usage = end_memory - start_memory
    load_time = load_end - load_start
    ml_time = ml_end - ml_start

    speedup = baseline_runtime / runtime if baseline_runtime else None
    efficiency = (
        (baseline_runtime / runtime) / num_workers
        if (baseline_runtime and runtime != 0)
        else None
    )

    results = {
        "framework": "Ray + scikit-learn (SGD partial_fit)",
        "csv_file": csv_file,
        "num_workers": num_workers,
        "runtime_seconds": runtime,
        "peak_memory_MB": memory_usage,
        "data_size_MB": data_size_mb,
        "data_columns": list(data_columns),
        "accuracy": accuracy,
        "ml_time_seconds": ml_time,
        "data_load_time_seconds": load_time,
        "speedup": speedup,
        "efficiency": efficiency
    }

    print(f"\n=== RESULTS for {num_workers} workers on {csv_file} ===")
    print(results)

    # ----------------------------
    # 10) Write to JSON
    # ----------------------------
    with open(output_json, "a") as f:
        json.dump(results, f)
        f.write("\n")

    # ----------------------------
    # 11) Shutdown Ray
    # ----------------------------
    ray.shutdown()


if __name__ == "__main__":
    # List of CSV files
    csv_files = ["large_data1.csv", "half2_data1.csv"]

    # Output JSON file for results
    output_json = "ray_ml_partial_fit_results.json"

    for csv_file in csv_files:
        print(f"\n=== Processing {csv_file} in chunks ===")
        baseline_runtime = None  # reset for each CSV

        for workers in [4, 8, 16]:
            ray_ml_partial_fit(
                csv_file=csv_file,
                num_workers=workers,
                output_json=output_json,
                baseline_runtime=baseline_runtime,
                batch_size=100_000
            )

            # After the first run, set baseline_runtime
            if baseline_runtime is None:
                with open(output_json, "r") as f:
                    first_result_line = f.readline().strip()
                    first_result = json.loads(first_result_line) if first_result_line else None
                    if first_result:
                        baseline_runtime = first_result["runtime_seconds"]

