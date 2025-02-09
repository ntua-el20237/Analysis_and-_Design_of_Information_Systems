# Comparison between Python Scaling Frameworks for Big Data Analysis and ML: Ray vs Dask

Distributed computing frameworks have become essential for processing large-scale datasets that surpass a single machine’s memory capacity. This project compares two Python-based distributed data processing frameworks, **Ray** and **Dask**, within the contexts of **data engineering (ETL)** and **machine learning tasks**.

## Methodology

We provide a methodology for generating a multi-gigabyte synthetic dataset—exceeding the available system memory—and examine the frameworks’ memory utilization, runtime, speedup, and efficiency. A simple machine learning benchmark using **logistic regression** showcases their respective scaling behaviors under various worker configurations.

### II.A Data Generation

To illustrate the problem of out-of-memory (OOM) datasets, we generated a synthetic CSV file named `large_data.csv`, designed to exceed the available RAM by approximately 10–20%. Following techniques similar to [1] and [2], our Python script uses:
- **NumPy** for generating random floating-point values (x and y) within a uniform distribution [0, 1].
- A **categorical column (category)** with four classes: A, B, C, D.
- An **incremental integer ID (id)**.

In early attempts, generating all rows in memory at once led to a Killed process error due to memory exhaustion. Consequently, we switched to a chunk-based generator that writes data in small chunks (on the order of millions of rows per chunk) to disk. This strategy successfully created multi-gigabyte CSV files (4–9 GB), confirming that it exceeded the available memory on a typical development laptop or workstation.

### II.B Splitting the Dataset

For comparative experiments, we performed an additional step of splitting `large_data.csv` into two halves:
1. `half1_data.csv`
2. `half2_data.csv`

Each half retains the same schema (`id`, `x`, `y`, `category`) and approximates half the row count. This division allows us to evaluate both the full dataset scale and a smaller portion (yet still sizable) of the data.

## Key Findings

Our experimental findings indicate that both frameworks can manage out-of-memory workloads through object spilling and partitioned data. However, they demonstrate different performance trade-offs concerning:

- **Speed**
- **Memory footprint**
- **Load balancing**

## Conclusion

This project guides in selecting the most appropriate framework based on workload characteristics and resource limitations. The choice between Ray and Dask ultimately depends on factors such as the data size, task complexity, and available system resources.

