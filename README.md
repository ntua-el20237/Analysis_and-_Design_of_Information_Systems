# Comparison between Python scaling frameworks for big data analysis and ML: Ray vs Dask

Distributed computing frameworks have become essential for processing large-scale datasets that surpass a single machine’s memory capacity. This project compares two Python-based distributed data processing frameworks, **Ray** and **Dask**, within the contexts of **data engineering (ETL)** and **machine learning tasks**. 

## Methodology

We provide a methodology for generating a multi-gigabyte synthetic dataset—exceeding the available system memory—and examine the frameworks’ memory utilization, runtime, speedup, and efficiency. A simple machine learning benchmark using **logistic regression** showcases their respective scaling behaviors under various worker configurations.

## Key Findings

Our experimental findings indicate that both frameworks can manage out-of-memory workloads through object spilling and partitioned data. However, they demonstrate different performance trade-offs concerning:

- **Speed**
- **Memory footprint**
- **Load balancing**

## Conclusion

This study guides practitioners in selecting the most appropriate framework based on workload characteristics and resource limitations. The choice between Ray and Dask ultimately depends on factors such as the data size, task complexity, and available system resources.
