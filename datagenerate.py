import pandas as pd 
import numpy as np 
import psutil 
import os 

available_ram = psutil.virtual_memory().available / (1024 ** 3)  # GB 
print(f"Available RAM: {available_ram:.2f} GB") 
csv_filename = 'large_data.csv' 


with open(csv_filename, 'w') as file: 
    file.write("id,x,y,category\n") 
    total_rows_written = 0 
    total_memory_used_gb = 0  # Track memory usage in GB 
    while True: 
        data_sample = { 
            'id': np.arange(total_rows_written + 1, total_rows_written + 1001), 
            'x': np.random.random(1000), 
            'y': np.random.random(1000), 
            'category': np.random.choice(['A', 'B', 'C', 'D'], 1000) 
        } 
        sample_df = pd.DataFrame(data_sample) 
        sample_memory_per_row = sample_df.memory_usage(deep=True).sum() / len(sample_df)  
        max_memory_for_chunk = available_ram * 0.2  # 20% of available RAM 
        chunk_size = int((max_memory_for_chunk * (1024 ** 3)) / sample_memory_per_row) 
        data = { 
            'id': np.arange(total_rows_written + 1, total_rows_written + chunk_size + 1), 
            'x': np.random.random(chunk_size), 
            'y': np.random.random(chunk_size), 
            'category': np.random.choice(['A', 'B', 'C', 'D'], chunk_size) 
        } 
        chunk_df = pd.DataFrame(data) 
        chunk_memory_gb = chunk_df.memory_usage(deep=True).sum() / (1024 ** 3) 
        total_memory_used_gb += chunk_memory_gb 
        chunk_df.to_csv(file, header=False, index=False) 
        total_rows_written += chunk_size 

        print(f"Written chunk with {chunk_size} rows. Total rows: {total_rows_written}, Memory used: {total_memory_used_gb:.2f} GB") 

        
        if total_memory_used_gb > available_ram * 1.1: 
            print(f"Reached memory limit of {available_ram * 1.1:.2f} GB. Stopping.") 
            break 

file_size = os.path.getsize(csv_filename) / (1024 ** 3)  # GB 
print(f"CSV file size: {file_size:.2f} GB")
