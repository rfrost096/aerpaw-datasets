import pandas as pd
import numpy as np
import glob
import os

# Configuration
directory_path = '/home/ryan/VT/Spring26/GradProject/DatasetWorkspace/CleanedDatasets/data/sigcols_mad_RSRP/' 
csv_files = glob.glob(os.path.join(directory_path, "*.csv"))

results = []

for file in csv_files:
    try:
        df = pd.read_csv(file)
        
        # Check if required columns exist
        required_cols = ['Timestamp', 'x', 'y', 'z']
        if all(col in df.columns for col in required_cols):
            
            # 1. Convert Timestamp (handle numeric or date strings)
            t_col = pd.to_numeric(df['Timestamp'], errors='coerce')
            if t_col.isna().all():
                t_col = pd.to_datetime(df['Timestamp'], errors='coerce')
            
            # 2. Convert Coordinates to numeric
            x = pd.to_numeric(df['x'], errors='coerce')
            y = pd.to_numeric(df['y'], errors='coerce')
            z = pd.to_numeric(df['z'], errors='coerce')

            # 3. Calculate Timestep Difference
            delta_t = t_col.diff().dropna()
            avg_t = delta_t.mean()
            
            # Convert Timedelta to seconds if necessary
            if hasattr(avg_t, 'total_seconds'):
                avg_t = avg_t.total_seconds()

            # 4. Calculate Euclidean Distance between consecutive samples
            # dist = sqrt( (x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2 )
            dx = x.diff()
            dy = y.diff()
            dz = z.diff()
            
            distances = np.sqrt(dx**2 + dy**2 + dz**2).dropna()
            avg_dist = distances.mean()

            results.append({
                "File Name": os.path.basename(file),
                "Samples": len(df),
                "Avg Timestep": round(float(avg_t), 4) if pd.notnull(avg_t) else "N/A",
                "Avg Distance": round(float(avg_dist), 4) if pd.notnull(avg_dist) else "N/A"
            })
    except Exception as e:
        print(f"Error processing {os.path.basename(file)}: {e}")

# Display the results
if results:
    summary_df = pd.DataFrame(results)
    print("\n" + summary_df.to_string(index=False))
else:
    print("No valid data processed. Check if columns 'Timestamp', 'x', 'y', and 'z' exist.")
