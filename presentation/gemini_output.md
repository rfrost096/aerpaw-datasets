Here is a comprehensive presentation outline designed for a 20-minute Final Examination presentation for your M.Eng. in Computer Engineering. 

The pacing assumes roughly 1 minute per slide, leaving 2-3 minutes at the end for Q&A.

# Presentation Title: AI-Enabled 3D Radio Environment Map Prediction for Aerial Wireless Networks
**Target Duration:** 20 Minutes
**Context:** Final Examination, M.Eng. in CPE at Virginia Tech

---

### 1. Title Slide (1 minute)
* **Title:** AI-Enabled 3D Radio Environment Map Prediction for Aerial Wireless Networks
* **Presenter:** Ryan Frost, M.Eng. Candidate, Virginia Tech
* **Acknowledgements:** Shadab Mahboob, Dr. Haihan Nan, Dr. Liu 

### 2. Introduction & Motivation (2 minutes)
* **The Context:** The integration of UAVs (Unmanned Aerial Vehicles) into 5G and Next-Gen wireless networks.
* **The Problem:** Accurately predicting signal quality in 3D aerial spaces is highly complex due to multi-path fading, varying altitudes, and dynamic environments.
* **The Goal:** To use limited real-world spectrum samples gathered by UAVs to interpolate and construct accurate 3D Radio Environment Maps (REMs).
* **My Contribution:** Building a robust, automated end-to-end software pipeline—from raw AERPAW datasets to an `InceptionTime` machine learning model—to recreate and expand upon recent state-of-the-art research.

### 3. Background: AERPAW & Dataset Exploration (3 minutes)
* **What is AERPAW?** Aerial Experimentation and Research Platform for Advanced Wireless (brief overview of the testbed at Lake Wheeler).
* **Dataset Requirements:** Needed data with 3D trajectories (varying longitudes, latitudes, and altitudes) paired with Signal Quality metrics (RSRP, SINR, etc.).
* **Selection Process:** Reviewed multiple datasets, eventually identifying Datasets 18, 22, and 24 (covering both 4G LTE and 5G NR) as viable targets.
* **Initial Findings:** Showcase early exploratory data analysis (e.g., assessing uniform sampling in the time dimension, identifying time steps, and initial 3D mapping).

### 4. The Data Preprocessing Pipeline (4-5 minutes)
*This is the core engineering work of the first half of the project.*
* **Challenge:** Raw dataset files were unstructured, misaligned in time, and varied heavily between flights and technologies. 
* **Data Alignment & Interpolation:** How I standardized columns and interpolated mismatched timestamps to ensure complete physical and signal records.
* **Spatial Projection:** Converting raw GPS data (Lat/Lon/Alt) into a localized 3D Cartesian coordinate system (x, y, z) relative to the Lake Wheeler Base Stations. Calculated 3D distance, azimuth, and elevation angles.
* **MAD Filtering:** Applying a Median Absolute Deviation (MAD) filter to trim out noise (like the non-representative drone take-off and landing phases).

### 5. Spatial Correlation Analysis (3 minutes)
*Highlighting the transition from preprocessing to feature extraction.*
* **Angular Binning:** Partitioning the 3D data points into a spherical grid (0.05 rad bins) centered at the base station.
* **Correlation Profiles:** Showcasing how I generated pairwise spatial correlations of RSRP aggregated by 5m radial separation bins.
* **Fast-Fading Factor:** Explaining the extraction of the fast-fading correlation factor in the spatio-temporal domain by removing the Line-of-Sight (LoS) path loss. 
* *(Visuals: Include 1-2 key graphs from your `report.md` here, such as the 3D Correlation Bins or Fast Fading Profiles).*

### 6. Machine Learning Architecture: InceptionTime (4 minutes)
*The culmination of the project.*
* **Problem Formulation:** Treating the radial spatial path not as random points, but as a 1-D sequence for sequence-to-sequence prediction.
* **Data Loading Strategy:** Explaining the PyTorch Dataloader (`InceptionDataset`) that aligns physical features (Tx Power, Frequency, 3D Dist, Azimuth, Elevation, x, y, z) into a fixed radial sequence length ($R_{max} = 500$ meters).
* **The Model:** Adapting the `InceptionTime` neural network architecture (stacked Inception modules) for spatial sequences. 
* **Loss Function:** Using a masked Smooth-L1 loss so the model is exclusively evaluated at the ground-truth radial bin for each specific sample.

### 7. Results & Discussion (2 minutes)
* **Pipeline Success:** Successfully processed 29 raw dataset files into 11 clean, combined, and model-ready flight datasets.
* **Replication of Research:** Successfully recreated the spatial-temporal correlation figures (Figures 3 & 4) from the foundational paper.
* **Model Training:** Brief overview of how the model performed during training (Loss convergence, prediction accuracy of RSRP/SINR).

### 8. Conclusion & Future Work (1 minute)
* **Summary:** Successfully built a highly configurable, end-to-end data processing and modeling pipeline for AERPAW datasets.
* **Challenges Overcome:** Dealing with dataset access issues early on, complex data alignment, and computing heavy O(N^2) pairwise correlations.
* **Future Work:** Running real-time inference on the drone, integrating different environment topographies, or exploring alternative ML architectures.

### 9. Q&A (2-3 minutes)
* *Buffer time for committee questions.*

---

### Tips for your slides:
* **Visuals over Text:** Your `report.md` contains excellent interactive Plotly maps. Take screenshots of these 3D trajectories and correlation plots to use on your slides. 
* **Table formatting:** Use the dataset summary table from your notes (the one showing flight IDs, columns, timestamp mean/std, and distance mean/std) on the Preprocessing slide to prove the scale of the data you organized.
