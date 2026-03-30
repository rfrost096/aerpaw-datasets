# AERPAW Datasets Processing Pipeline

This repository contains a comprehensive data processing and machine learning pipeline for AERPAW (Aerial Experimentation and Research Platform for Advanced Wireless) datasets. The codebase specifically focuses on the methodology described in the paper *"AI-Enabled Wireless Propagation Modeling and Radio Environment Maps for 5G Aerial Wireless Networks"*.

## Pipeline Overview

The pipeline located in `src/aerpaw_processing/paper/` is divided into two main stages:
1. **Data Preprocessing:** Cleans, aligns, and transforms raw UAV flight data into structured spatial datasets.
2. **Model Training (InceptionTime):** Uses the processed data to train a 1-D sequence-to-sequence model for Radio Environment Map (REM) prediction.

---

## 1. Data Preprocessing (`preprocess_steps.py`)

The preprocessing script processes AERPAW datasets (e.g., datasets 18, 22, and 24) by executing a sequence of transformations.

### Key Steps
- **Data Loading & Combining:** Reads CSV files across different datasets, flights, and technologies (4G LTE, 5G NR), combining them based on shared timestamps.
- **Column Standardization:** Renames columns to standardized aliases and removes unnecessary metrics (configurable to keep only signal quality metrics like RSRP, SINR).
- **Interpolation:** Aligns mismatched timestamps and interpolates continuous variables or selects nearest categorical variables to ensure complete records.
- **Coordinate Projection:** Converts Longitude/Latitude into localized 3D Cartesian coordinates (x, y, z) relative to the Base Station. Calculates 3D distance (`d3D`), azimuth, and elevation angles.
- **Altitude Filtering:** Applies a Median Absolute Deviation (MAD) filter to remove non-representative altitude data (e.g., UAV take-off and landing phases).
- **Correlation Computation:** 
  - Partitions 3D data points into angular bins.
  - Computes pairwise RSRP spatial correlations and extracts fast-fading correlation factors in the spatio-temporal domain.

### Usage
You can run the preprocessing pipeline directly. It supports various command-line arguments to toggle specific steps:
```bash
python -m aerpaw_processing.paper.preprocess_steps [OPTIONS]

Options:
  --no-delete-columns   Preserve all columns instead of deleting non-essential ones.
  --no-signal_only      Keep columns beyond just signal quality metrics.
  --no-mad-filter       Disable median absolute deviation filtering for altitude.
  --label-col           Set specific label column (default: RSRP).
  --generate-report     Generate a report of each processing step.
```

---

## 2. Machine Learning Pipeline

The repository implements an `InceptionTime` neural network architecture adapted for 1-D sequence-to-sequence REM reconstruction.

### Dataloading (`inception_dataloader.py`)
The `InceptionDataset` class automatically handles loading the preprocessed CSV datasets. It aligns the physical features (Transmit Power, Center Frequency, 3D Distance, Azimuth, Elevation, and Cartesian coordinates) into a fixed radial sequence length (default $R_{max} = 500$ meters).

### Model Architecture (`inception_model.py`)
The model relies on stacked Inception modules to predict RSRP at every radial step.
- **Input:** `(Batch, 8, R_max)` — 8 physics-aligned features along the radial sequence.
- **Output:** `(Batch, R_max)` — Predicted RSRP (dBm) sequence.
- **Loss Strategy:** Uses a masked Smooth-L1 loss to evaluate the prediction exclusively at the ground-truth radial bin for each sample.

### Training
You can train the model from the CLI:
```bash
python -m aerpaw_processing.paper.inception_model [OPTIONS]

Options:
  --target              Target feature to predict (default: RSRP_NR_5G).
  --epochs              Number of training epochs (default: 50).
  --batch_size          Batch size for training (default: 64).
  --checkpoint_dir      Directory to save the best model weights.
```

Or you can programmatically invoke the training loop via `inception_run.py`:
```python
from aerpaw_processing.paper.inception_model import train
from aerpaw_processing.paper.preprocess_utils import DatasetConfig
from aerpaw_processing.paper.inception_dataloader import InceptionDataset

dataset = InceptionDataset(DatasetConfig())
model = train(dataset, epochs=50, checkpoint_dir="checkpoints")
```

## Environment Configuration
The pipeline relies on environment variables defined in `.env` (a template is provided in `config_stub.env`) for dataset directories and context saving paths:
- `DATASET_18_HOME`, `DATASET_22_HOME`, `DATASET_24_HOME`
- `DATASET_CLEAN_HOME`
- `CONTEXT_HOME`