# aerpaw-datasets

A Python package for loading, preprocessing, analyzing, and visualizing UAV cellular measurement datasets collected on the [AERPAW](https://aerpaw.org/) testbed.

## Supported Datasets

| # | Name | Technology | Flights |
|---|------|------------|---------|
| 18 | [Ericsson 5G NSA RF and Throughput Measurements](https://aerpaw.org/dataset/aerpaw-ericsson-5g-uav-experiment/) | LTE 4G + NR 5G | Yaw 315, Yaw 45 |
| 22 | [Android-based 4G LTE Semi-Circular UAV Trajectory](https://aerpaw.org/dataset/dataset-22-android-based-4g-lte-measurements-for-semi-circular-uav-trajectory-around-a-private-aerpaw-base-station/) | LTE 4G | 4G LTE Flight |
| 24 | [Keysight Nemo and PawPrints Horizontal Sweep Measurements](https://aerpaw.org/dataset/dataset-24-android-based-4g-lte-measurements-for-semi-circular-uav-trajectory-around-a-private-aerpaw-base-station/) | LTE 4G, NR 5G | Multiple Nemo and PawPrints flights at 30m and 50m altitude |

All datasets were collected at the AERPAW Lake Wheeler site near Raleigh, NC. Flight paths, altitudes, and cellular technologies vary by dataset — see `config_file.yaml` for the full breakdown.

## Project Structure

```
aerpaw-datasets/
├── config.env                   # Dataset home paths (user-defined)
├── config stub.env              # Template for config.env
├── scripts/
│   ├── init.sh                  # Environment initialization
│   ├── generate_graphs.sh       # Example graph invocations
│   └── generate_kml_22.sh       # KML generation for dataset 22
├── presentations/               # LaTeX Beamer slide generation
└── src/aerpaw_processing/
    ├── preprocessing/           # Data loading, cleaning, and normalization
    ├── graph/                   # 3D visualization and feature correlation
    ├── analysis/                # Flight characteristic statistics
    ├── dataloader/              # PyTorch Dataset wrapper
    ├── KMLPostProcessing/       # KML file generation from datasets
    ├── dataset12/               # Dataset 12-specific 2D graph processing
    └── resources/
        ├── config/              # Configuration loading (config_file.yaml)
        ├── tower_locations.py   # AERPAW Lake Wheeler tower coordinates
        └── step_tracker.py      # Logging utility
```

## Setup

### 1. Install the package

```bash
pip install -e .
```

### 2. Configure dataset paths

Copy `config stub.env` to `config.env` and set the paths to where each dataset is stored on your machine:

```bash
DATASET_18_HOME=/path/to/dataset18
DATASET_22_HOME=/path/to/dataset22
DATASET_24_HOME=/path/to/dataset24
DATASET_CLEAN_HOME=/path/to/cleaned/output
```

## Configuration

Dataset structure, flight definitions, and column schema are defined in `src/aerpaw_processing/resources/config/config_file.yaml`.

**Datasets** are organized into **flights**, each of which can contain one or more technology streams (e.g., `LTE_4G`, `NR_5G`). Each technology lists the raw CSV files to merge.

**Categories** define a canonical column schema shared across all datasets. Raw column names from each dataset are mapped to these standard names during preprocessing:

| Category | Example Columns |
|---|---|
| Timestamp | `Timestamp` |
| Location | `Longitude`, `Latitude`, `Altitude` |
| Signal Quality | `RSRP`, `RSRQ`, `SINR`, `RSSI`, `TA`, CSI variants |
| Radio Performance | `Throughput`, `CQI`, `MCS`, `RI`, `iperf_Throughput` |
| Connection Status | `Is_Connected`, `ASU`, `Level` |
| Network Identity | `Cell_ID`, `MCC`, `MNC`, `PCI`, `EARFCN`, `Bands`, etc. |
| Base Station | `Base_Station_Distance`, `Base_Station_Bearing`, `Base_Station_Elevation` |

## Modules

### Preprocessing

Loads raw CSVs, standardizes column names, merges multi-file and multi-technology flights, and applies optional cleaning transformations.

**Entry point:** `preprocess-main` (or `python -m aerpaw_processing.preprocessing.preprocess_main`)

**Key options:**

| Flag | Description | Default |
|---|---|---|
| `--no-relative-time` | Keep absolute timestamps | Relative time enabled |
| `--no-project-coords` | Keep longitude/latitude | Projected x/y/z enabled |
| `--alt-median-abs-deviation` | Filter altitude outliers via MAD | Disabled |
| `--no-fill` | Skip forward/backward fill of NaN values | Fill enabled |
| `--no-save-data` | Do not write cleaned CSVs | Save enabled |

When saving is enabled, cleaned CSVs are written to `DATASET_CLEAN_HOME` named `Dataset_<num>_<Flight_Name>.csv`.

**Python API:**

```python
from aerpaw_processing.preprocessing.preprocess_main import process_datasets

data_dict = process_datasets(
    relative_time=True,
    project_coords=True,
    save_cleaned_data=True,
)
# data_dict[dataset_num][flight_name] → pd.DataFrame
```

### Graph

Provides 3D spatial visualization of UAV flight paths colored by a chosen KPI, and mutual information feature analysis. Depends on preprocessed data (run preprocessing first or let graph functions call it internally).

**Entry points:**

| Command | Description |
|---|---|
| `graph-mutual` | Mutual information scores for a single flight |
| `graph-avg-mutual` | Average mutual information across all flights |
| `graph-label` | 3D scatter plot of a KPI along the flight path |
| `graph-label-temporal` | Animated 3D scatter showing KPI evolution over time |

**Example usage:**

```bash
# Mutual correlation for one flight
graph-mutual --flight-id Dataset_18_Yaw_315_Flight --label RSRP

# Average mutual correlation across all flights
graph-avg-mutual

# 3D spatial distribution of PCI (categorical)
graph-label --flight-ids Dataset_22_4G_LTE_Flight --label PCI

# Temporal animation of RSRP
graph-label-temporal --flight-ids Dataset_24_Nemo_5G_30m_Flight_1 --label RSRP
```

Run any command with `--help` to see all options, including `--save-path` to write PNG/HTML output files.

See `scripts/generate_graphs.sh` for a full set of example invocations.

**Flight IDs** follow the format `Dataset_<num>_<Flight_Name_With_Underscores>`. Valid IDs can be retrieved programmatically:

```python
from aerpaw_processing.preprocessing.preprocess_utils import get_all_flight_ids
print(get_all_flight_ids())
```

### Analysis

Computes summary statistics for each flight, such as average sampling interval and average distance between consecutive measurements.

**Entry point:** `analyze`

```bash
analyze columns,num_rows,timestamp_mean_std_s,distance_mean_std --save
```

**Available characteristics:**

| Value | Description |
|---|---|
| `columns` | All column names per flight |
| `unique_columns` | Columns unique to each flight (not shared across all) |
| `num_rows` | Row count after preprocessing |
| `timestamp_mean_std_s` | Mean ± std of time between samples (seconds) |
| `distance_mean_std` | Mean ± std of distance between consecutive GPS points (meters) |

**Python API:**

```python
from aerpaw_processing.analysis.analysis_main import DatasetFlightDetails, FlightCharacteristic

details = DatasetFlightDetails()
df = details.get_characteristics([
    FlightCharacteristic.NUM_ROWS,
    FlightCharacteristic.TIMESTAMP_MEAN_STD,
    FlightCharacteristic.DISTANCE_MEAN_STD,
])
print(df)
```

### Dataloader

A PyTorch `Dataset` that loads a preprocessed (cleaned) CSV for a given flight and returns `(features, label)` tensor pairs. The label column is user-defined.

Requires that preprocessing has been run with `save_cleaned_data=True` beforehand.

```python
from aerpaw_processing.dataloader.dataloader import SignalDataset
from torch.utils.data import DataLoader

dataset = SignalDataset(
    dataset_num=22,
    flight_name="4G LTE Flight",
    label_col="RSRP",
)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### KML Post-Processing

Generates KML files from dataset measurements for visualization in tools such as Google Earth. See `src/aerpaw_processing/KMLPostProcessing/README.md` for details.

### Dataset 12

Contains dataset-specific processing code and scripts for generating 2D graphs, provided as-is.

## Tower Locations

Cell tower coordinates for the AERPAW Lake Wheeler site are defined in `src/aerpaw_processing/resources/tower_locations.py`. Five towers (LW1–LW5) are specified with latitude, longitude, and altitude. Altitudes are relative to the LW1 base station and include 10 m for the mounted radio equipment.

If a Digital Elevation Model (DEM) raster is available, set `CALCULATE_ALTITUDE_WITH_DEM = True` in `tower_locations.py` and configure `DEM_PATH` in `config.env` to recompute altitudes from terrain data.

## Presentations

The `presentations/` directory contains a LaTeX Beamer template and a `build_presentation.py` script that uses the analysis and graph outputs to generate a slide deck summarizing flight results.

## Dependencies

Key dependencies (see `pyproject.toml` for the full list):

- `pandas`, `pyarrow` — data loading and manipulation
- `plotly` — interactive 3D visualization
- `matplotlib` — static plot generation
- `scikit-learn` — mutual information regression
- `pyproj` — coordinate projection
- `torch` — PyTorch dataloader
- `python-dotenv` — environment variable loading

## Additional Sources:

Dataset 12 Post Processing: https://drive.google.com/drive/folders/1COJfijvlQ37xpi1NfxZkO8waGpadE2U2

https://sites.google.com/ncsu.edu/aerpaw-wiki/aerpaw-user-manual/4-sample-experiments-repository/4-1-radio-software/4-1-4-uhd-python-api-experiments/lte-iq-post-processing
BS tower setup (transmitter): tower height - 10 m, transmit power - 10 dBm, carrier frequency - 3,51 GHz, bandwidth - 1.4 MHz, antenna - dipole (RM-WB1)

https://arxiv.org/abs/2510.08752
Fig. 23c shows the deployment of these sensors at the
LWRFL, where a single N6841A unit is mounted on each
of the four towers labeled LW2 through LW5, approximately
10 meters above ground level.

