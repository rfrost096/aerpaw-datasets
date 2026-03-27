import pandas as pd
import numpy as np
import os
from pathlib import Path
import logging
from typing import cast
import argparse
import re

from pyproj import Proj

from aerpaw_processing.resources.tower_locations import towers
from aerpaw_processing.resources.config.config_init import load_env, CONFIG
from aerpaw_processing.paper.preprocess_utils import (
    StepEnum,
    DatasetConfig,
    add_step_entry,
    get_step_entry,
    get_col_tech_name,
    get_tech_independent_cols,
    get_alias_map,
    get_col_categorical_map,
    get_env_var,
    RELATIVE_TIMESTAMP_COL,
    TIMESTAMP_PATTERN,
)
from aerpaw_processing.paper.preprocess_report import generate_report

load_env()

logger = logging.getLogger(__name__)

# CONSTANTS:

BASE_TOWER = towers[0]
TIMESTAMP_COL = "Timestamp"
TECH_LIST = ["LTE_4G", "NR_5G"]

# DEFAULT SETTINGS:
REMOVE_COL = True
SIGNAL_ONLY = True
MAD_FILTER_FLAG = True
LABEL_COL = "RSRP"
SAVE_DATA = True
SAVE_CONTEXT = False
GEN_REPORT = False


def read_data(context: pd.DataFrame):
    """Read all dataset csv files into pandas DataFrame

    Use CONFIG file to iterate through all datasets (18, 22, 24), all flights for each
    dataset (18, 24 have multiple, 22 has a single), all technologies (flights in
    dataset 18 have both 4G and 5G on board, flights in 22 and 24 have either 4G
    or 5G), and files for each technology (flights in dataset 18 has a seperate csv file
    for each key performance indicator like RSPR, RSRQ, SINR, the other datasets only have
    one file per flight).

    Each dataset file is added to a details table, with dataset id (18, 22, or 24), flight
    name, technology used in file, file path, and a "data" column with the actual file data.
    Each of these details tables are entries into the file DataFrame that is returned by the
    function.
    """
    data_list: list[pd.DataFrame] = []

    for dataset in CONFIG.datasets:
        for flight in dataset.flights:
            for tech in flight.tech_list:
                for file in tech.files:
                    dataset_details = {
                        "dataset_id": dataset.num,
                        "flight_name": flight.name,
                        "tech": tech.name,
                        "filepath": file,
                    }

                    df_details = pd.DataFrame([dataset_details])

                    dataset_home = get_env_var(f"DATASET_{dataset.num}_HOME")

                    dataset_path = Path(dataset_home) / file.lstrip("/")

                    df = pd.read_csv(dataset_path)

                    df_details["data"] = [df]

                    data_list.append(df_details)

    working = pd.concat(data_list, ignore_index=True)

    add_step_entry(StepEnum.READ_DATA, working, context)


def rename_columns(context: pd.DataFrame, step_list: list[StepEnum]):
    """Rename all columns for each dataset file to a standardized naming scheme.

    Use get_alias_map to get a map of (key) alias = (value) standard name. Iterate
    over all entries in input df. For each dataset in "data" column, rename all
    columns using alias map.
    """

    working: pd.DataFrame = get_step_entry(step_list[-1], context)

    alias_map = get_alias_map(signal_only=False)
    working["data"] = working["data"].apply(lambda df: df.rename(columns=alias_map))

    step_list.append(StepEnum.RENAME_COLUMNS)

    add_step_entry(step_list[-1], working, context)


def remove_columns(context: pd.DataFrame, step_list: list[StepEnum], signal_only: bool):
    """Remove extra columns that aren't defined in CONFIG"""

    working: pd.DataFrame = get_step_entry(step_list[-1], context)

    alias_map = get_alias_map(signal_only=signal_only)
    working["data"] = working["data"].apply(
        lambda df: df[[c for c in df.columns if c in alias_map.values()]]
    )

    step_list.append(StepEnum.REMOVE_COLUMNS)

    add_step_entry(step_list[-1], working, context)


def _combine_dfs(
    dfs: list[pd.DataFrame], combine_column: str, log_info: str
) -> pd.DataFrame:
    """Helper function to combine multiple DataFrames on a shared column.

    Logic shared between combine_tech_files and combine_flight_techs.
    """
    if not dfs:
        raise ValueError("No dataframes to combine")

    combined_df = dfs[0]

    for i in range(1, len(dfs)):
        next_df = dfs[i]

        # Check if the combine_column matches exactly between the dataframes.
        # Exact match means they have the same values in the same order.
        is_exact = combined_df[combine_column].equals(next_df[combine_column])

        if is_exact:
            # IF EXACT MATCH:
            # The dataframes are perfectly aligned on the combine_column.
            # Merging will result in a DataFrame with the same number of rows.
            # No new rows will be created, and no NaNs will be introduced due to alignment.
            # This is the ideal case for combining multiple KPI files from the same flight.
            pass
        else:
            # IF NOT EXACT MATCH:
            # The dataframes have different sets of values in the combine_column or different lengths.
            # By using an 'outer' join, we ensure that all unique values from both
            # dataframes are preserved. Missing values for columns from one dataframe
            # at a combine_column value only present in the other will be filled with NaN.
            # This ensures no data is lost even if sampling rates or start/end times differ slightly.
            logger.warning(f"Non-exact match for {combine_column} in {log_info}")

        # Drop columns from next_df that already exist in combined_df (except the merge key)
        # to avoid duplicate columns with suffixes like _x and _y
        cols_to_drop = [
            c
            for c in next_df.columns
            if c in combined_df.columns and c != combine_column
        ]
        combined_df = pd.merge(
            combined_df,
            next_df.drop(columns=cols_to_drop),
            on=combine_column,
            how="outer",
        )

    return combined_df


def combine_tech_files(
    context: pd.DataFrame,
    step_list: list[StepEnum],
    combine_column: str = TIMESTAMP_COL,
):
    """Combine tech files based on a shared column (default is Timestamp)

    Each dataset may have multiple flights, and each flight may have multiple technologies, and
    each technology may have multiple files. Dataset, flight, and technology, combine all files
    resutling in dataset with multiple flights, flights with multiple technologies, but only
    one DataFrame for each technology. The previous identifier 'filepath' should be removed from
    the new DataFrames as it was used to distinguish between the now combined files."""

    working: pd.DataFrame = get_step_entry(step_list[-1], context)

    combined_rows = []

    # Group by dataset_id, flight_name, and tech to find files that should be combined
    for keys, group in working.groupby(["dataset_id", "flight_name", "tech"]):
        dataset_id, flight_name, tech = cast(tuple, keys)
        dfs = group["data"].tolist()

        log_info = f"dataset {dataset_id}, flight {flight_name}, tech {tech}"
        combined_df = _combine_dfs(dfs, combine_column, log_info)

        combined_rows.append(
            {
                "dataset_id": dataset_id,
                "flight_name": flight_name,
                "tech": tech,
                "data": combined_df,
            }
        )

    step_list.append(StepEnum.COMBINE_TECH_FILES)

    working_combined = pd.DataFrame(combined_rows)
    add_step_entry(step_list[-1], working_combined, context)


def combine_flight_techs(
    context: pd.DataFrame,
    step_list: list[StepEnum],
    combine_column: str = TIMESTAMP_COL,
):
    """Combine all flight technologies into one flight data file.

    There is only practically 2 technologies, LTE_4G and NR_5G, but the implementation should
    work for any number of technologies. The same logic is used to combine the tech files
    before."""

    working: pd.DataFrame = get_step_entry(step_list[-1], context)
    tech_independent_cols = get_tech_independent_cols()

    combined_rows = []

    # Group by dataset_id and flight_name to find technology dataframes that should be combined
    for keys, group in working.groupby(["dataset_id", "flight_name"]):
        dataset_id, flight_name = cast(tuple, keys)

        renamed_dfs = []
        for _, row in group.iterrows():
            tech = cast(str, row["tech"])
            df: pd.DataFrame = cast(pd.DataFrame, row["data"].copy())

            # Rename tech-dependent columns by appending tech name
            new_cols = {}
            for col in df.columns:
                if col not in tech_independent_cols and col != combine_column:
                    new_cols[col] = get_col_tech_name(col, tech)

            df = df.rename(columns=new_cols)
            renamed_dfs.append(df)

        log_info = f"dataset {dataset_id}, flight {flight_name}"
        combined_df = _combine_dfs(renamed_dfs, combine_column, log_info)

        combined_rows.append(
            {"dataset_id": dataset_id, "flight_name": flight_name, "data": combined_df}
        )

    working_combined = pd.DataFrame(combined_rows)

    step_list.append(StepEnum.COMBINE_FLIGHT_TECHS)
    add_step_entry(step_list[-1], working_combined, context)


def interpolate_to_label(
    context: pd.DataFrame, step_list: list[StepEnum], label_col: str
):
    """Fill in NAN columns for samples that have a value for label_col

    After combining all tech files into one dataset per flight, the user may want to focus on
    a specific column. For the Dataset 18 flights specifically, some timestamps are off by just
    a fraction of a second (~0.006, sometimes more/less up to about 0.1). To fix this issue,
    I will just do a simple interpolation of the other values.

    1. Check for multiple label_col (ex. label_col = RSRP, check for RSRP, RSRP_LTE_4G, RSRP_NR_5G
                                     based on TECH_LIST)
    2. Ensure that every sample with any of the label_col has a value for ALL label_col
    3. Interpolate all non-categorical values and choose the closest categorical values based on
       TIMESTAMP_COL (use get_col_categorical_map to get a map from column name to categorical status"""

    working: pd.DataFrame = get_step_entry(step_list[-1], context)
    cat_map = get_col_categorical_map()

    # 1. Identify multiple label columns
    label_variants = [label_col] + [
        get_col_tech_name(label_col, tech) for tech in TECH_LIST
    ]

    new_rows = []
    for _, row in working.iterrows():
        df: pd.DataFrame = cast(pd.DataFrame, row["data"]).copy()

        actual_labels = [c for c in label_variants if c in df.columns]
        if not actual_labels:
            logger.warning(
                f"No label columns {label_variants} found for flight {row['flight_name']}"
            )
            new_rows.append(
                {
                    "dataset_id": row["dataset_id"],
                    "flight_name": row["flight_name"],
                    "data": df,
                }
            )
            continue

        # Ensure timestamp column is datetime for interpolation
        df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
        df = df.sort_values(TIMESTAMP_COL)

        # Set timestamp as index for interpolation
        df = df.set_index(TIMESTAMP_COL)

        # Identify rows that have at least one label value
        has_label = df[actual_labels].notna().any(axis=1)

        for col in df.columns:
            # Skip if all values are NaN
            if cast(bool, df[col].isna().all()):
                continue

            is_categorical = cat_map.get(col, False)
            is_numeric = pd.api.types.is_numeric_dtype(df[col])

            if is_categorical or not is_numeric:
                # Categorical or non-numeric: choose closest
                valid = df[col].dropna()
                if not valid.empty:
                    if df.index.is_unique:
                        # reindex with method='nearest' finds the closest timestamp's value
                        nearest_values = valid.reindex(df.index, method="nearest")
                        df[col] = df[col].fillna(nearest_values)
                    else:
                        # Fallback if timestamps are not unique
                        df[col] = df[col].ffill().bfill()
            else:
                # Continuous numeric: interpolate based on index (Timestamp)
                df[col] = df[col].interpolate(method="index", limit_direction="both")

        # Filter to only rows that have a value for at least one of the label_col
        # After interpolation, if a row had at least one label, it should now have ALL labels
        # (if they were available at other timestamps)
        df_filtered = df[has_label].reset_index()

        new_row = {
            "dataset_id": row["dataset_id"],
            "flight_name": row["flight_name"],
            "data": df_filtered,
        }
        new_rows.append(new_row)

    working_interpolated = pd.DataFrame(new_rows)

    step_list.append(StepEnum.INTERPOLATE_TO_LABEL)

    add_step_entry(step_list[-1], working_interpolated, context)


def add_relative_time_col(context: pd.DataFrame, step_list: list[StepEnum]):
    working = get_step_entry(step_list[-1], context)

    def apply_mad_filter(df):
        first_valid = df["data"][TIMESTAMP_COL].first_valid_index()

        if re.match(
            TIMESTAMP_PATTERN, str(df["data"][TIMESTAMP_COL].iloc[first_valid])
        ):
            df["data"][TIMESTAMP_COL] = pd.to_datetime(
                df["data"][TIMESTAMP_COL], format="%Y-%m-%d %H:%M:%S.%f"
            )
        else:
            df["data"][TIMESTAMP_COL] = pd.to_datetime(
                df["data"][TIMESTAMP_COL], unit="ms"
            )

        series: pd.Series = cast(pd.Series, df["data"][TIMESTAMP_COL])

        df[RELATIVE_TIMESTAMP_COL] = series - series.iloc[0]

        return df

    working["data"] = working["data"].apply(apply_mad_filter)

    step_list.append(StepEnum.ADD_RELATIVE_TIME)

    add_step_entry(step_list[-1], working, context)


def project_coordinates(context: pd.DataFrame, step_list: list[StepEnum]):
    """Project Longitude, Latitude to x, y coordinates. Add additional 'z' column that equals Altitude.

    Use the BASE_TOWER as the x=0, y=0 origin for flight coordinates. BASE_TOWER is the launch point
    for the UAV's. Altitude at BASE_TOWER is 0, which is reflected in csv files. The actual radio equipment
    for the tower is 10m up. This location (x=0m,y=0m,z=10m) will be used later to calculate distance from
    UAV to tower radio."""
    local_proj = Proj(
        proj="aeqd",
        lat_0=BASE_TOWER.lat,
        lon_0=BASE_TOWER.lon,
        datum="WGS84",
    )

    working: pd.DataFrame = get_step_entry(step_list[-1], context)

    new_rows = []
    for _, row in working.iterrows():
        df: pd.DataFrame = cast(pd.DataFrame, row["data"]).copy()

        x, y = local_proj(df["Longitude"].values, df["Latitude"].values)
        df["x"] = x
        df["y"] = y
        df["z"] = df["Altitude"]

        new_row = {
            "dataset_id": row["dataset_id"],
            "flight_name": row["flight_name"],
            "data": df,
        }
        new_rows.append(new_row)

    working_projected = pd.DataFrame(new_rows)
    step_list.append(StepEnum.PROJECT_COORDINATES)
    add_step_entry(step_list[-1], working_projected, context)


def mean_abs_deviation_filter(context: pd.DataFrame, step_list: list[StepEnum]):

    working: pd.DataFrame = get_step_entry(step_list[-1], context)

    def apply_mad_filter(df):
        median_alt = df["Altitude"].median()

        mad_alt = (df["Altitude"] - median_alt).abs().median()

        threshold_multiplier = 3

        return df[
            (df["Altitude"] - median_alt).abs() <= (threshold_multiplier * mad_alt)
        ]

    working["data"] = working["data"].apply(apply_mad_filter)

    step_list.append(StepEnum.MAD_FILTER)

    add_step_entry(step_list[-1], working, context)


def calculate_angular_bin(context: pd.DataFrame, step_list: list[StepEnum]):
    """Partition the 3D data points into a spherical grid centered at the Base Station.

    Section II.B.1 Angular Binning in AI-Enabled Wireless Propagation Modeling and Radio Environment Maps for 5G Aerial Wireless Networks

    Use elevation and azimuth bins of size 0.05 radians. The grid should contain
    (pi/0.05) * (2 pi)/(0.05) = 7895 bins. There should be a new column called 'bin' where
    each data point is assigned a bin identifier."""
    working: pd.DataFrame = get_step_entry(step_list[-1], context)

    new_rows = []

    BS_Z = 10.0
    BIN_SIZE = 0.05

    num_phi_bins = int(np.ceil(2 * np.pi / BIN_SIZE))

    for _, row in working.iterrows():
        df: pd.DataFrame = cast(pd.DataFrame, row["data"]).copy()

        if "x" in df.columns and "y" in df.columns and "z" in df.columns:
            x = np.array(df["x"])
            y = np.array(df["y"])
            z_rel = np.array(df["z"].values) - BS_Z

            d_h = np.sqrt(x**2 + y**2)

            theta = np.arctan2(d_h, z_rel)

            phi = np.arctan2(y, x)
            phi = np.mod(phi, 2 * np.pi)

            bin_theta = np.floor(theta / BIN_SIZE)
            bin_phi = np.floor(phi / BIN_SIZE)

            df["bin"] = bin_theta * num_phi_bins + bin_phi
            df["bin"] = df["bin"].astype(str)

        new_row = {
            "dataset_id": row["dataset_id"],
            "flight_name": row["flight_name"],
            "data": df,
        }
        new_rows.append(new_row)

    working_binned = pd.DataFrame(new_rows)
    add_step_entry(StepEnum.CALCULATE_BIN, working_binned, context)


def calculate_correlation(context: pd.DataFrame, label_col: str = LABEL_COL):
    """Correlation Computation within each angular group.

    Section II.B.1 Correlation Computation in AI-Enabled Wireless Propagation Modeling and Radio Environment Maps for 5G Aerial Wireless Networks

    For each bin, compute pairwise RSRP correlations.
    r[m, n] = (Omega_m - Omega_bar)(Omega_n - Omega_bar) / sigma^2
    Pairwise correlation values r[m, n] are then categorized based on the radial separation
    between the measurement pairs d3D[m, n] into radial separation bins of size 5 m.
    """
    working: pd.DataFrame = get_step_entry(StepEnum.CALCULATE_BIN, context)

    new_rows = []

    # 5m radial separation bins
    RADIAL_BIN_SIZE = 5.0

    for _, row in working.iterrows():
        df: pd.DataFrame = cast(pd.DataFrame, row["data"]).copy()

        # Figure out the RSRP label column (use the first available among possible variants)
        label_variants = [label_col] + [
            get_col_tech_name(label_col, tech) for tech in TECH_LIST
        ]
        actual_label = None
        for col in label_variants:
            if col in df.columns:
                actual_label = col
                break

        if actual_label is None or "bin" not in df.columns or "x" not in df.columns:
            logger.warning(
                f"Missing required columns for correlation computation in flight {row['flight_name']}"
            )
            new_rows.append(
                {
                    "dataset_id": row["dataset_id"],
                    "flight_name": row["flight_name"],
                    "data": pd.DataFrame(),  # Add empty dataframe for correlation results.
                }
            )
            continue

        # Drop NaN values in label
        valid_df = df.dropna(subset=[actual_label, "x", "y", "z", "bin"]).copy()

        if valid_df.empty:
            new_rows.append(
                {
                    "dataset_id": row["dataset_id"],
                    "flight_name": row["flight_name"],
                    "data": pd.DataFrame(),
                }
            )
            continue

        omega_bar = valid_df[actual_label].mean()
        sigma_sq = valid_df[actual_label].var(ddof=0)  # using population variance

        if sigma_sq == 0 or cast(bool, pd.isna(sigma_sq)):
            # Cannot compute correlation if variance is 0
            new_rows.append(
                {
                    "dataset_id": row["dataset_id"],
                    "flight_name": row["flight_name"],
                    "data": pd.DataFrame(),
                }
            )
            continue

        # Add 3D distance column
        valid_df["d3D"] = np.sqrt(
            valid_df["x"] ** 2 + valid_df["y"] ** 2 + (valid_df["z"] - 10.0) ** 2
        )

        # We need to compute pairwise correlations for each bin
        #
        # For a bin with N points, we compute the correlation between every
        # point and every other point in that SAME bin.
        #
        # Visual Example (Bin with 4 points: A, B, C, D):
        #
        # Pairwise combinations (N=4, so N*(N-1)/2 = 6 pairs):
        # A <-> B
        # A <-> C
        # A <-> D
        # B <-> C
        # B <-> D
        # C <-> D
        #
        # Because we pair every point with every other point, the total number
        # of generated correlation pairs will be much larger than the number
        # of original data points. A dataset of 1000 points might generate tens
        # of thousands of pairs depending on how dense the bins are.

        corr_data = []

        for bin_id, group in valid_df.groupby("bin"):
            n_samples = len(group)
            if n_samples < 2:
                continue

            rsrp_vals = np.array(group[actual_label])
            d3d_vals = np.array(group["d3D"].values)

            # Math: (Omega_m - Omega_bar)
            # Deviation of point m from the dataset's global mean.
            norm_rsrp = rsrp_vals - omega_bar

            # Math: (Omega_m - Omega_bar) * (Omega_n - Omega_bar) / sigma^2
            # The outer product multiplies every element in `norm_rsrp` with every
            # other element to efficiently create an N x N matrix of all pair products.
            # We then divide by the global variance (sigma_sq).
            # `r_mn` element (m,n) is the correlation between point m and point n.
            rsrp_prod = np.outer(norm_rsrp, norm_rsrp)
            r_mn = rsrp_prod / sigma_sq

            # Math: d3D[m, n] = | d3D_m - d3D_n |
            # Compute the absolute radial distance between point m and point n.
            # `np.subtract.outer` efficiently creates an N x N matrix of these differences.
            dist_mn = np.abs(np.subtract.outer(d3d_vals, d3d_vals))

            # We only want unique pairs, so we use upper triangle without diagonal
            # (since the matrix is symmetric and the diagonal is m=n, which is a point correlated with itself)
            # This extracts the values corresponding to pairs (e.g. A-B, A-C, B-C) without duplicates (B-A) or self-pairs (A-A).
            iu = np.triu_indices(n_samples, k=1)

            r_mn_flat = r_mn[iu]
            dist_mn_flat = dist_mn[iu]

            bin_ids_flat = np.full(len(r_mn_flat), bin_id)

            corr_df = pd.DataFrame(
                {"bin": bin_ids_flat, "r": r_mn_flat, "dist": dist_mn_flat}
            )

            corr_data.append(corr_df)

        if corr_data:
            flight_corr_df = pd.concat(corr_data, ignore_index=True)
            # Categorize into radial separation bins of size 5m
            flight_corr_df["radial_bin"] = (
                np.floor(flight_corr_df["dist"] / RADIAL_BIN_SIZE) * RADIAL_BIN_SIZE
            )

            # Now we aggregate over all angular bins to generate a composite RSRP correlation profile
            # as a function of radial distance
            aggregated_corr = cast(
                pd.Series, flight_corr_df.groupby("radial_bin")["r"].mean()
            ).reset_index()
            aggregated_corr["num_pairs"] = (
                flight_corr_df.groupby("radial_bin").size().values
            )

            bin_stats = (
                flight_corr_df.groupby("bin")
                .agg({"r": "mean", "dist": "max"})
                .reset_index()
            )

            new_rows.append(
                {
                    "dataset_id": row["dataset_id"],
                    "flight_name": row["flight_name"],
                    "data": aggregated_corr,
                    "bin_stats": bin_stats,
                }
            )
        else:
            new_rows.append(
                {
                    "dataset_id": row["dataset_id"],
                    "flight_name": row["flight_name"],
                    "data": pd.DataFrame(),
                    "bin_stats": pd.DataFrame(),
                }
            )

    working_corr = pd.DataFrame(new_rows)
    add_step_entry(StepEnum.CORRELATION_COMPUTATION, working_corr, context)


def calculate_fast_fading_correlation(
    context: pd.DataFrame, step_list: list[StepEnum], label_col: str = LABEL_COL
):
    """Fast fading factor correlation in spatio-temporal domain.

    Section II.B.2 in AI-Enabled Wireless Propagation Modeling and Radio Environment Maps for 5G Aerial Wireless Networks.
    """
    working: pd.DataFrame = get_step_entry(step_list[-1], context)

    new_rows = []

    # We bin the calculated correlation values based on the spatial separation
    # between the points being compared. The paper groups these into 3m bins.
    SPATIAL_BIN_SIZE = 3.0

    for _, row in working.iterrows():
        df: pd.DataFrame = cast(pd.DataFrame, row["data"]).copy()

        # Figure out the RSRP label column
        label_variants = [label_col] + [
            get_col_tech_name(label_col, tech) for tech in TECH_LIST
        ]
        actual_label = None
        for col in label_variants:
            if col in df.columns:
                actual_label = col
                break

        if actual_label is None or "x" not in df.columns:
            logger.warning(
                f"Missing required columns for fast fading correlation in flight {row['flight_name']}"
            )
            new_rows.append(
                {
                    "dataset_id": row["dataset_id"],
                    "flight_name": row["flight_name"],
                    "data": pd.DataFrame(),
                }
            )
            continue

        # Drop NaN values in label and coords
        valid_df = df.dropna(subset=[actual_label, "x", "y", "z"]).copy()

        if valid_df.empty:
            new_rows.append(
                {
                    "dataset_id": row["dataset_id"],
                    "flight_name": row["flight_name"],
                    "data": pd.DataFrame(),
                }
            )
            continue

        # Ensure sequential order by Timestamp if available so that the sliding
        # window logic truly captures temporally adjacent measurements.
        if TIMESTAMP_COL in valid_df.columns:
            valid_df[TIMESTAMP_COL] = pd.to_datetime(valid_df[TIMESTAMP_COL])
            valid_df = valid_df.sort_values(TIMESTAMP_COL)

        # --- Math and Reasoning for Fast Fading Extraction ---
        # The radio propagation model (Eq. 4 in the paper) decomposes path loss into:
        # 1. Deterministic Line-of-Sight (LoS) path loss.
        # 2. Stochastic fast-fading factor.
        #
        # The received signal strength (RSRP) can be modeled as:
        # RSRP = P_TX - PL_LoS + nu
        # Where:
        #   PL_LoS = 20 * log10(4 * pi * f_c / c) + 20 * log10(d3D) - Gain_BS - Gain_UAV
        #   nu = the residual fast fading factor (plus any regional shadowing).
        #
        # Because we want to isolate 'nu', we need to remove the distance-dependent
        # LoS path loss component from the measured RSRP. The term 20 * log10(d3D)
        # represents the inverse square law of signal attenuation over distance.
        # By removing this trend, the remaining signal variations ('nu') represent
        # the small-scale multipath fading and orientation-affected receiver gain changes.

        # First, calculate the 3D spatial separation between the UAV and the Base Station (BS).
        # The BS is assumed to be at the coordinate origin (0, 0) with a Z-height of 10.0m.
        d3D_bs = np.sqrt(
            valid_df["x"] ** 2 + valid_df["y"] ** 2 + (valid_df["z"] - 10.0) ** 2
        )

        # Calculate the fast fading factor 'nu'.
        # Since RSRP is in dBm, we isolate nu by adding back the 20*log10(d3D) path loss term:
        # nu = RSRP - (-20 * log10(d3D)) = RSRP + 20 * log10(d3D).
        # (We use np.clip to prevent log10(0) if distance is exactly 0).
        nu = valid_df[actual_label] + 20 * np.log10(np.clip(d3D_bs, 1e-6, None))

        # To compute pairwise correlation, we need the zero-mean (centered) fading factor.
        nu_mean = nu.mean()
        nu_centered = (nu - nu_mean).values

        # Compute the global variance (sigma_nu^2) of the fast fading factor across the
        # entire flight trajectory. This acts as the normalization term for the correlation.
        sigma_nu_sq = nu_centered.var(ddof=0)

        if sigma_nu_sq == 0 or pd.isna(sigma_nu_sq):
            new_rows.append(
                {
                    "dataset_id": row["dataset_id"],
                    "flight_name": row["flight_name"],
                    "data": pd.DataFrame(),
                }
            )
            continue

        x_vals = valid_df["x"].values
        y_vals = valid_df["y"].values
        z_vals = valid_df["z"].values

        n_samples = len(nu_centered)

        r_list = []
        dist_list = []

        # --- Spatio-Temporal Correlation Calculation ---
        # In an aerial environment, UAV mobility means temporal samples are directly
        # mapped to spatial positions. Due to the inherent temporal variability of
        # fast-fading factors, correlation is expected to be localized.
        # As described in the paper, we evaluate the correlation using a sliding
        # window of the 5 most recent samples to capture this localized effect.
        for m in range(1, 6):
            if n_samples <= m:
                continue

            # Align the current measurement (nu_i) with the m-th preceding measurement (nu_{i-m}).
            # By slicing from index `m` onwards, we get all valid "current" samples.
            nu_i = nu_centered[m:]
            # By slicing from the beginning up to `-m`, we get the corresponding "preceding" samples.
            nu_i_m = nu_centered[:-m]

            # The pairwise correlation coefficient is calculated as:
            # r[i, m] = (nu_i * nu_{i-m}) / sigma_nu^2
            # Since the means are already subtracted, their product represents covariance,
            # which is then normalized by the global variance to yield the correlation 'r'.
            r_vals = (nu_i * nu_i_m) / sigma_nu_sq

            # Compute the absolute 3D spatial separation distance between the
            # measurement positions corresponding to nu_i and nu_{i-m}.
            dx = np.array(x_vals[m:]) - np.array(x_vals[:-m])
            dy = np.array(y_vals[m:]) - np.array(y_vals[:-m])
            dz = np.array(z_vals[m:]) - np.array(z_vals[:-m])

            d_vals = np.sqrt(dx**2 + dy**2 + dz**2)

            # Accumulate the correlation and distance values for this window size 'm'.
            r_list.extend(r_vals)
            dist_list.extend(d_vals)

        if r_list:
            ff_df = pd.DataFrame({"r": r_list, "dist": dist_list})

            # Group the computed pairwise correlation values based on their spatial separation distance.
            # Bins are spaced by SPATIAL_BIN_SIZE (3 meters).
            ff_df["spatial_bin"] = (
                np.floor(ff_df["dist"] / SPATIAL_BIN_SIZE) * SPATIAL_BIN_SIZE
            )

            # Average the correlation values within each spatial bin to produce a
            # smooth spatial correlation profile of the fast-fading behavior.
            aggregated_ff = (
                ff_df.groupby("spatial_bin")
                .agg(r=("r", "mean"), num_pairs=("r", "count"))
                .reset_index()
            )

            new_rows.append(
                {
                    "dataset_id": row["dataset_id"],
                    "flight_name": row["flight_name"],
                    "data": aggregated_ff,
                }
            )
        else:
            new_rows.append(
                {
                    "dataset_id": row["dataset_id"],
                    "flight_name": row["flight_name"],
                    "data": pd.DataFrame(),
                }
            )

    working_ff = pd.DataFrame(new_rows)
    add_step_entry(StepEnum.FAST_FADING_CORRELATION, working_ff, context)


def save_datasets(context: pd.DataFrame, step_list: list[StepEnum]):
    cleaned_home = Path(get_env_var("DATASET_CLEAN_HOME"))
    working = get_step_entry(step_list[-1], context)

    for _, row in working.iterrows():
        cast(pd.DataFrame, row["data"]).to_csv(
            cleaned_home
            / f"Dataset_{row['dataset_id']}_{cast(str, row['flight_name']).replace(' ', '_')}.csv"
        )


def save_context(context: pd.DataFrame):
    context_home = get_env_var("CONTEXT_HOME")
    os.makedirs(context_home, exist_ok=True)
    context.to_json(context_home + "/context.json")


def process(config: DatasetConfig):
    context = pd.DataFrame(columns=["step", "step_data"])

    read_data(context)

    step_list: list[StepEnum] = [StepEnum.READ_DATA]

    rename_columns(context, step_list)

    if config.remove_cols:
        remove_columns(context, step_list, config.signal_only)

    combine_tech_files(context, step_list)

    combine_flight_techs(context, step_list)

    interpolate_to_label(context, step_list, config.label_col)

    project_coordinates(context, step_list)

    if config.mad_filter:
        mean_abs_deviation_filter(context, step_list)

    calculate_angular_bin(context, step_list)

    calculate_correlation(context, config.label_col)

    calculate_fast_fading_correlation(context, step_list, config.label_col)

    if config.save_data:
        save_datasets(context, step_list)

    if config.save_context_data:
        save_context(context)

    if config.gen_report:
        generate_report(context, config.mad_filter)


def process_datasets():
    parser = argparse.ArgumentParser(
        description="""Process AERPAW datasets for
report generation or dataloading scripts."""
    )
    parser.add_argument(
        "--no-delete-columns",
        dest="delete_columns",
        action="store_false",
        default=True,
        help="""Do not delete any columns. The script renames important columns
for easier cross-dataset processing (i.e. RSRP, SINR). Other columns may exist, but
they are removed by default. Set this flag to preserve all columns.""",
    )
    parser.add_argument(
        "--no-signal_only",
        dest="signal_only",
        action="store_false",
        default=True,
        help="""Do not remove all columns except for signal quality columns. The script
uses only keeps signal quality columns by default. If other columns that are classified
as important in config.yaml are desired, set this flag. This flag will be ignored if
--no-delete-columns is set.""",
    )
    parser.add_argument(
        "--no-mad-filter",
        dest="mad_filter",
        action="store_false",
        default=True,
        help="""Do not filter datapoints based on median absolute deviation. The script
will take the median altitude value and only include points that are within certain 
standard deviations from the median. This will filter datapoints to only include points
at the general flight altitude of the UAV (the initial ascention of the UAV will be
removed)""",
    )
    parser.add_argument(
        "--label-col",
        dest="label_col",
        type=str,
        default=LABEL_COL,
        help="""Set a specific label column for correlation analysis. Default is RSRP.
Other label columns (SINR, RSRQ) are untested.""",
    )
    parser.add_argument(
        "--no-save-data",
        dest="save_data",
        action="store_false",
        default=True,
        help="Do not save the processed datasets",
    )
    parser.add_argument(
        "--save-context",
        dest="save_context",
        action="store_true",
        default=False,
        help="Save the context object that includes each processing step.",
    )
    parser.add_argument(
        "--generate-report",
        dest="generate_report",
        action="store_true",
        default=False,
        help="Generate report of each processing step.",
    )

    args = parser.parse_args()

    config = DatasetConfig(
        remove_cols=args.remove_cols,
        signal_only=args.signal_only,
        mad_filter=args.mad_filter,
        label_col=args.label_col,
        save_data=args.save_data,
        save_context_data=args.save_context,
        gen_report=args.generate_report,
    )

    process(config)


if __name__ == "__main__":
    config = DatasetConfig(
        remove_cols=REMOVE_COL,
        signal_only=SIGNAL_ONLY,
        mad_filter=MAD_FILTER_FLAG,
        label_col=LABEL_COL,
        save_data=SAVE_DATA,
        save_context_data=SAVE_CONTEXT,
        gen_report=GEN_REPORT,
    )
    process(config)
