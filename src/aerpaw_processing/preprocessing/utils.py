import logging
import pandas as pd
import os
from functools import reduce
from pyproj import Proj
from aerpaw_processing.resources.tower_locations import towers
from aerpaw_processing.resources.config.config_init import CONFIG, load_env

load_env()

logger = logging.getLogger(__name__)


def load_datasets(dataset_num: int, filepaths: list[str]):

    data_list: list[pd.DataFrame] = []

    for path in filepaths:
        dataset_path = os.getenv(f"DATASET_{dataset_num}_HOME")

        if dataset_path is None:
            logger.error(
                f"Environment variable for dataset {dataset_num} path is not set (DATASET_{dataset_num}_PATH)."
            )
            raise EnvironmentError(
                f"Environment variable for dataset {dataset_num} path is not set."
            )

        abs_path = os.path.join(dataset_path, path)

        try:
            data = load_data(abs_path)
        except FileNotFoundError:
            logger.error(f"File not found: {abs_path}")
            raise
        except pd.errors.EmptyDataError:
            logger.error(f"No data: {abs_path} is empty.")
            raise
        except Exception as e:
            logger.error(f"Error loading {abs_path}: {e}")
            raise

        data_list.append(data)

    return data_list


def load_data(abs_path: str):
    data = pd.read_csv(
        abs_path,
        na_values=["Unavailable"],
        engine="pyarrow",
    )
    return data


def convert_columns(
    data_list: list[pd.DataFrame], merge_col: str | None
) -> list[pd.DataFrame]:
    ordered_cols = [col.name for category in CONFIG.categories for col in category.cols]
    result: list[pd.DataFrame] = []

    for data in data_list:
        for original_col in list(data.keys()):
            found = False
            for category in CONFIG.categories:
                for col in category.cols:
                    if original_col in col.alias_list:
                        if col.name in data.columns:
                            data.drop(columns=[original_col], inplace=True)
                        else:
                            data.rename(columns={original_col: col.name}, inplace=True)
                        found = True
                    elif original_col == col.name or original_col == merge_col:
                        found = True
            if not found:
                data.drop(columns=[original_col], inplace=True)

        sorted_cols = [col for col in ordered_cols if col in data.columns]
        if merge_col is not None and merge_col in data.columns:
            sorted_cols.append(merge_col)
        elif merge_col is not None:
            error = f"Merge column '{merge_col}' not found in data columns."
            logger.error(error)
            raise ValueError(error)
        result.append(data[sorted_cols])

    return result


def merge_datasets(dfs_list: list[pd.DataFrame], merge_col: str, how: str = "outer"):
    def clean_merge(left: pd.DataFrame, right: pd.DataFrame):
        overlapping_cols = set(left.columns).intersection(set(right.columns)) - {
            merge_col
        }
        right_clean = right.drop(columns=list(overlapping_cols))
        return pd.merge(left, right_clean, on=merge_col, how=how)  # type: ignore

    merged_df = reduce(clean_merge, dfs_list)

    return merged_df


def rename_tech_columns(
    data: pd.DataFrame, tech_name: str, merge_col: str | None, common_cols: set[str]
) -> pd.DataFrame:
    signal_quality_idx = [cat.category for cat in CONFIG.categories].index(
        "Signal Quality"
    )

    if signal_quality_idx == -1:
        error = "Signal Quality category not found in configuration."
        logger.error(error)
        raise ValueError(error)

    tech_dependent_cols = [
        col.name for col in CONFIG.categories[signal_quality_idx].cols
    ]

    location_idx = [cat.category for cat in CONFIG.categories].index("Location")

    if location_idx == -1:
        error = "Location category not found in configuration."
        logger.error(error)
        raise ValueError(error)

    tech_independent_cols = [col.name for col in CONFIG.categories[location_idx].cols]

    tech_independent_cols.append(get_timestamp_col())

    renamed_data = data.copy()
    for col in list(renamed_data.keys()):

        if (
            col in tech_dependent_cols
            or (col in common_cols and col not in tech_independent_cols)
        ) and col != merge_col:
            renamed_data.rename(columns={col: f"{col}_{tech_name}"}, inplace=True)
    return renamed_data


def format_timestamp(data: pd.DataFrame) -> pd.DataFrame:
    timestamp_col_name = get_timestamp_col()

    if timestamp_col_name not in data.columns:
        error = f"Timestamp column '{timestamp_col_name}' not found in data columns."
        logger.error(error)
        raise ValueError(error)

    try:
        raw_col = data[timestamp_col_name]

        numeric_vals = pd.to_numeric(raw_col, errors="coerce")
        is_numeric = numeric_vals.notna()

        parsed_dates = pd.Series(index=raw_col.index, dtype="datetime64[ns]")

        if is_numeric.any():
            parsed_dates.loc[is_numeric] = pd.to_datetime(
                numeric_vals[is_numeric], unit="ns"
            )

        if (~is_numeric).any():
            parsed_dates.loc[~is_numeric] = pd.to_datetime(raw_col[~is_numeric])

        data[timestamp_col_name] = parsed_dates

    except Exception as e:
        logger.error(f"Error formatting timestamp column '{timestamp_col_name}': {e}")
        raise

    return data


def filter_features(data: pd.DataFrame) -> pd.DataFrame:
    req_categories = {"Timestamp", "Location"}
    other_categories = {"Signal Quality"}

    req_features = {
        col.name
        for category in CONFIG.categories
        if category.category in req_categories
        for col in category.cols
    }
    other_features = {
        col.name
        for category in CONFIG.categories
        if category.category in other_categories
        for col in category.cols
    }

    req_cols = [
        data_col
        for data_col in data.columns
        if any(
            data_col == col or data_col.startswith(f"{col}_") for col in req_features
        )
    ]
    other_cols = [
        data_col
        for data_col in data.columns
        if any(
            data_col == col or data_col.startswith(f"{col}_") for col in other_features
        )
    ]

    filtered_data = data[req_cols + other_cols].copy()

    filtered_data.dropna(axis=1, how="all", inplace=True)

    req_cols = [col for col in req_cols if col in filtered_data.columns]
    other_cols = [col for col in other_cols if col in filtered_data.columns]

    if req_cols:
        filtered_data.dropna(subset=req_cols, how="any", inplace=True)

    if other_cols:
        filtered_data.dropna(subset=other_cols, how="all", inplace=True)

    return filtered_data


def convert_to_relative_time(data: pd.DataFrame) -> pd.DataFrame:
    timestamp_col_name = get_timestamp_col()

    if timestamp_col_name not in data.columns:
        error = f"Timestamp column '{timestamp_col_name}' not found in data columns."
        logger.error(error)
        raise ValueError(error)

    try:
        data_series: pd.Series[pd.Timestamp] = data[timestamp_col_name]

        time_diff = data_series - data_series.iloc[0]

        data[timestamp_col_name] = time_diff

    except Exception as e:
        logger.error(
            f"Error converting to relative time using column '{timestamp_col_name}': {e}"
        )
        raise

    return data


def project_coordinates(data: pd.DataFrame) -> pd.DataFrame:
    base_tower = towers[0]

    local_proj = Proj(
        proj="aeqd",
        lat_0=base_tower.lat,
        lon_0=base_tower.lon,
        datum="WGS84",
    )

    data["Longitude"], data["Latitude"] = local_proj(
        data["Longitude"].values, data["Latitude"].values
    )

    data.rename(
        columns={"Longitude": "x", "Latitude": "y", "Altitude": "z"}, inplace=True
    )

    return data


def get_median_abs_deviation(
    data: pd.DataFrame, multiplier: float = 4.0
) -> pd.DataFrame:
    altitude_col = get_altitude_col()

    if altitude_col not in data.columns:
        error = f"Altitude column '{altitude_col}' not found in data columns."
        logger.error(error)
        raise ValueError(error)

    try:
        series = data[altitude_col]
        median = series.median()
        mad = (series - median).abs().median()
        lower_bound = median - (multiplier * mad)
        upper_bound = median + (multiplier * mad)
        data = data[series.between(lower_bound, upper_bound)]
    except Exception as e:
        logger.error(f"Error calculating MAD for altitude: {e}")
        raise

    return data


def get_flight_id(dataset_num: int, flight_name: str) -> str:
    return f"Dataset_{dataset_num}_{flight_name.replace(' ', '_')}"


def remove_duplicate_timestamps(data: pd.DataFrame) -> pd.DataFrame:

    timestamp_col_name = get_timestamp_col()

    if timestamp_col_name not in data.columns:
        error = f"Timestamp column '{timestamp_col_name}' not found in data columns."
        logger.error(error)
        raise ValueError(error)

    try:
        data.drop_duplicates(subset=[timestamp_col_name], keep="first", inplace=True)
    except Exception as e:
        logger.error(
            f"Error removing duplicate timestamps using column '{timestamp_col_name}': {e}"
        )
        raise

    return data


def get_timestamp_col():
    timestamp_cat_idx = [cat.category for cat in CONFIG.categories].index("Timestamp")

    if timestamp_cat_idx == -1:
        error = "Timestamp column not found in Location category of configuration."
        logger.error(error)
        raise ValueError(error)

    timestamp_col = CONFIG.categories[timestamp_cat_idx].cols[0]

    return timestamp_col.name


def get_altitude_col():
    return "Altitude"
