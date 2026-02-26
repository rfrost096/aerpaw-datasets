import logging
import pandas as pd
import os
from functools import reduce
from pyproj import Proj
import re
from aerpaw_processing.resources.tower_locations import towers
from aerpaw_processing.resources.config.config_init import (
    CONFIG,
    load_env,
    TIMESTAMP_PATTERN,
)

load_env()

logger = logging.getLogger(__name__)

ALTITUDE_COL = "Altitude"
INDEX_COL = "Index"


def _get_category(name: str):
    """Return the Category object whose .category matches *name*, or raise."""
    for cat in CONFIG.categories:
        if cat.category == name:
            return cat
    raise ValueError(f"Category '{name}' not found in configuration.")


def load_datasets(dataset_num: int, filepaths: list[str]) -> list[pd.DataFrame]:
    data_list: list[pd.DataFrame] = []

    for path in filepaths:
        dataset_path = os.getenv(f"DATASET_{dataset_num}_HOME")

        if dataset_path is None:
            msg = f"Environment variable DATASET_{dataset_num}_HOME is not set."
            logger.error(msg)
            raise EnvironmentError(msg)

        abs_path = os.path.join(dataset_path, path)

        try:
            data_list.append(load_data(abs_path))
        except FileNotFoundError:
            logger.error(f"File not found: {abs_path}")
            raise
        except pd.errors.EmptyDataError:
            logger.error(f"No data: {abs_path} is empty.")
            raise
        except Exception as e:
            logger.error(f"Error loading {abs_path}: {e}")
            raise

    return data_list


def load_data(abs_path: str) -> pd.DataFrame:
    return pd.read_csv(abs_path, na_values=["Unavailable"], engine="pyarrow")


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
        if merge_col is not None:
            if merge_col in data.columns:
                sorted_cols.append(merge_col)
            else:
                msg = f"Merge column '{merge_col}' not found in data columns."
                logger.error(msg)
                raise ValueError(msg)
        result.append(data[sorted_cols])

    return result


def merge_datasets(dfs_list: list[pd.DataFrame], merge_col: str, how: str = "outer"):
    def clean_merge(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
        overlapping_cols = set(left.columns) & set(right.columns) - {merge_col}
        return pd.merge(
            left, right.drop(columns=list(overlapping_cols)), on=merge_col, how=how  # type: ignore
        )

    return reduce(clean_merge, dfs_list)


def rename_tech_columns(
    data: pd.DataFrame, tech_name: str, merge_col: str | None, common_cols: set[str]
) -> pd.DataFrame:
    tech_dependent_cols = [col.name for col in _get_category("Signal Quality").cols]
    tech_independent_cols = [col.name for col in _get_category(get_location_cat()).cols]
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
    timestamp_col = get_timestamp_col()

    if timestamp_col not in data.columns:
        msg = f"Timestamp column '{timestamp_col}' not found in data columns."
        logger.error(msg)
        raise ValueError(msg)

    try:
        first_valid = data[timestamp_col].first_valid_index()
        if re.match(TIMESTAMP_PATTERN, str(data[timestamp_col].iloc[first_valid])):  # type: ignore
            data[timestamp_col] = pd.to_datetime(
                data[timestamp_col], format="%Y-%m-%d %H:%M:%S.%f"
            )
        else:
            data[timestamp_col] = pd.to_datetime(data[timestamp_col], unit="ms")
    except Exception as e:
        logger.error(f"Error formatting timestamp column '{timestamp_col}': {e}")
        raise

    return data


def filter_features(data: pd.DataFrame) -> pd.DataFrame:
    req_categories = {get_timestamp_cat(), get_location_cat()}
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
        c
        for c in data.columns
        if any(c == f or c.startswith(f"{f}_") for f in req_features)
    ]
    other_cols = [
        c
        for c in data.columns
        if any(c == f or c.startswith(f"{f}_") for f in other_features)
    ]

    filtered = data[req_cols + other_cols].copy()
    filtered.dropna(axis=1, how="all", inplace=True)

    req_cols = [c for c in req_cols if c in filtered.columns]
    other_cols = [c for c in other_cols if c in filtered.columns]

    if req_cols:
        filtered.dropna(subset=req_cols, how="any", inplace=True)
    if other_cols:
        filtered.dropna(subset=other_cols, how="all", inplace=True)

    return filtered


def convert_to_relative_time(data: pd.DataFrame) -> pd.DataFrame:
    timestamp_col = get_timestamp_col()

    if timestamp_col not in data.columns:
        msg = f"Timestamp column '{timestamp_col}' not found in data columns."
        logger.error(msg)
        raise ValueError(msg)

    try:
        series: pd.Series = data[timestamp_col]
        data[timestamp_col] = series - series.iloc[0]
    except Exception as e:
        logger.error(
            f"Error converting to relative time using column '{timestamp_col}': {e}"
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
    data: pd.DataFrame, project_cords: bool, multiplier: float = 4.0
) -> pd.DataFrame:
    altitude_col = "z" if project_cords else ALTITUDE_COL

    if altitude_col not in data.columns:
        msg = f"Altitude column '{altitude_col}' not found in data columns."
        logger.error(msg)
        raise ValueError(msg)

    try:
        series = data[altitude_col]
        median = series.median()
        mad = (series - median).abs().median()
        data = data[
            series.between(median - multiplier * mad, median + multiplier * mad)
        ]
    except Exception as e:
        logger.error(f"Error calculating MAD for altitude: {e}")
        raise

    return data


def get_flight_id(dataset_num: int, flight_name: str) -> str:
    return f"Dataset_{dataset_num}_{flight_name.replace(' ', '_')}"


def get_dataset_and_flight_from_id(flight_id: str) -> tuple[int, str]:
    match = re.match(r"Dataset_(\d+)_(.+)", flight_id)
    if not match:
        msg = (
            f"Flight ID '{flight_id}' does not match expected format "
            "'Dataset_<num>_<name>'."
        )
        logger.error(msg)
        raise ValueError(msg)

    return int(match.group(1)), match.group(2).replace("_", " ")


def remove_duplicate_timestamps(data: pd.DataFrame) -> pd.DataFrame:
    timestamp_col = get_timestamp_col()

    if timestamp_col not in data.columns:
        msg = f"Timestamp column '{timestamp_col}' not found in data columns."
        logger.error(msg)
        raise ValueError(msg)

    try:
        data.drop_duplicates(subset=[timestamp_col], keep="first", inplace=True)
    except Exception as e:
        logger.error(
            f"Error removing duplicate timestamps using column '{timestamp_col}': {e}"
        )
        raise

    return data


def get_timestamp_col() -> str:
    return _get_category("Timestamp").cols[0].name


def get_timestamp_cat() -> str:
    return _get_category("Timestamp").category


def get_location_cat() -> str:
    return _get_category("Location").category


def get_index_col() -> str:
    return INDEX_COL


def get_label_col(data: pd.DataFrame, label_col: str) -> str:
    candidates = get_all_matching_cols(data, label_col)

    if not candidates:
        msg = (
            f"No label column found for '{label_col}' in data columns: "
            f"{list(data.columns)}."
        )
        logger.error(msg)
        raise ValueError(msg)

    if len(candidates) > 1:
        logger.info(
            f"Multiple label columns found for '{label_col}': {candidates}. "
            "Using the first one."
        )

    return candidates[0]


def get_all_matching_cols(data: pd.DataFrame, label_col: str) -> list[str]:
    return [
        col
        for col in data.columns
        if col.lower() == label_col.lower()
        or col.lower().startswith(label_col.lower() + "_")
    ]


def get_column_order(
    exclude_category: list[str] | None = None, project_cords: bool = False
) -> list[str]:
    _COORD_MAP = {"Longitude": "x", "Latitude": "y", "Altitude": "z"}
    col_order: list[str] = []

    for config_category in CONFIG.categories:
        if exclude_category and config_category.category in exclude_category:
            continue
        for col in config_category.cols:
            if col.name == INDEX_COL:
                continue
            name = _COORD_MAP.get(col.name, col.name) if project_cords else col.name
            col_order.append(name)

    return col_order


def get_all_flight_ids() -> list[str]:
    return [
        get_flight_id(dataset.num, flight.name)
        for dataset in CONFIG.datasets
        for flight in dataset.flights
    ]
