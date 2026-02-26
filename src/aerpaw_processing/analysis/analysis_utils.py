import logging
import pandas as pd
from aerpaw_processing.preprocessing.preprocess_utils import (
    remove_duplicate_timestamps,
    get_timestamp_col,
)
from aerpaw_processing.resources.config.config_init import load_env

load_env()


logger = logging.getLogger(__name__)


def get_columns(
    flight_dict: dict[str, pd.DataFrame], unique: bool = False
) -> pd.Series[str]:
    if unique:
        columns: list[list[str]] = [
            list(flight_data.columns) for flight_data in flight_dict.values()
        ]
        common_elements = set(columns[0]).intersection(*columns[1:])

        unique_columns = [
            ", ".join([col for col in df_cols if col not in common_elements])
            for df_cols in columns
        ]

        return pd.Series(unique_columns, dtype="string")
    else:
        return pd.Series(
            [", ".join(flight_data.columns) for flight_data in flight_dict.values()],
            dtype="string",
        )


def get_num_rows(flight_dict: dict[str, pd.DataFrame]) -> pd.Series[int]:
    return pd.Series(
        [len(flight_data) for flight_data in flight_dict.values()],
        dtype="int",
    )


def get_timestamp_mean_std(flight_dict: dict[str, pd.DataFrame]) -> pd.Series[str]:
    results: list[str] = []
    for _, flight_data in flight_dict.items():
        data = flight_data.copy()
        data = remove_duplicate_timestamps(data)
        timestamp_diffs: pd.Series[pd.Timedelta] = data[get_timestamp_col()].diff()
        timestamp_mean = timestamp_diffs.mean().total_seconds()
        timestamp_std = timestamp_diffs.std().total_seconds()
        results.append(f"{timestamp_mean:.2f} ± {timestamp_std:.2f}")
    return pd.Series(results, dtype="string")


def get_distance_mean_std(flight_dict: dict[str, pd.DataFrame]) -> pd.Series[str]:
    results: list[str] = []
    for _, flight_data in flight_dict.items():
        distance_x = flight_data["x"].diff()
        distance_y = flight_data["y"].diff()
        distance_z = flight_data["z"].diff()

        distance = (distance_x**2 + distance_y**2 + distance_z**2) ** 0.5

        distance_mean = distance.mean()
        distance_std = distance.std()

        results.append(f"{distance_mean:.2f} ± {distance_std:.2f}")

    return pd.Series(results, dtype="string")
