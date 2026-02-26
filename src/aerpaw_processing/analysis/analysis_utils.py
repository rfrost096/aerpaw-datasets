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
) -> pd.Series:
    if unique:
        all_cols = [list(df.columns) for df in flight_dict.values()]
        common = set(all_cols[0]).intersection(*all_cols[1:])
        return pd.Series(
            [", ".join(c for c in cols if c not in common) for cols in all_cols],
            dtype="string",
        )
    return pd.Series(
        [", ".join(df.columns) for df in flight_dict.values()],
        dtype="string",
    )


def get_num_rows(flight_dict: dict[str, pd.DataFrame]) -> pd.Series:
    return pd.Series([len(df) for df in flight_dict.values()], dtype="int")


def get_timestamp_mean_std(flight_dict: dict[str, pd.DataFrame]) -> pd.Series:
    results: list[str] = []
    timestamp_col = get_timestamp_col()
    for df in flight_dict.values():
        data = remove_duplicate_timestamps(df.copy())
        diffs = data[timestamp_col].diff()
        mean = diffs.mean().total_seconds()  # type: ignore
        std = diffs.std().total_seconds()  # type: ignore
        results.append(f"{mean:.2f} ± {std:.2f}")
    return pd.Series(results, dtype="string")


def get_distance_mean_std(flight_dict: dict[str, pd.DataFrame]) -> pd.Series:
    results: list[str] = []
    for df in flight_dict.values():
        distance = (
            df["x"].diff() ** 2 + df["y"].diff() ** 2 + df["z"].diff() ** 2
        ) ** 0.5
        results.append(f"{distance.mean():.2f} ± {distance.std():.2f}")
    return pd.Series(results, dtype="string")
