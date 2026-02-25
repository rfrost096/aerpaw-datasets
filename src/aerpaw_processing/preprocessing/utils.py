import logging
import pandas as pd
import os
from functools import reduce
from aerpaw_processing.resources.config.config_init import CONFIG, load_env
from aerpaw_processing.resources.config.config_class import Config

load_env()

logger = logging.getLogger(__name__)

ignore_cols = {"ID", "Timestamp", "Longitude", "Latitude", "Altitude"}


def load_datasets(dataset_num: int, filepaths: list[str]):

    data_list: list[pd.DataFrame] = []

    for path in filepaths:
        dataset_path = os.getenv(f"DATASET_{dataset_num}_PATH")

        if dataset_path is None:
            logger.error(
                f"Environment variable for dataset {dataset_num} path is not set (DATASET_{dataset_num}_PATH)."
            )
            raise EnvironmentError(
                f"Environment variable for dataset {dataset_num} path is not set."
            )

        abs_path = dataset_path + path

        try:
            data = pd.read_csv(
                abs_path,
                na_values=["Unavailable"],
                engine="pyarrow",
            )
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


def convert_columns(
    data_list: list[pd.DataFrame], config: Config
) -> list[pd.DataFrame]:
    ordered_cols = [col.name for category in config.categories for col in category.cols]
    result: list[pd.DataFrame] = []

    for data in data_list:
        for original_col in list(data.keys()):
            found = False
            for category in config.categories:
                for col in category.cols:
                    if original_col in col.alias_list:
                        if col.name in data.columns:
                            data.drop(columns=[original_col], inplace=True)
                        else:
                            data.rename(columns={original_col: col.name}, inplace=True)
                        found = True
                    elif original_col == col.name or original_col in ignore_cols:
                        found = True
            if not found:
                data.drop(columns=[original_col], inplace=True)

        if "Timestamp" in data.columns:
            data["Timestamp"] = pd.to_datetime(
                data["Timestamp"], unit="ms", errors="coerce"
            )

        sorted_cols = [col for col in ordered_cols if col in data.columns]
        if "ID" in data.columns:
            sorted_cols.append("ID")
        result.append(data[sorted_cols])

    return result


def merge_datasets(dfs_list: list[pd.DataFrame], merge_col: str, how: str = "outer"):
    """
    Merges a list of pandas DataFrames based on a specific column.

    Parameters:
    - dfs_list (list): A list of pandas DataFrames to merge.
    - merge_col (str): The name of the column to merge on.
    - how (str): Type of merge to be performed ('inner', 'outer', 'left', 'right'). Default is 'outer'.

    Returns:
    - pd.DataFrame: A single merged DataFrame.
    """

    def clean_merge(left: pd.DataFrame, right: pd.DataFrame):
        overlapping_cols = set(left.columns).intersection(set(right.columns)) - {
            merge_col
        }
        right_clean = right.drop(columns=list(overlapping_cols))
        return pd.merge(left, right_clean, on=merge_col, how=how)  # type: ignore

    merged_df = reduce(clean_merge, dfs_list)

    return merged_df


def merge_tech_datasets(lte_4g: pd.DataFrame, nr_5g: pd.DataFrame):

    for col in list(lte_4g.keys()):
        if col not in ignore_cols:
            lte_4g.rename(columns={col: f"{col} (4G LTE)"}, inplace=True)
    for col in list(nr_5g.keys()):
        if col not in ignore_cols:
            nr_5g.rename(columns={col: f"{col} (5G NR)"}, inplace=True)

    data = merge_datasets([lte_4g, nr_5g], "ID")

    data.drop(columns=["ID"], inplace=True)

    return data
