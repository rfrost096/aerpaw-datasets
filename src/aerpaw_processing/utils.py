import pandas as pd
from typing import Any
from pathlib import Path
import os
from dotenv import load_dotenv, find_dotenv
from aerpaw_processing.resources.config import Config
import yaml
from importlib import resources
from functools import reduce

load_dotenv(find_dotenv("config.env"))


def load_config() -> Config:
    details = resources.files("aerpaw_processing.resources") / "config.yaml"

    with details.open("r") as f:
        return Config(**yaml.safe_load(f))


def find_file(dataset_num: int, data_filenames: list[str]) -> list[str] | None:
    dataset_path = str(os.getenv(f"DATASET_{dataset_num}_HOME"))
    path = Path(dataset_path)
    file_paths: list[str] = []

    for filename in data_filenames:
        matches: list[str] = []
        for file_path in path.rglob(filename):
            if file_path.is_file():
                matches.append(str(file_path.resolve()))

        if len(matches) > 1:
            print("Warning: ambiguous dataset paths")
            for match in matches:
                print(f"\t{match}")

        if len(matches) < 1:
            print(
                f"Error: Dataset file not found in directory {dataset_path}, filename {data_filenames}."
            )
            return

        file_paths.append(matches[0])

    return file_paths


def load_datasets(filepaths: list[str]):
    datasets: list[pd.DataFrame] = []
    for _, path in enumerate(filepaths):
        read_data = pd.read_csv(
            path,
            na_values=["Unavailable"],
            engine="pyarrow",
        )

        datasets.append(read_data)

    return datasets


def check_field_alias(field: str, keys: Any) -> str | None:
    if field.upper() in keys:
        return field.upper()
    elif field.lower() in keys:
        return field.lower()
    elif field.capitalize() in keys:
        return field.capitalize()
    return None


def combine_datasets(
    datasets: list[pd.DataFrame],
    graph_name: str | None = None,
    fields: list[str] = [],
    cap_mode: bool = False,
    alt_mode: bool = False,
):
    processed_dfs: list[pd.DataFrame] = []

    for df in datasets:
        if len(fields) == 0:
            fields = df.keys().tolist()
        else:
            for field in fields:
                if field not in df.keys():
                    alias = check_field_alias(field, df.keys())
                    if alias is None:
                        print(f"Error: Graph requires field '{field}'")
                        return
                    else:
                        df[field] = df[alias]

        plot_df = df.dropna(subset=fields).copy()

        if cap_mode and graph_name is not None:
            p_lower = plot_df[graph_name].quantile(0.10)
            p_upper = plot_df[graph_name].quantile(0.90)

            plot_df[graph_name] = plot_df[graph_name].clip(lower=p_lower, upper=p_upper)

        if alt_mode:
            alias = check_field_alias("altitude", plot_df.keys())
            if alias is None:
                print("Altitude data not included")
            else:
                plot_df["altitude"] = plot_df[alias]
                median_alt = plot_df["altitude"].median()
                mad = (plot_df["altitude"] - median_alt).abs().median()

                multiplier = 4.0

                lower_bound = median_alt - (multiplier * mad)
                upper_bound = median_alt + (multiplier * mad)

                plot_df = plot_df[
                    (plot_df["altitude"] >= lower_bound)
                    & (plot_df["altitude"] <= upper_bound)
                ]

        processed_dfs.append(plot_df)

    combined_df = pd.concat(processed_dfs, ignore_index=True)

    return combined_df


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


ignore_cols = {"ID", "Timestamp", "Longitude", "Latitude", "Altitude"}


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
