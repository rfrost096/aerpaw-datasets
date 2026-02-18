import pandas as pd
from typing import Any
from pathlib import Path
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv("config.env"))


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
            print(f"Error: Dataset file not found in directory {dataset_path}.")
            return

        file_paths.append(matches[0])

    return file_paths


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
