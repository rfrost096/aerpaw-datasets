import pandas as pd
import argparse
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from aerpaw_processing.utils import combine_datasets, find_file, load_config


DEFAULT_GRAPH = "pearson"
config = load_config()


def graph_feature(
    combined_data: pd.DataFrame, graph_name: str, save_path: str | None = None
):
    working_data = combined_data.copy()

    compare_column: str | None = None

    if "rsrp" in working_data.columns:
        compare_column = "rsrp"
    elif "RSRP" in working_data.columns:
        compare_column = "RSRP"
    else:
        for col in working_data.columns:
            if "rsrp" in col.lower():
                compare_column = col
                break

    if compare_column is None:
        print("Error: No RSRP column found for correlation analysis.")
        return

    x = working_data.drop(columns=[compare_column])
    x = x.select_dtypes(include=["number"])
    y = working_data[compare_column]

    scores_dict = {}

    if graph_name == "pearson":
        scores_dict = x.corrwith(y).dropna().to_dict()

    elif graph_name == "mutual":
        for col in x.columns:
            valid_mask = x[col].notna() & y.notna()
            x_valid = x.loc[valid_mask, col]
            y_valid = y.loc[valid_mask]

            if len(x_valid) > 1:
                score = mutual_info_regression(
                    x_valid.to_frame(), y_valid, random_state=42
                )[0]
                scores_dict[col] = score
    else:
        print(f"Error: graph_name '{graph_name}' not supported.")
        return None

    if not scores_dict:
        print("Error: No valid overlapping data points found for analysis.")
        return None

    col_sort_order: list[str] = ["RSRP"]
    for cat in config.categories:
        if cat.category != "Location":
            col_sort_order.extend([col.name for col in cat.cols])

    col_sort_order.reverse()

    scores_series = pd.Series(scores_dict)

    existing_cols: list[str] = []

    for c in col_sort_order:
        for col in scores_series.index:
            if c.lower() in col.lower() and col not in existing_cols:
                existing_cols.append(col)

    scores_series = scores_series.reindex(existing_cols)

    plt.figure(figsize=(8, max(4, len(scores_series) * 0.3)))
    scores_series.plot(kind="barh", color="steelblue", edgecolor="black")

    plt.title(f"Feature Correlation ({graph_name.capitalize()}) with {compare_column}")
    plt.xlabel("Score")
    plt.ylabel("Features")
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    return scores_series


def main():
    parser = argparse.ArgumentParser(
        description="Create interesting graph to explore dataset features."
    )
    parser.add_argument(
        "-d", "--dataset", type=int, required=True, help="Dataset number."
    )
    parser.add_argument(
        "-f",
        "--filenames",
        type=str,
        required=True,
        help="Data file names, comma separated.",
    )
    parser.add_argument(
        "-k",
        "--kpi",
        action="store_true",
        help="Graph KPI mode, graph_name can be any column name in dataset.",
    )
    parser.add_argument(
        "-c",
        "--cap",
        action="store_true",
        help="Cap the dataset to the 10th - 90th percentile.",
    )
    parser.add_argument(
        "-a",
        "--alt",
        action="store_true",
        help="Only graph data points around the median altitude.",
    )
    parser.add_argument(
        "-g",
        "--graph-name",
        type=str,
        default=DEFAULT_GRAPH,
        help="Graph name. Default is " + DEFAULT_GRAPH + ".",
    )
    parser.add_argument(
        "--fields",
        type=str,
        help="Fields to be analyzed. Format: name1,name2,...",
    )
    options = parser.parse_args()

    dataset_num: int = int(options.dataset)
    filenames: list[str] = str(options.filenames).split(",")
    cap_mode: bool = options.cap
    alt_mode: bool = options.alt
    graph_name: str = options.graph_name
    fields: list[str] = []
    if options.fields is not None:
        fields = str(options.fields).split(",")

    data_paths = find_file(dataset_num, filenames)
    if data_paths is None:
        return

    datasets: list[pd.DataFrame] = []
    for idx, path in enumerate(data_paths):
        read_data = pd.read_csv(
            path,
            na_values=["Unavailable"],
            engine="pyarrow",
        )
        read_data["dataset_file"] = filenames[idx]

        datasets.append(read_data)

    combined_data = combine_datasets(datasets, graph_name, fields, cap_mode, alt_mode)

    if combined_data is None:
        return

    combined_data = combined_data[fields]

    if graph_name in ["pearson", "mutual"]:
        graph_feature(combined_data, graph_name)
    else:
        print(f"graph_name '{graph_name}' not implemented")


if __name__ == "__main__":
    main()
