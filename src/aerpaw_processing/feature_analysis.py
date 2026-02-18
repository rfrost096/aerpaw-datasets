import pandas as pd
import argparse
from yellowbrick.target import FeatureCorrelation
from aerpaw_processing.utils import combine_datasets, find_file

DEFAULT_GRAPH = "pearson"


def graph_pearson(combined_data: pd.DataFrame):
    working_data = combined_data.copy()

    x = working_data.drop(columns=["rsrp"])
    x = x.select_dtypes(include=["number"])

    y = working_data["rsrp"]

    visualizer = FeatureCorrelation(labels=x.columns)

    visualizer.fit(x, y)

    visualizer.show()


def graph_mutual(combined_data: pd.DataFrame):
    working_data = combined_data.copy()

    x = working_data.drop(columns=["rsrp"])
    x = x.select_dtypes(include=["number"])

    y = working_data["rsrp"]

    visualizer = FeatureCorrelation(method="mutual_info-regression", labels=x.columns)

    visualizer.fit(x, y)

    visualizer.show()


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
    options = parser.parse_args()

    dataset_num: int = int(options.dataset)
    filenames: list[str] = str(options.filenames).split(",")
    cap_mode: bool = options.cap
    alt_mode: bool = options.alt
    graph_name: str = options.graph_name

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

    combined_data = combine_datasets(
        datasets, graph_name, cap_mode=cap_mode, alt_mode=alt_mode
    )

    if combined_data is None:
        return

    if graph_name == "pearson":
        graph_pearson(combined_data)
    elif graph_name == "mutual":
        graph_mutual(combined_data)
    else:
        print(f"graph_name '{graph_name}' not implemented")


if __name__ == "__main__":
    main()
