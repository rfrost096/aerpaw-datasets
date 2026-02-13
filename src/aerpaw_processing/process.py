import pandas as pd
import plotly.express as px  # type: ignore
import plotly.graph_objects as go  # type: ignore
import argparse
import os
from dotenv import load_dotenv, find_dotenv
from aerpaw_processing.tower_locations import Tower, towers  # type: ignore
from pathlib import Path

DEFAULT_GRAPH = "rsrp,rsrq"

GRAPH_OPTIONS = ["pci"]

load_dotenv(find_dotenv("config.env"))

def combine_datasets(path_map: dict[str, pd.DataFrame], graph_name: str, fields: list[str]):
    processed_dfs: list[pd.DataFrame] = []

    for file_path, df in path_map.items():
        for field in fields:
            if field not in df.keys():
                print(f"Error: Graph '{graph_name}' requires field '{field}'")
                return

        
        plot_df = df.dropna(subset=fields).copy()  # type: ignore

        # p10 = plot_df[graph_name].quantile(0.10)
        # p90 = plot_df[graph_name].quantile(0.90)

        # plot_df[graph_name] = plot_df[graph_name].clip(lower=p10, upper=p90)

        plot_df["dataset_file"] = file_path.split("/")[-1]
        
        processed_dfs.append(plot_df)

    combined_df = pd.concat(processed_dfs, ignore_index=True)

    return combined_df


def plot_kpi(path_map: dict[str, pd.DataFrame], graph_name: str):
    combined_df = combine_datasets(path_map, graph_name, ["latitude", "longitude", "altitude", graph_name])

    if combined_df is None:
        return

    fig = px.scatter_3d(  # type: ignore
        combined_df,
        x="longitude",
        y="latitude",
        z="altitude",
        color=graph_name,
        color_continuous_scale="Viridis",
        title=f"3D Spatial Distribution of {graph_name.upper()}",
        hover_data=["technology", "bands", "pci", "dataset_file"], 
    )

    fig.update_traces(marker=dict(size=3))  # type: ignore

    fig.show()  # type: ignore


def plot_pci(path_map: dict[str, pd.DataFrame], towers: list[Tower] | None = None):
    combined_df = combine_datasets(path_map, "pci", ["latitude", "longitude", "altitude", "pci"])

    if combined_df is None:
        return
    
    combined_df["pci_str"] = combined_df["pci"].astype(str)

    fig = px.scatter_3d(  # type: ignore
        combined_df,
        x="longitude",
        y="latitude",
        z="altitude",
        color="pci_str",
        title="3D Spatial Distribution of Physical Cell IDs (PCI)",
        hover_data=["technology", "bands", "rsrp"],
    )

    if towers is not None:
        tower_names = [t.name for t in towers]
        tower_lons = [t.lon for t in towers]
        tower_lats = [t.lat for t in towers]
        tower_alts = [t.alt for t in towers]

        fig.add_trace(  # type: ignore
            go.Scatter3d(
                x=tower_lons,
                y=tower_lats,
                z=tower_alts,
                mode="markers+text",
                marker=dict(size=8, color="black", symbol="diamond"),
                text=tower_names,
                textposition="top center",
                name="Cell Towers",
                hoverinfo="text+x+y",
            )
        )

    fig.update_traces(marker=dict(size=3))  # type: ignore

    if len(combined_df["pci"].unique()) > 20:
        fig.update_layout(showlegend=False)  # type: ignore

    fig.show()  # type: ignore

def find_file(dataset_path: str, data_filenames: list[str]) -> list[str] | None:
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

def main():
    parser = argparse.ArgumentParser(
        description="Create interesting graph to explore dataset features."
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=int,
        required=True,
        help="Dataset number."
    )
    parser.add_argument(
        "-f",
        "--filenames",
        type=str,
        required=True,
        help="Data file names, comma separated."
    )
    parser.add_argument(
        "-k",
        "--kpi",
        action='store_true',
        help="Graph KPI mode, graph_name can be any column name in dataset."
    )
    parser.add_argument(
        "-g",
        "--graph-name",
        type=str,
        default=DEFAULT_GRAPH,
        help="Graph name(s). Format is 'name1,name2,...'. Default is "
        + DEFAULT_GRAPH
        + ".",
    )
    options = parser.parse_args()

    graph_list: list[str] = str(options.graph_name).split(",")
    dataset_num: int = int(options.dataset)
    filenames: list[str] = str(options.filenames).split(",")
    kpi_mode: bool = options.kpi

    for graph_name in graph_list:
        if not kpi_mode and graph_name not in GRAPH_OPTIONS:
            print(
                f"Graph name provided is not an option, graph name: {graph_name}, graph options: {GRAPH_OPTIONS}"
            )
            exit()

    data_paths = find_file(str(os.getenv(f"DATASET_{dataset_num}_HOME")), filenames)
    path_map: dict[str, pd.DataFrame] = {}

    if data_paths is None:
        return

    for path in data_paths:
        path_map[path] = pd.read_csv(  # type: ignore
            path,
            na_values=["Unavailable"],
            engine="pyarrow",
        )

    for graph_name in graph_list:
        if kpi_mode:
            plot_kpi(path_map, graph_name)
        else:
            if graph_name == "pci":
                plot_pci(path_map, towers)


if __name__ == "__main__":
    main()
