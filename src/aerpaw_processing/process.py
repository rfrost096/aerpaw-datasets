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

def combine_datasets(path_map: dict[str, pd.DataFrame], graph_name: str, fields: list[str], cap_mode: bool = False, alt_mode: bool = True):
    processed_dfs: list[pd.DataFrame] = []

    for _, df in path_map.items():
        for field in fields:
            if field not in df.keys():

                if field.upper() in df.keys():
                    df[field] = df[field.upper()]
                elif field.capitalize() in df.keys():
                    df[field] = df[field.capitalize()]
                else:
                    print(f"Error: Graph '{graph_name}' requires field '{field}'")
                    return

        
        plot_df = df.dropna(subset=fields).copy()  # type: ignore

        if cap_mode:
            p_lower = plot_df[graph_name].quantile(0.10)
            p_upper = plot_df[graph_name].quantile(0.90)

            plot_df[graph_name] = plot_df[graph_name].clip(lower=p_lower, upper=p_upper)
        
        if alt_mode:
            if "altitude" not in fields:
                print("Altitude data not included")
            else:
                median_alt = plot_df["altitude"].median()
                mad = (plot_df["altitude"] - median_alt).abs().median()

                multiplier = 4.0 

                lower_bound = median_alt - (multiplier * mad)
                upper_bound = median_alt + (multiplier * mad)

                plot_df = plot_df[(plot_df["altitude"] >= lower_bound) & (plot_df["altitude"] <= upper_bound)]
        
        processed_dfs.append(plot_df)

    combined_df = pd.concat(processed_dfs, ignore_index=True)

    return combined_df


def plot_kpi(path_map: dict[str, pd.DataFrame], graph_name: str, cap_mode: bool, alt_mode: bool):
    combined_df = combine_datasets(path_map, graph_name, ["latitude", "longitude", "altitude", graph_name], cap_mode, alt_mode)

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
        hover_data=["dataset_file"], 
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
        hover_data=["dataset_file"],
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
        "-c",
        "--cap",
        action='store_true',
        help="Cap the dataset to the 10th - 90th percentile."
    )
    parser.add_argument(
        "-a",
        "--alt",
        action='store_true',
        help="Only graph data points around the median altitude."
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
    cap_mode: bool = options.cap
    alt_mode: bool = options.alt

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

    for idx, path in enumerate(data_paths):
        path_map[path] = pd.read_csv(  # type: ignore
            path,
            na_values=["Unavailable"],
            engine="pyarrow",
        )

        path_map[path]["dataset_file"] = filenames[idx]

    for graph_name in graph_list:
        if kpi_mode:
            plot_kpi(path_map, graph_name, cap_mode, alt_mode)
        else:
            if graph_name == "pci":
                plot_pci(path_map, towers)


if __name__ == "__main__":
    main()
