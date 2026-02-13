import pandas as pd
import plotly.express as px  # type: ignore
import plotly.graph_objects as go  # type: ignore
import argparse
import os
from dotenv import load_dotenv, find_dotenv
from aerpaw_processing.tower_locations import Tower, towers  # type: ignore
from pathlib import Path

DEFAULT_GRAPH = "rsrp,rsrq,pci"

GRAPH_OPTIONS = ["rsrp", "rsrq", "pci"]

load_dotenv(find_dotenv("config.env"))

def check_required_fields(df: pd.DataFrame, graph_name: str, fields: list[str]):
    valid: bool = True
    for field in fields:
        if field not in df.keys():
            print(f"Error: Graph '{graph_name}' requires field '{field}'")
            valid = False
    return valid


def plot_kpi(df: pd.DataFrame, graph_name: str):
    if not check_required_fields(df, graph_name, ["latitude", "longitude", "altitude", graph_name]):
        return

    plot_df = df.dropna(subset=["latitude", "longitude", "altitude", graph_name])  # type: ignore

    p10 = plot_df[graph_name].quantile(0.10)
    p90 = plot_df[graph_name].quantile(0.90)

    plot_df[graph_name] = plot_df[graph_name].clip(lower=p10, upper=p90)

    fig = px.scatter_3d(  # type: ignore
        plot_df,
        x="longitude",
        y="latitude",
        z="altitude",
        color=graph_name,
        color_continuous_scale="Viridis",
        title=f"3D Spatial Distribution of {graph_name.upper()}",
        hover_data=["technology", "bands", "pci"],
    )

    fig.update_traces(marker=dict(size=3))  # type: ignore

    fig.show()  # type: ignore


def plot_pci(df: pd.DataFrame, towers: list[Tower] | None = None):
    """
    Plots the Physical Cell Identity (PCI) as discrete clusters.
    Helps visualize cell dominance and handover zones.
    """
    if not check_required_fields(df, "pci", ["latitude", "longitude", "altitude", "pci"]):
        return

    plot_df = df.dropna(subset=["latitude", "longitude", "altitude", "pci"])  # type: ignore

    plot_df["pci_str"] = plot_df["pci"].astype(str)

    fig = px.scatter_3d(  # type: ignore
        plot_df,
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

    if len(plot_df["pci"].unique()) > 20:
        fig.update_layout(showlegend=False)  # type: ignore

    fig.show()  # type: ignore

def find_file(dataset_path: str, data_filename: str) -> str | None:
    path = Path(dataset_path)
    matches: list[str] = []
    
    for file_path in path.rglob(data_filename):
        if file_path.is_file():
            matches.append(str(file_path.resolve()))
    
    if len(matches) > 1:
        print("Warning: ambiguous dataset paths")
        for match in matches:
            print(f"\t{match}")
        return matches[0]

    if len(matches) < 1:
        print(f"Error: Dataset file not found in directory {dataset_path}.")
        return

    return matches[0]

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
        "--filename",
        type=str,
        required=True,
        help="Data file name."
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
    filename: str = str(options.filename)

    for graph_name in graph_list:
        if graph_name not in GRAPH_OPTIONS:
            print(
                f"Graph name provided is not an option, graph name: {graph_name}, graph options: {GRAPH_OPTIONS}"
            )
            exit()

    data_path = find_file(str(os.getenv(f"DATASET_{dataset_num}_HOME")), filename)

    if data_path is None:
        return

    df = pd.read_csv(  # type: ignore
        data_path,
        na_values=["Unavailable"],
        parse_dates=["companion_abs_time_readable"],
        engine="pyarrow",
    )

    for graph_name in graph_list:
        if graph_name in ["rsrp", "rsrq"]:
            plot_kpi(df, graph_name)
        if graph_name == "pci":
            plot_pci(df, towers)


if __name__ == "__main__":
    main()
