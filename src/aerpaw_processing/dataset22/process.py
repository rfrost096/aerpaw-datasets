import pandas as pd
import plotly.express as px  # type: ignore
import plotly.graph_objects as go  # type: ignore
import argparse
import os
from dotenv import load_dotenv, find_dotenv
from aerpaw_processing.tower_locations import Tower, towers

DEFAULT_GRAPH = "rsrp,rsrq,pci"

GRAPH_OPTIONS = ["rsrp", "rsrq", "pci"]

load_dotenv(find_dotenv("config.env"))

dtype_map_dataset_22 = {
    "technology": "category",
    "dbm": "Int16",
    "rsrp": "Int16",
    "rsrq": "Int16",
    "rssi": "Int16",
    "asu": "Int16",
    "earfcn": "Int32",
    "pci": "Int16",
    "ta": "Int16",
    "ci": "Int32",
    "tac": "Int16",
    "bands": "category",
    "modes": "category",
    "mcc": "UInt16",
    "mnc": "UInt16",
    "is_connected": "bool",
    "phone_abs_time": "int64",
    "rel_time": "float64",
    "companion_abs_time": "float64",
    "longitude": "float64",
    "latitude": "float64",
    "altitude": "float64",
    "companion_abs_time_readable": "object",
}


def plot_kpi(df: pd.DataFrame, graph_name: str):

    plot_df = df.dropna(subset=["latitude", "longitude", "altitude", graph_name])  # type: ignore

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

        ground_altitude = plot_df["altitude"].min() if not plot_df.empty else 0
        tower_alts = [ground_altitude] * len(towers)

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


def main():
    parser = argparse.ArgumentParser(
        description="Create interesting graph to explore dataset features."
    )
    parser.add_argument(
        "-g",
        "--graph-name",
        type=str,
        default=DEFAULT_GRAPH,
        help="Graph name(s). Format is 'name1,name2,...' Default is "
        + DEFAULT_GRAPH
        + ".",
    )
    options = parser.parse_args()

    graph_list: list[str] = str(options.graph_name).split(",")

    for graph_name in graph_list:
        if graph_name not in GRAPH_OPTIONS:
            print(
                f"Graph name provided is not an option, graph name: {graph_name}, graph options: {GRAPH_OPTIONS}"
            )
            exit()

    data_path = (
        str(os.getenv("DATASET_22_HOME"))
        + "/"
        + str(os.getenv("DATASET_22_LOGS"))
        + "/"
        + str(os.getenv("DATASET_22_FILE"))
    )

    df = pd.read_csv(  # type: ignore
        data_path,
        dtype=dtype_map_dataset_22,  # type: ignore
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
