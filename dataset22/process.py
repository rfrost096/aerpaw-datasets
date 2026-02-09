import pandas as pd
from pathlib import Path
import plotly.express as px  # type: ignore
import argparse
import os
from dotenv import load_dotenv

DEFAULT_GRAPH = "rsrp"

script_dir = Path(__file__).resolve().parent

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


def plot_rsrp(df: pd.DataFrame):

    plot_df = df.dropna(subset=["latitude", "longitude", "altitude", "rsrp"])  # type: ignore

    fig = px.scatter_3d(  # type: ignore
        plot_df,
        x="longitude",
        y="latitude",
        z="altitude",
        color="rsrp",
        color_continuous_scale="Viridis",
        title="3D Spatial Distribution of RSRP Signal Strength",
        hover_data=["technology", "bands", "pci"],
    )

    fig.update_traces(marker=dict(size=3))  # type: ignore

    fig.show()  # type: ignore


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate KML file of the specified KPI and cell PCI"
    )
    parser.add_argument(
        "-g",
        "--graph-name",
        type=str,
        default=DEFAULT_GRAPH,
        help="Graph name. Default is " + DEFAULT_GRAPH,
    )
    options = parser.parse_args()

    load_dotenv(script_dir / "../config.env")

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

    if options.graph_name == "rsrp":
        plot_rsrp(df)
