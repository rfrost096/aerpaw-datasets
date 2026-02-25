import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import argparse
from aerpaw_processing.tower_locations import Tower, towers
from aerpaw_processing.utils import combine_datasets, find_file
from aerpaw_processing.resources.config.config_init import load_env

load_env()

DEFAULT_GRAPH = "rsrp,rsrq"

GRAPH_OPTIONS = ["pci"]

DEFAULT_TIME_COLUMN = "companion_abs_time"
DEFAULT_NUM_BINS = 20

RELATIVE_TIME_PREFIX = "relative_"


def process_time_data(df: pd.DataFrame, time_col: str):
    """Process time column for dataset.
    The two supported formats are the "readable" format and the Unix timestamp.

    readable ex: 2023-10-26 12:26:22.770

    unix ex: 1707432321651.64

    :param path_map: Map of all datasets to be graphed
    :type path_map: dict[str, pd.DataFrame]
    :param time_col: Time column of datasets
    :type time_col: str
    """
    parsed_dates = pd.to_datetime(
        df[time_col],
        format="%Y-%m-%d %H:%M:%S.%f",  # This format is the "readable" format
        errors="coerce",
    )
    invalid_count = parsed_dates.isna().sum()

    if invalid_count == 0:
        df[time_col] = parsed_dates
    else:
        df[time_col] = pd.to_datetime(
            df[time_col], unit="ms", errors="coerce"  # Unix timestamp units
        )

    df = df.sort_values(time_col)

    df[RELATIVE_TIME_PREFIX + time_col] = df[time_col] - df[time_col][0]
    df[RELATIVE_TIME_PREFIX + time_col] = df[
        RELATIVE_TIME_PREFIX + time_col
    ].dt.total_seconds()  # type: ignore

    return df


def plot_kpi(
    datasets: list[pd.DataFrame], graph_name: str, cap_mode: bool, alt_mode: bool
):
    combined_df = combine_datasets(
        datasets,
        graph_name,
        ["latitude", "longitude", "altitude", graph_name],
        cap_mode,
        alt_mode,
    )

    if combined_df is None:
        return

    fig = px.scatter_3d(
        combined_df,
        x="longitude",
        y="latitude",
        z="altitude",
        color=graph_name,
        color_continuous_scale="Viridis",
        title=f"3D Spatial Distribution of {graph_name.upper()}",
        hover_data=["dataset_file"],
    )

    fig.update_traces(marker=dict(size=3))

    fig.show()


def plot_kpi_temporal(
    datasets: list[pd.DataFrame],
    graph_name: str,
    cap_mode: bool,
    alt_mode: bool,
    relative: bool,
    time_col: str,
    n_bins: int,
):
    """Plot a 3D spatial scatter of a KPI animated over time.

    The dataset's time column is divided into ``n_bins`` equal-width buckets.
    Each bucket becomes one animation frame so the user can step or play
    through the temporal evolution of the KPI across the flight path.

    Args:
        path_map:   Mapping of file paths to their loaded DataFrames.
        graph_name: Column name of the KPI to visualise.
        cap_mode:   Clip KPI values to the 10th–90th percentile.
        alt_mode:   Filter outlier altitudes via MAD before plotting.
        time_col:   Name of the timestamp column in the source data.
                    Case-insensitive fallback matching is attempted when the
                    exact name is not found.
        n_bins:     Number of time buckets to animate across (default 20).
    """
    required_fields = ["latitude", "longitude", "altitude", graph_name, time_col]
    combined_df = combine_datasets(
        datasets, graph_name, required_fields, cap_mode, alt_mode
    )

    if combined_df is None:
        return

    time_df = process_time_data(combined_df, time_col)

    if relative:
        time_col = RELATIVE_TIME_PREFIX + time_col

    x_range = [time_df["longitude"].min(), time_df["longitude"].max()]
    y_range = [time_df["latitude"].min(), time_df["latitude"].max()]

    c_min = float(time_df[graph_name].min())
    c_max = float(time_df[graph_name].max())

    if alt_mode:
        z_axis = time_col
        z_label = "Time (s)"
        hover_extras = ["dataset_file", "altitude"]

        z_range = [time_df[z_axis].min(), time_df[z_axis].max()]

        fig = px.scatter_3d(
            time_df,
            x="longitude",
            y="latitude",
            z=z_axis,
            color=graph_name,
            color_continuous_scale="Viridis",
            range_color=[c_min, c_max],
            range_x=x_range,
            range_y=y_range,
            range_z=z_range,
            title=f"Spatial-Temporal Distribution of {graph_name.upper()}",
            hover_data=hover_extras,
            labels={z_axis: z_label},
        )
    else:
        z_axis = "altitude"
        hover_extras = ["dataset_file", time_col]

        z_range = [time_df[z_axis].min(), time_df[z_axis].max()]

        if relative:
            time_delta = pd.to_timedelta(time_df[time_col], unit="s")
        else:
            time_delta = time_df[time_col]

        t_min, t_max = time_delta.min(), time_delta.max()

        bin_edges: pd.TimedeltaIndex | pd.DatetimeIndex
        bin_labels: list[str]

        if relative:
            bin_edges = pd.timedelta_range(t_min, t_max, periods=n_bins + 1)
            bin_labels = [
                f"{bin_edges[i].total_seconds():.2f}s - {bin_edges[i+1].total_seconds():.2f}s"
                for i in range(n_bins)
            ]

        else:
            bin_edges = pd.date_range(t_min, t_max, periods=n_bins + 1)  # type: ignore
            bin_labels = [
                f"{bin_edges[i].strftime('%H:%M:%S')}–{bin_edges[i+1].strftime('%H:%M:%S')}"
                for i in range(n_bins)
            ]

        time_df["time_bin"] = pd.cut(  # type: ignore
            time_delta, bins=bin_edges, labels=bin_labels, include_lowest=True  # type: ignore
        )

        time_df = time_df.dropna(subset=["time_bin"])
        time_df["time_bin"] = time_df["time_bin"].astype(str)
        frame_order = (
            time_df.drop_duplicates("time_bin")
            .sort_values(time_col)["time_bin"]
            .tolist()
        )
        time_df["time_bin"] = pd.Categorical(
            time_df["time_bin"], categories=frame_order, ordered=True
        )
        time_df = time_df.sort_values("time_bin")

        fig = px.scatter_3d(
            time_df,
            x="longitude",
            y="latitude",
            z=z_axis,
            color=graph_name,
            animation_frame="time_bin",
            color_continuous_scale="Viridis",
            range_color=[c_min, c_max],
            range_x=x_range,
            range_y=y_range,
            range_z=z_range,
            title=f"Spatial-Temporal Distribution of {graph_name.upper()} ({n_bins} time bins)",
            hover_data=hover_extras,
        )

        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 600
        fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 200

    fig.update_traces(marker=dict(size=3))

    fig.show()


def plot_pci(datasets: list[pd.DataFrame], towers: list[Tower] | None = None):
    combined_df = combine_datasets(
        datasets, "pci", ["latitude", "longitude", "altitude", "pci"]
    )

    if combined_df is None:
        return

    combined_df["pci_str"] = combined_df["pci"].astype(str)

    fig = px.scatter_3d(
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

        fig.add_trace(
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

    fig.update_traces(marker=dict(size=3))

    if len(combined_df["pci"].unique()) > 20:
        fig.update_layout(showlegend=False)

    fig.show()


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
        help="Graph name(s). Format is 'name1,name2,...'. Default is "
        + DEFAULT_GRAPH
        + ".",
    )
    parser.add_argument(
        "-t",
        "--temporal",
        action="store_true",
        help=(
            "Spatial-temporal KPI mode. Animates a 3D KPI scatter across "
            "time bins. Implies --kpi; use --graph-name to pick the KPI column."
        ),
    )
    parser.add_argument(
        "--time-col",
        type=str,
        default=DEFAULT_TIME_COLUMN,
        help=(
            "Name of the timestamp column used for temporal animation. "
            f"Default is {DEFAULT_TIME_COLUMN}."
        ),
    )
    parser.add_argument(
        "--time-bins",
        type=int,
        default=DEFAULT_NUM_BINS,
        help="Number of equal-width time buckets to animate across. Default is "
        + str(DEFAULT_NUM_BINS)
        + ".",
    )
    parser.add_argument(
        "--relative",
        action="store_true",
        help="Make time data relative to the first timestamp.",
    )
    options = parser.parse_args()

    graph_list: list[str] = str(options.graph_name).split(",")
    dataset_num: int = int(options.dataset)
    filenames: list[str] = str(options.filenames).split(",")
    temporal_mode: bool = options.temporal
    kpi_mode: bool = options.kpi or temporal_mode  # --temporal implies --kpi
    cap_mode: bool = options.cap
    alt_mode: bool = options.alt
    relative: bool = options.relative
    time_col: str = options.time_col
    n_bins: int = options.time_bins

    for graph_name in graph_list:
        if not kpi_mode and graph_name not in GRAPH_OPTIONS:
            print(
                f"Graph name provided is not an option, graph name: {graph_name}, graph options: {GRAPH_OPTIONS}"
            )
            exit()

    data_paths = find_file(dataset_num, filenames)

    if data_paths is None:
        return

    datasets: list[pd.DataFrame] = []

    for idx, path in enumerate(data_paths):
        data = pd.read_csv(
            path,
            na_values=["Unavailable"],
            engine="pyarrow",
        )

        data["dataset_file"] = filenames[idx]

        datasets.append(data)

    for graph_name in graph_list:
        if temporal_mode:
            plot_kpi_temporal(
                datasets, graph_name, cap_mode, alt_mode, relative, time_col, n_bins
            )
        elif kpi_mode:
            plot_kpi(datasets, graph_name, cap_mode, alt_mode)
        else:
            if graph_name == "pci":
                plot_pci(datasets, towers)


if __name__ == "__main__":
    main()
