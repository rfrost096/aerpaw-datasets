from collections import defaultdict
import logging
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from aerpaw_processing.resources.tower_locations import towers
from aerpaw_processing.preprocessing.preprocess_utils import (
    get_index_col,
    get_column_order,
    get_location_cat,
    get_all_matching_cols,
    project_coordinates,
    get_timestamp_col,
    get_all_flight_ids,
    get_dataset_and_flight_from_id,
)
from aerpaw_processing.resources.config.config_init import CONFIG, load_env

load_env()

logger = logging.getLogger(__name__)


def _get_location_cols(project_cords: bool) -> list[str]:
    """Return the [x, y, z] column names for the current coordinate mode."""
    if project_cords:
        return ["x", "y", "z"]
    location_cat = next(
        cat for cat in CONFIG.categories if cat.category == get_location_cat()
    )
    return [col.name for col in location_cat.cols]


def _add_tower_trace(fig: go.Figure, project_cords: bool) -> go.Figure:
    """Overlay cell-tower markers on *fig* and return it."""
    tower_df = pd.DataFrame(
        {
            "Name": [t.name for t in towers],
            "Longitude": [t.lon for t in towers],
            "Latitude": [t.lat for t in towers],
            "Altitude": [t.alt for t in towers],
        }
    )
    if project_cords:
        tower_df = project_coordinates(tower_df)
        tower_df.rename(
            columns={"x": "Longitude", "y": "Latitude", "z": "Altitude"}, inplace=True
        )
    fig.add_trace(
        go.Scatter3d(
            x=tower_df["Longitude"],
            y=tower_df["Latitude"],
            z=tower_df["Altitude"],
            mode="markers+text",
            marker=dict(size=8, color="black", symbol="diamond"),
            text=tower_df["Name"],
            textposition="top center",
            name="Cell Towers",
            hoverinfo="text+x+y",
        )
    )
    return fig


def _order_scores(
    scores_dict: dict[str, float], project_cords: bool
) -> dict[str, float]:
    """Return *scores_dict* reordered to match the configured column order."""
    col_order = get_column_order(project_cords=project_cords)
    col_order.reverse()
    index_col = get_index_col()
    ordered_keys = [
        score_key
        for col in col_order
        for score_key in scores_dict
        if col in score_key and score_key != index_col
    ]
    return {k: scores_dict[k] for k in ordered_keys}


def _show_or_save(fig, save_path: str | None) -> None:  # type: ignore
    if save_path:
        fig.write_image(save_path)
    else:
        fig.show()


def _show_or_save_plt(save_path: str | None) -> None:
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def get_mutual_correlation_scores(df: pd.DataFrame, label_col: str) -> dict[str, float]:
    working_data = df.copy()
    x = working_data.drop(columns=[label_col]).select_dtypes(include=["number"])
    y = working_data[label_col]

    scores_dict: dict[str, float] = {}
    for col in x.columns:
        valid_mask = x[col].notna() & y.notna()
        x_valid = x.loc[valid_mask, col]
        y_valid = y.loc[valid_mask]
        if len(x_valid) > 1:
            scores_dict[col] = mutual_info_regression(
                x_valid.to_frame(), y_valid, random_state=42
            )[0]

    if not scores_dict:
        msg = (
            "Mutual information regression failed: no valid overlapping data "
            "points found for analysis."
        )
        logger.error(msg)
        raise ValueError(msg)

    return scores_dict


def _plot_mutual_correlation_bar(
    scores_series: pd.Series, title: str, save_path: str | None = None
) -> None:
    plt.figure(figsize=(8, max(4, len(scores_series) * 0.3)))
    scores_series.plot(kind="barh", color="steelblue", edgecolor="black")
    plt.title(title)
    plt.xlabel("Score")
    plt.ylabel("Features")
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    _show_or_save_plt(save_path)


def graph_mutual_correlation(
    df: pd.DataFrame,
    label_col: str,
    project_cords: bool = False,
    save_path: str | None = None,
) -> pd.Series:
    scores_dict = get_mutual_correlation_scores(df, label_col)
    scores_series = pd.Series(_order_scores(scores_dict, project_cords))
    _plot_mutual_correlation_bar(
        scores_series,
        title=f"Mutual Correlation with {label_col}",
        save_path=save_path,
    )
    return scores_series


def graph_average_mutual_correlation(
    dfs: list[pd.DataFrame],
    label: str,
    project_cords: bool = False,
    save_path: str | None = None,
) -> dict[str, pd.Series]:
    """
    Graph the average mutual information scores for each identified label column
    across all DataFrames in *dfs*.
    """
    label_cols: set[str] = set()
    for i, df in enumerate(dfs):
        try:
            label_cols.update(get_all_matching_cols(df, label))
        except Exception as e:
            logger.warning(f"Could not retrieve label col for DataFrame {i}: {e}")

    if not label_cols:
        msg = "No label columns found across any of the provided DataFrames."
        logger.error(msg)
        raise ValueError(msg)

    results_tracker: dict[str, pd.Series] = {}

    for target_label in label_cols:
        column_scores: dict[str, list[float]] = defaultdict(list)

        for i, df in enumerate(dfs):
            if target_label not in df.columns:
                continue
            try:
                for col, score in get_mutual_correlation_scores(
                    df, target_label
                ).items():
                    column_scores[col].append(score)
            except ValueError as e:
                logger.warning(
                    f"Skipping DataFrame {i} for label '{target_label}': {e}"
                )

        if not column_scores:
            logger.warning(
                f"Failed to calculate mutual correlation for label '{target_label}'. Skipping."
            )
            continue

        avg_scores = {col: sum(s) / len(s) for col, s in column_scores.items()}
        scores_series = pd.Series(_order_scores(avg_scores, project_cords))

        current_save_path: str | None = None
        if save_path:
            p = Path(save_path)
            current_save_path = str(p.with_stem(f"{p.stem}_{target_label}"))

        _plot_mutual_correlation_bar(
            scores_series,
            title=f"Average Mutual Correlation with {target_label}",
            save_path=current_save_path,
        )
        plt.close()

        results_tracker[target_label] = scores_series

    if not results_tracker:
        msg = (
            "Failed to calculate mutual correlation across any DataFrames "
            "for any label."
        )
        logger.error(msg)
        raise ValueError(msg)

    return results_tracker


def graph_label(
    df: pd.DataFrame,
    label_col: str,
    filter_features: bool = True,
    categorical: bool = False,
    project_cords: bool = False,
    graph_towers: bool = False,
    relative_time: bool = False,
    save_path: str | None = None,
) -> None:
    location_cols = _get_location_cols(project_cords)
    timestamp_col = get_timestamp_col()

    hover_dict: dict[str, bool | str] = {}
    if filter_features:
        hover_dict = {col: True for col in df.columns}
    else:
        signal_quality_cols = next(
            cat for cat in CONFIG.categories if cat.category == "Signal Quality"
        ).cols
        for sq_col in signal_quality_cols:
            for col in df.columns:
                if sq_col.name in col:
                    hover_dict[col] = True
        hover_dict[timestamp_col] = True

    if relative_time:
        df[timestamp_col] = df[timestamp_col].apply(format_timedelta)
    elif timestamp_col in hover_dict:
        hover_dict[timestamp_col] = "|%y-%m-%d %H:%M:%S.%L"

    if "Flight_ID" in df.columns:
        hover_dict["Flight_ID"] = True

    if categorical:
        df[label_col] = df[label_col].astype(str)

    plot_args = {  # type: ignore
        "data_frame": df,
        "x": location_cols[0],
        "y": location_cols[1],
        "z": location_cols[2],
        "color": label_col,
        "title": f"3D Spatial Distribution of {label_col}",
        "hover_data": hover_dict,
    }
    if not categorical:
        plot_args["color_continuous_scale"] = "Viridis"

    fig = px.scatter_3d(**plot_args)  # type: ignore

    if graph_towers:
        fig = _add_tower_trace(fig, project_cords)

    fig.update_traces(marker=dict(size=3))
    _show_or_save(fig, save_path)


def graph_label_temporal(
    df: pd.DataFrame,
    label_col: str,
    n_bins: int = 20,
    project_cords: bool = False,
    alt_median_abs_deviation: bool = False,
    relative_time: bool = False,
    graph_towers: bool = False,
    save_path: str | None = None,
) -> None:
    """Plot a 3D spatial scatter of a KPI animated over time.

    The dataset's time column is divided into ``n_bins`` equal-width buckets.
    Each bucket becomes one animation frame so the user can step or play
    through the temporal evolution of the KPI across the flight path.

    Args:
        df: Source DataFrame to visualise.
        label_col: Column name of the KPI to colour points by.
        n_bins: Number of time buckets to animate across (default 20).
        project_cords: Use projected x/y/z columns instead of the configured
            location category columns.
        alt_median_abs_deviation: When True, plots time on the z-axis instead
            of altitude (useful after altitude MAD filtering collapses z range).
        relative_time: Treat the timestamp column as a relative offset in
            seconds rather than an absolute datetime.
        graph_towers: Overlay cell tower markers on the plot.
        save_path: If provided, write the figure to this path instead of
            displaying it interactively.
    """
    location_cols = _get_location_cols(project_cords)
    x_col, y_col, z_col = location_cols

    timestamp_col = get_timestamp_col()
    x_range = [df[x_col].min(), df[x_col].max()]
    y_range = [df[y_col].min(), df[y_col].max()]
    c_min = float(df[label_col].min())
    c_max = float(df[label_col].max())

    hover_dict: dict[str, bool | str] = {col: True for col in df.columns}
    if alt_median_abs_deviation:
        hover_dict.pop(timestamp_col, None)
    elif relative_time:
        df[timestamp_col] = df[timestamp_col].apply(format_timedelta)
    else:
        hover_dict[timestamp_col] = "|%y-%m-%d %H:%M:%S.%L"  # type: ignore

    if "Flight_ID" in df.columns:
        hover_dict["Flight_ID"] = True

    common_scatter_kwargs = dict(
        x=x_col,
        y=y_col,
        color=label_col,
        color_continuous_scale="Viridis",
        range_color=[c_min, c_max],
        range_x=x_range,
        range_y=y_range,
        hover_data=hover_dict,
    )

    if alt_median_abs_deviation:
        df[timestamp_col] = df[timestamp_col].dt.total_seconds()
        z_range = [df[timestamp_col].min(), df[timestamp_col].max()]
        fig = px.scatter_3d(
            df,
            z=timestamp_col,
            range_z=z_range,
            title=f"Spatial-Temporal Distribution of {label_col}",
            labels={timestamp_col: "Time (s)"},
            **common_scatter_kwargs,  # type: ignore
        )
    else:
        z_range = [df[z_col].min(), df[z_col].max()]
        time_series = (
            pd.to_timedelta(df[timestamp_col], unit="s")
            if relative_time
            else df[timestamp_col]
        )
        t_min, t_max = time_series.min(), time_series.max()

        if relative_time:
            bin_edges = pd.timedelta_range(t_min, t_max, periods=n_bins + 1)
            bin_labels = [
                f"{bin_edges[i].total_seconds():.2f}s – "
                f"{bin_edges[i + 1].total_seconds():.2f}s"
                for i in range(n_bins)
            ]
        else:
            bin_edges = pd.date_range(t_min, t_max, periods=n_bins + 1)  # type: ignore
            bin_labels = [
                f"{bin_edges[i].strftime('%H:%M:%S')}–"
                f"{bin_edges[i + 1].strftime('%H:%M:%S')}"
                for i in range(n_bins)
            ]

        df["time_bin"] = pd.cut(
            time_series, bins=bin_edges, labels=bin_labels, include_lowest=True  # type: ignore
        )
        df = df.dropna(subset=["time_bin"])
        df["time_bin"] = df["time_bin"].astype(str)

        frame_order = (
            df.drop_duplicates("time_bin")
            .sort_values(timestamp_col)["time_bin"]
            .tolist()
        )
        df["time_bin"] = pd.Categorical(
            df["time_bin"], categories=frame_order, ordered=True
        )
        df = df.sort_values("time_bin")

        fig = px.scatter_3d(
            df,
            z=z_col,
            animation_frame="time_bin",
            range_z=z_range,
            title=f"Spatial-Temporal Distribution of {label_col} ({n_bins} time bins)",
            **common_scatter_kwargs,  # type: ignore
        )
        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 600  # type: ignore
        fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 200  # type: ignore

    if graph_towers:
        fig = _add_tower_trace(fig, project_cords)

    fig.update_traces(marker=dict(size=3))
    _show_or_save(fig, save_path)


def format_timedelta(td: pd.Timedelta) -> str:
    total_seconds = int(td.total_seconds())
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    milliseconds = td.microseconds // 1000
    return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def combine_dfs_graph(
    data_dict: dict[int, dict[str, pd.DataFrame]],
    flight_ids: list[str],
    label: str | None,
) -> tuple[pd.DataFrame, str]:
    all_ids = get_all_flight_ids()
    invalid = [fid for fid in flight_ids if fid not in all_ids]
    if invalid:
        raise ValueError(f"Invalid flight ID(s): {invalid}. Available: {all_ids}")

    dfs: list[pd.DataFrame] = []
    for fid in flight_ids:
        dataset_num, flight_name = get_dataset_and_flight_from_id(fid)
        df = data_dict[dataset_num][flight_name]
        df["Flight_ID"] = fid
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    if label is None:
        return combined, ""

    common_label_cols: set[str] = set.intersection(  # type: ignore
        *[set(get_all_matching_cols(df, label)) for df in dfs]
    )

    if not common_label_cols:
        raise ValueError(
            f"No common label column for '{label}' across the provided flight IDs."
        )

    label_col = common_label_cols.pop()
    if common_label_cols:
        logger.warning(
            f"Multiple common label columns for '{label}': "
            f"{{{label_col}, *{common_label_cols}}}. Using '{label_col}'."
        )

    return combined, label_col
