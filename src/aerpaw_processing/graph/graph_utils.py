from collections import defaultdict
import logging
import numpy as np
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
            if len(label_cols) > 1:
                current_save_path = str(p.with_stem(f"{p.stem}_{target_label}"))
            else:
                current_save_path = save_path

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


# ---------------------------------------------------------------------------
# Spatial RSRP Correlation (replicates Fig. 3 from the IEEE JSTEAP paper)
# ---------------------------------------------------------------------------


def _uav_to_spherical(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Convert UAV positions to spherical coordinates relative to the BS.

    Returns a copy of *df* with three new columns:
      - ``d3D``   : 3-D Euclidean distance from the BS (metres)
      - ``elev``  : elevation angle from BS horizontal plane (radians)
      - ``azim``  : azimuth angle (radians, [0, 2π))
    """
    df = project_coordinates(df.copy())

    dx = df["x"]
    dy = df["y"]
    dz = df["z"] - towers[0].alt

    d3D = np.sqrt(dx**2 + dy**2 + dz**2)

    with np.errstate(invalid="ignore"):
        elev = np.where(d3D > 0, np.arcsin(np.clip(dz / d3D, -1.0, 1.0)), 0.0)
    azim = (np.arctan2(dy, dx)) % (2 * np.pi)

    out = df.copy()
    out["d3D"] = d3D
    out["elev"] = elev
    out["azim"] = azim
    return out


def graph_spatial_rsrp_correlation(
    df: pd.DataFrame,
    label_col: str,
    angular_bin_size_rad: float = 0.05,
    radial_bin_size_m: float = 5.0,
    title: str | None = None,
    save_path: str | None = None,
) -> pd.DataFrame:
    """Plot average pairwise RSRP spatial correlation vs. radial separation.

    Reproduces Figure 3 from the paper *AI-Enabled Wireless Propagation
    Modeling and Radio Environment Maps for Aerial Wireless Networks*.

    The algorithm:

    1. **Angular Binning** – Convert UAV positions to spherical coordinates
       centred at the base station (``towers[0]``).  Partition the sphere into
       elevation × azimuth cells of width *angular_bin_size_rad* radians and
       group all measurements that fall into the same cell.

    2. **Correlation Computation** – Within each angular cell compute every
       pairwise normalised correlation

           r[m, n] = (Ωm − Ω̄)(Ωn − Ω̄) / σ²

       where Ω̄ and σ² are the global mean and variance of *label_col*.
       Each pair is then placed into a *radial-separation* bin of width
       *radial_bin_size_m* metres, where radial separation is

           Δd = |d3D(m) − d3D(n)|

    3. **Statistical Aggregation** – Average all correlation values that fall
       into each radial-separation bin.

    The resulting figure mirrors the paper's Fig. 3: a dual-axis matplotlib
    plot with the average correlation on the left y-axis (line) and the
    number of contributing point-pairs on the right y-axis (bar).

    Args:
        df: DataFrame containing at least *label_col*, ``Latitude``,
            ``Longitude``, and ``Altitude`` columns.
        label_col: Name of the RSRP (or other signal) column to correlate.
        angular_bin_size_rad: Angular cell size in radians (default 0.05,
            matching the paper).
        radial_bin_size_m: Width of each radial-separation histogram bin in
            metres (default 5, matching the paper).
        title: Optional plot title.  Defaults to
            ``"Spatial Correlation of <label_col> (spherical)"``
        save_path: If provided, save the figure here instead of showing it.

    Returns:
        A DataFrame with columns ``radial_separation_m``, ``avg_correlation``,
        and ``num_pairs`` (one row per radial-separation bin) for downstream
        inspection.
    """
    # ------------------------------------------------------------------
    # 1. Drop rows with missing signal or location data
    # ------------------------------------------------------------------
    required_cols = [label_col, "Latitude", "Longitude", "Altitude"]
    working = df[required_cols].dropna().reset_index(drop=True)

    if working.empty:
        raise ValueError(f"No valid rows found after dropping NaNs in {required_cols}.")

    # ------------------------------------------------------------------
    # 2. Compute spherical coordinates relative to the base station
    # ------------------------------------------------------------------
    working = _uav_to_spherical(working)

    # ------------------------------------------------------------------
    # 3. Global RSRP statistics (used for all pairwise correlations)
    # ------------------------------------------------------------------
    rsrp = working[label_col].to_numpy(dtype=float)
    omega_bar = rsrp.mean()
    sigma2 = rsrp.var()

    if sigma2 == 0:
        raise ValueError(
            f"Zero variance in '{label_col}' – cannot compute correlation."
        )

    centered = rsrp - omega_bar  # (Ωm − Ω̄)

    # ------------------------------------------------------------------
    # 4. Angular binning
    # ------------------------------------------------------------------
    elev_bins = np.floor(working["elev"].to_numpy() / angular_bin_size_rad).astype(int)
    azim_bins = np.floor(working["azim"].to_numpy() / angular_bin_size_rad).astype(int)
    bin_keys = list(zip(elev_bins, azim_bins))

    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, key in enumerate(bin_keys):
        groups[key].append(idx)

    # ------------------------------------------------------------------
    # 5. Pairwise correlation, binned by radial separation
    # ------------------------------------------------------------------
    d3D = working["d3D"].to_numpy(dtype=float)
    max_sep = d3D.max()  # upper bound for bins
    n_radial_bins = int(np.ceil(max_sep / radial_bin_size_m)) + 1

    corr_sums = np.zeros(n_radial_bins, dtype=float)
    corr_counts = np.zeros(n_radial_bins, dtype=int)

    for indices in groups.values():
        if len(indices) < 2:
            continue
        idx_arr = np.array(indices)
        # Vectorised pairwise computation within the angular cell
        cent_g = centered[idx_arr]  # shape (k,)
        d3D_g = d3D[idx_arr]  # shape (k,)

        # All unique pairs (i < j)
        i_idx, j_idx = np.triu_indices(len(idx_arr), k=1)

        pair_corr = (cent_g[i_idx] * cent_g[j_idx]) / sigma2
        pair_sep = np.abs(d3D_g[i_idx] - d3D_g[j_idx])

        bin_idx = np.floor(pair_sep / radial_bin_size_m).astype(int)
        bin_idx = np.clip(bin_idx, 0, n_radial_bins - 1)

        np.add.at(corr_sums, bin_idx, pair_corr)
        np.add.at(corr_counts, bin_idx, 1)

    # ------------------------------------------------------------------
    # 6. Average per radial-separation bin
    # ------------------------------------------------------------------
    valid_mask = corr_counts > 0
    avg_corr = np.where(valid_mask, corr_sums / corr_counts, np.nan)
    bin_centres = (np.arange(n_radial_bins) + 0.5) * radial_bin_size_m

    result_df = pd.DataFrame(
        {
            "radial_separation_m": bin_centres[valid_mask],
            "avg_correlation": avg_corr[valid_mask],
            "num_pairs": corr_counts[valid_mask],
        }
    )

    # ------------------------------------------------------------------
    # 7. Plot (dual y-axis, matching the paper's Fig. 3 style)
    # ------------------------------------------------------------------
    _, ax1 = plt.subplots(figsize=(7, 4))

    x = result_df["radial_separation_m"].to_numpy()
    y_corr = result_df["avg_correlation"].to_numpy()
    y_count = result_df["num_pairs"].to_numpy()

    # Right axis – number of pairs (salmon/pink bars, behind the line)
    ax2 = ax1.twinx()
    ax2.bar(
        x,
        y_count,
        width=radial_bin_size_m * 0.9,
        color="salmon",
        alpha=0.45,
        label="Num Points",
        zorder=1,
    )
    ax2.set_ylabel("Num Points", fontsize=11)
    ax2.set_ylim(bottom=0)

    # Left axis – average correlation (blue line, on top)
    ax1.plot(
        x, y_corr, color="steelblue", linewidth=1.8, label="Avg Correlation", zorder=2
    )
    ax1.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax1.set_ylim(-1.0, 1.0)
    ax1.set_xlabel("Radial separation distance (m)", fontsize=11)
    ax1.set_ylabel("Avg Correlation", fontsize=11)
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)

    # Legend – combine both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

    plot_title = title or f"Spatial Correlation of {label_col} (spherical)"
    plt.title(plot_title, fontsize=12)
    plt.tight_layout()

    _show_or_save_plt(save_path)

    return result_df


# ---------------------------------------------------------------------------
# Fast-Fading Factor Correlation (replicates Fig. 4 from the IEEE JSTEAP paper)
# ---------------------------------------------------------------------------


def _fit_los_path_loss(d3D: np.ndarray, rsrp: np.ndarray) -> np.ndarray:
    """Fit a log-distance LoS path loss model and return the fitted RSRP values.

    Model:  RSRP_LoS(d) = α − β · log10(d)

    A least-squares fit of *rsrp* against *log10(d3D)* is performed to
    estimate the intercept α and slope β.  The fitted values are returned
    so the caller can compute the residual fast-fading factor

        ν = rsrp − RSRP_LoS(d)

    Points where d3D ≤ 0 are excluded from the fit but are assigned the
    intercept value (α) so the caller always receives an array of the same
    length as *rsrp*.
    """
    valid = d3D > 0
    log_d = np.log10(d3D[valid])
    # Design matrix: [1, log10(d)] for intercept + slope
    A = np.column_stack([np.ones(log_d.shape), log_d])
    coeffs, _, _, _ = np.linalg.lstsq(A, rsrp[valid], rcond=None)
    alpha, beta = coeffs

    los_all = np.full_like(rsrp, alpha, dtype=float)
    los_all[valid] = alpha + beta * log_d
    return los_all


def graph_fast_fading_correlation(
    df: pd.DataFrame,
    label_col: str,
    window: int = 5,
    radial_bin_size_m: float = 3.0,
    fast_fading_col: str | None = None,
    title: str | None = None,
    save_path: str | None = None,
) -> pd.DataFrame:
    """Plot fast-fading factor correlation vs. spatial separation distance.

    Reproduces Figure 4 from the paper *AI-Enabled Wireless Propagation
    Modeling and Radio Environment Maps for Aerial Wireless Networks*.

    **Algorithm**

    1. **Fast-fading extraction** – If *fast_fading_col* is not supplied, a
       log-distance LoS path loss model (``RSRP_LoS = α − β·log10(d3D)``) is
       fitted to the data via least squares and subtracted from the measured
       RSRP, yielding the residual fast-fading factor

           ν = Ω − RSRP_LoS(d)

       This mirrors the paper's ``ν = Ω − P_TX_SS − PL_LoS``.

    2. **Rolling-window correlation** – For each sample *i*, the correlation
       with each of the *window* preceding samples *m* ∈ {1, …, window} is
       computed as

           r[i, m] = (νᵢ · νᵢ₋ₘ) / σ²ν

       where σ²ν is the variance of all fast-fading values in the dataset.

    3. **Spatial binning** – Each (r[i,m], d3D(UAV_i, UAV_{i-m})) pair is
       placed into a radial-separation bin of width *radial_bin_size_m* metres.

    4. **Aggregation & plot** – Correlation values within each bin are
       averaged and plotted as a line graph, matching the style of Fig. 4.

    Args:
        df: DataFrame containing at least *label_col*, ``Latitude``,
            ``Longitude``, and ``Altitude`` columns, ordered chronologically.
        label_col: The RSRP (or signal) column name.
        window: Number of preceding samples to include in the rolling
            correlation (default 5, matching the paper).
        radial_bin_size_m: Spatial separation bin width in metres
            (default 3, matching the paper).
        fast_fading_col: Optional pre-computed fast-fading column name.
            When provided the LoS fitting step is skipped entirely.
        title: Optional plot title.
        save_path: If provided, save the figure here instead of showing it.

    Returns:
        A DataFrame with columns ``spatial_separation_m``, ``avg_correlation``,
        and ``num_pairs`` (one row per separation bin).
    """
    # ------------------------------------------------------------------
    # 1. Drop rows with missing data and preserve chronological order
    # ------------------------------------------------------------------
    required_cols = [label_col, "Latitude", "Longitude", "Altitude"]
    if fast_fading_col:
        required_cols.append(fast_fading_col)

    working = df[required_cols].dropna().reset_index(drop=True)

    if len(working) < window + 1:
        raise ValueError(
            f"Not enough valid rows ({len(working)}) for a rolling window of {window}."
        )

    # ------------------------------------------------------------------
    # 2. Compute 3-D UAV-to-UAV distances (not UAV-to-BS)
    # ------------------------------------------------------------------
    working = project_coordinates(working)

    # Cartesian UAV positions (metres, origin = BS)
    uav_x = working["x"]
    uav_y = working["y"]
    uav_z = working["z"] - towers[0].alt

    # UAV-to-BS distance (needed for LoS fit)
    d3D_bs = np.sqrt(uav_x**2 + uav_y**2 + uav_z**2)

    # ------------------------------------------------------------------
    # 3. Extract / compute fast-fading factor ν
    # ------------------------------------------------------------------
    rsrp = working[label_col].to_numpy(dtype=float)

    if fast_fading_col:
        nu = working[fast_fading_col].to_numpy(dtype=float)
    else:
        los_fitted = _fit_los_path_loss(d3D_bs, rsrp)
        nu = rsrp - los_fitted

    sigma2_nu = nu.var()
    if sigma2_nu == 0:
        raise ValueError(
            "Zero variance in the fast-fading factor – cannot compute correlation."
        )

    # ------------------------------------------------------------------
    # 4. Rolling-window pairwise correlation, binned by UAV-to-UAV separation
    # ------------------------------------------------------------------
    n = len(nu)
    max_possible_sep = np.sqrt(
        (uav_x.max() - uav_x.min()) ** 2
        + (uav_y.max() - uav_y.min()) ** 2
        + (uav_z.max() - uav_z.min()) ** 2
    )
    n_bins = int(np.ceil(max_possible_sep / radial_bin_size_m)) + 1

    corr_sums = np.zeros(n_bins, dtype=float)
    corr_counts = np.zeros(n_bins, dtype=int)

    for i in range(window, n):
        for m in range(1, window + 1):
            j = i - m
            # r[i, m] = (νᵢ · νᵢ₋ₘ) / σ²ν
            r = (nu[i] * nu[j]) / sigma2_nu

            # UAV-to-UAV 3-D separation
            dx = uav_x[i] - uav_x[j]
            dy = uav_y[i] - uav_y[j]
            dz = uav_z[i] - uav_z[j]
            sep = np.sqrt(dx**2 + dy**2 + dz**2)

            bin_idx = min(int(sep / radial_bin_size_m), n_bins - 1)
            corr_sums[bin_idx] += r
            corr_counts[bin_idx] += 1

    # ------------------------------------------------------------------
    # 5. Average per bin
    # ------------------------------------------------------------------
    valid_mask = corr_counts > 0
    avg_corr = np.where(valid_mask, corr_sums / corr_counts, np.nan)
    bin_centres = (np.arange(n_bins) + 0.5) * radial_bin_size_m

    result_df = pd.DataFrame(
        {
            "spatial_separation_m": bin_centres[valid_mask],
            "avg_correlation": avg_corr[valid_mask],
            "num_pairs": corr_counts[valid_mask],
        }
    )

    # ------------------------------------------------------------------
    # 6. Plot – single axis line chart matching Fig. 4 style
    # ------------------------------------------------------------------
    _, ax = plt.subplots(figsize=(7, 4))

    x = result_df["spatial_separation_m"].to_numpy()
    y = result_df["avg_correlation"].to_numpy()

    ax.plot(x, y, marker="s", markersize=4, linewidth=1.8, color="steelblue")
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax.set_xlabel("Spatial separation distance (m)", fontsize=11)
    ax.set_ylabel("Avg Correlation", fontsize=11)
    ax.set_ylim(-0.5, 1.05)
    ax.grid(axis="both", linestyle="--", alpha=0.4)

    plot_title = (
        title or f"Fast-Fading Correlation of {label_col} (rolling window={window})"
    )
    ax.set_title(plot_title, fontsize=12)
    plt.tight_layout()

    _show_or_save_plt(save_path)

    return result_df
