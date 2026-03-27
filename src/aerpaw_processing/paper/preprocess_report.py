import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import cast
import plotly.express as px
import os

from aerpaw_processing.resources.tower_locations import towers
from aerpaw_processing.resources.config.config_init import load_env
from aerpaw_processing.paper.preprocess_utils import (
    StepEnum,
    get_label_col,
    get_step_entry,
    get_env_var,
)

load_env()

logger = logging.getLogger(__name__)

BASE_TOWER = towers[0]


def _generate_3d_plot(
    step_df: pd.DataFrame,
    report_dir: Path,
    step_name: str,
    label: str,
    title_suffix: str,
    report: list,
    is_categorical: bool = False,
):
    assets_dir = report_dir / "report_assets"
    assets_dir.mkdir(exist_ok=True, parents=True)

    import plotly.graph_objects as go

    for _, r in cast(pd.DataFrame, step_df).iterrows():
        df: pd.DataFrame = cast(pd.DataFrame, r["data"])
        flight_name = r["flight_name"]
        dataset_id = r["dataset_id"]

        rsrp_col = get_label_col(df)

        label_col = get_label_col(df, label)

        if rsrp_col == label_col:
            rsrp_col = "Timestamp"

        if "x" in df.columns and "y" in df.columns and "z" in df.columns:
            plot_df = df.dropna(subset=["x", "y", "z"])
            if label_col:
                plot_df = plot_df.dropna(subset=[label_col])

            if not plot_df.empty:
                if is_categorical and label_col:
                    plot_df[label_col] = plot_df[label_col].astype(str)

                labels = {"x": "x (m)", "y": "y (m)", "z": "z (m)"}

                fig = px.scatter_3d(
                    plot_df,
                    x="x",
                    y="y",
                    z="z",
                    color=label_col,
                    title=f"3D trajectory: Dataset {dataset_id} - {flight_name}{title_suffix}",
                    labels=labels,
                    template="plotly_dark",
                    color_discrete_sequence=px.colors.qualitative.Alphabet
                    if is_categorical
                    else None,
                    hover_data=[rsrp_col],
                )

                # Add base tower as a point
                fig.add_trace(
                    go.Scatter3d(
                        x=[0],
                        y=[0],
                        z=[10],
                        mode="markers",
                        marker=dict(size=8, color="red", symbol="diamond"),
                        name="Base Tower",
                    )
                )

                if step_name == "bin":
                    BIN_SIZE = 0.05
                    num_phi_bins = int(np.ceil(2 * np.pi / BIN_SIZE))

                    if "bin" in plot_df.columns and not plot_df.empty:
                        # 1. Bin with most points
                        bin1 = plot_df["bin"].value_counts().idxmax()

                        # 2. Bin with the furthest point in the x direction
                        bin2 = plot_df.sort_values("x", ascending=False).iloc[0]["bin"]

                        # 3. Bin in the middle of x and y range
                        mid_x = (plot_df["x"].max() + plot_df["x"].min()) / 2
                        mid_y = (plot_df["y"].max() + plot_df["y"].min()) / 2
                        dist_sq = (plot_df["x"] - mid_x) ** 2 + (
                            plot_df["y"] - mid_y
                        ) ** 2
                        temp_df = plot_df.copy()
                        temp_df["dist"] = dist_sq
                        bin3 = temp_df.sort_values("dist").iloc[0]["bin"]

                        bins_to_plot = []
                        seen_bins = set()
                        colors = ["yellow", "cyan", "magenta"]
                        names = ["Most Points", "Furthest X", "Middle XY"]

                        for i, (b_val, name_desc) in enumerate(
                            zip([bin1, bin2, bin3], names)
                        ):
                            if b_val not in seen_bins:
                                seen_bins.add(b_val)
                                bins_to_plot.append(
                                    {
                                        "bin_id": b_val,
                                        "color": colors[i],
                                        "name": f"Bin: {name_desc}",
                                    }
                                )

                        for b_info in bins_to_plot:
                            b_str = b_info["bin_id"]
                            b = int(float(b_str))

                            b_df = plot_df[plot_df["bin"] == b_str]
                            r_max = np.sqrt(
                                b_df["x"] ** 2 + b_df["y"] ** 2 + (b_df["z"] - 10) ** 2
                            ).max()
                            if pd.isna(r_max) or r_max == 0:
                                r_max = 100

                            bin_theta = b // num_phi_bins
                            bin_phi = b % num_phi_bins

                            theta_min = bin_theta * BIN_SIZE
                            theta_max = (bin_theta + 1) * BIN_SIZE
                            phi_min = bin_phi * BIN_SIZE
                            phi_max = (bin_phi + 1) * BIN_SIZE

                            rays = [
                                (theta_min, phi_min),
                                (theta_min, phi_max),
                                (theta_max, phi_min),
                                (theta_max, phi_max),
                            ]

                            for idx, (t, p) in enumerate(rays):
                                fig.add_trace(
                                    go.Scatter3d(
                                        x=[0, r_max * np.sin(t) * np.cos(p)],
                                        y=[0, r_max * np.sin(t) * np.sin(p)],
                                        z=[10, 10 + r_max * np.cos(t)],
                                        mode="lines",
                                        line=dict(color=b_info["color"], width=4),
                                        name=b_info["name"]
                                        if idx == 0
                                        else f"{b_info['name']} line {idx + 1}",
                                        showlegend=(idx == 0),
                                    )
                                )

                fig.update_layout(
                    scene=dict(
                        xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title="z (m)"
                    ),
                    margin=dict(l=0, r=0, b=0, t=30),
                )

                safe_flight_name = flight_name.replace(" ", "_").replace("/", "_")
                step_suffix = f"_{step_name}" if step_name else ""
                base_name = f"dataset_{dataset_id}_{safe_flight_name}{step_suffix}_3d"
                png_filename = f"{base_name}.png"
                html_filename = f"{base_name}.html"

                png_path = assets_dir / png_filename
                html_path = assets_dir / html_filename

                fig.write_html(str(html_path))

                try:
                    fig.write_image(str(png_path))
                    report.append(f"#### Flight: {flight_name}\n")
                    rel_png = f"report_assets/{png_filename}"
                    rel_html = f"report_assets/{html_filename}"
                    report.append(f"[![{flight_name}]({rel_png})]({rel_html})\n")
                except Exception as e:
                    logger.warning(
                        f"Could not save static image for {flight_name}: {e}. Ensure 'kaleido' is installed."
                    )
                    rel_html = f"report_assets/{html_filename}"
                    report.append(f"#### Flight: {flight_name}\n")
                    report.append(f"[Interactive 3D Plot]({rel_html})\n")
            else:
                report.append(f"#### Flight: {flight_name}\n")
                report.append("No valid data points for 3D plot.\n")


def report_read_data(step_df: pd.DataFrame, report: list[str]):
    num_files = len(step_df)
    report.append(f"- **Total files read**: {num_files}")
    report.append("\n### Datasets Read\n")
    report.append("| Dataset | Tech | Flight | Filepath | Rows | Cols |")
    report.append("| --- | --- | --- | --- | --- | --- |")
    for _, r in cast(pd.DataFrame, step_df).iterrows():
        df = r["data"]
        report.append(
            f"| {r['dataset_id']} | {r['tech']} | {r['flight_name']} | {r['filepath']} | {df.shape[0]:,} | {df.shape[1]} |"
        )
    report.append("\n")


def report_rename_columns(
    context: pd.DataFrame, step_df: pd.DataFrame, report: list[str]
):
    read_data_df = get_step_entry(StepEnum.READ_DATA, context)
    report.append("- Columns renamed to standard names using `alias_map`.")
    report.append("\n### Columns Renamed per Dataset\n")
    report.append("| Dataset | Tech | Flight | Filepath | Cols Changed |")
    report.append("| --- | --- | --- | --- | --- |")

    for i in range(len(step_df)):
        old_df = read_data_df.iloc[i]["data"]
        new_df = cast(pd.DataFrame, step_df).iloc[i]["data"]

        # Count columns where the name has changed
        changed_count = sum(
            1 for old, new in zip(old_df.columns, new_df.columns) if old != new
        )

        r = cast(pd.DataFrame, step_df).iloc[i]
        report.append(
            f"| {r['dataset_id']} | {r['tech']} | {r['flight_name']} | {r['filepath']} | {changed_count} |"
        )

    # Show columns from the first dataset as a sample of standardized names
    cols = cast(pd.DataFrame, step_df["data"]).iloc[0].columns.tolist()
    report.append(f"\n- **Total standardized columns (sample)**: {len(cols)}")
    report.append(f"- **Columns (sample)**: `{', '.join(cols)}`")
    report.append("\n")


def report_remove_columns(
    context: pd.DataFrame, step_df: pd.DataFrame, report: list[str]
):
    rename_data_df = get_step_entry(StepEnum.RENAME_COLUMNS, context)
    report.append("- Unwanted columns (not defined in CONFIG) removed.")
    report.append("\n### Columns Removed per Dataset\n")
    report.append(
        "| Dataset | Tech | Flight | Filepath | Cols Removed | Remaining Columns |"
    )
    report.append("| --- | --- | --- | --- | --- | --- |")

    for i in range(len(step_df)):
        old_df = rename_data_df.iloc[i]["data"]
        new_df = cast(pd.DataFrame, step_df).iloc[i]["data"]

        removed_count = len(old_df.columns) - len(new_df.columns)
        remaining_cols = ", ".join(new_df.columns.tolist())

        r = cast(pd.DataFrame, step_df).iloc[i]
        report.append(
            f"| {r['dataset_id']} | {r['tech']} | {r['flight_name']} | {r['filepath']} | {removed_count} | `{remaining_cols}` |"
        )
    report.append("\n")


def report_combine_tech_files(
    context: pd.DataFrame, step_df: pd.DataFrame, report: list[str]
):

    prev_step = (
        StepEnum.REMOVE_COLUMNS
        if StepEnum.REMOVE_COLUMNS.value in context["step"].values
        else StepEnum.RENAME_COLUMNS
    )
    prev_step_df = get_step_entry(prev_step, context)

    num_combined = len(step_df)
    report.append(
        f"- Multiple KPI files combined into {num_combined} technology-specific datasets per flight."
    )
    report.append("\n### Combined Datasets\n")
    report.append("| Dataset | Tech | Flight | Rows | Cols | Prev Rows | Columns |")
    report.append("| --- | --- | --- | --- | --- | --- | --- |")
    for _, r in cast(pd.DataFrame, step_df).iterrows():
        df = r["data"]

        # Get row counts of files before combination
        matched_prev = prev_step_df[
            (prev_step_df["dataset_id"] == r["dataset_id"])
            & (prev_step_df["flight_name"] == r["flight_name"])
            & (prev_step_df["tech"] == r["tech"])
        ]
        prev_rows = [f"{d.shape[0]:,}" for d in matched_prev["data"]]
        prev_rows_str = ", ".join(prev_rows)
        cols_str = ", ".join(cast(pd.DataFrame, df).columns.tolist())

        report.append(
            f"| {r['dataset_id']} | {r['tech']} | {r['flight_name']} | {df.shape[0]:,} | {df.shape[1]} | {prev_rows_str} | `{cols_str}`"
        )

    report.append("\n")


def report_combine_flight_techs(
    context: pd.DataFrame, step_df: pd.DataFrame, report: list[str]
):

    prev_step_df = get_step_entry(StepEnum.COMBINE_TECH_FILES, context)

    num_combined = len(step_df)
    report.append(
        f"- Multiple technologies combined into {num_combined} single flight datasets."
    )
    report.append("\n### Combined Flight Datasets\n")
    report.append("| Dataset | Flight | Rows | Cols | Techs Combined | Columns |")
    report.append("| --- | --- | --- | --- | --- | --- |")
    for _, r in cast(pd.DataFrame, step_df).iterrows():
        df = r["data"]

        matched_prev = prev_step_df[
            (prev_step_df["dataset_id"] == r["dataset_id"])
            & (prev_step_df["flight_name"] == r["flight_name"])
        ]
        techs = ", ".join(matched_prev["tech"].tolist())
        cols_str = ", ".join(cast(pd.DataFrame, df).columns.tolist())

        report.append(
            f"| {r['dataset_id']} | {r['flight_name']} | {df.shape[0]:,} | {df.shape[1]} | {techs} | `{cols_str}`"
        )

    report.append("\n")


def report_interpolate_to_label(
    context: pd.DataFrame, step_df: pd.DataFrame, report: list[str]
):

    prev_step_df = get_step_entry(StepEnum.COMBINE_FLIGHT_TECHS, context)
    report.append("- Interpolated and standardized data around label columns.")
    report.append("\n### Interpolated Datasets\n")
    report.append("| Dataset | Flight | Rows | Cols | Prev Rows |")
    report.append("| --- | --- | --- | --- | --- |")
    for _, r in cast(pd.DataFrame, step_df).iterrows():
        df = r["data"]
        matched_prev = prev_step_df[
            (prev_step_df["dataset_id"] == r["dataset_id"])
            & (prev_step_df["flight_name"] == r["flight_name"])
        ]
        prev_rows = (
            matched_prev.iloc[0]["data"].shape[0] if not matched_prev.empty else 0
        )
        report.append(
            f"| {r['dataset_id']} | {r['flight_name']} | {df.shape[0]:,} | {df.shape[1]} | {prev_rows:,} |"
        )
    report.append("\n")


def generate_report(context: pd.DataFrame, mad_filter: bool):
    """Generate a markdown report from the processing context."""

    # context_home = Path(get_env_var("CONTEXT_HOME"))
    script_dir = Path(__file__).resolve().parent
    report_home = script_dir / "../../../"
    if not mad_filter:
        report_home = report_home / "report/"
    else:
        report_home = report_home / "report_mad/"

    os.makedirs(report_home, exist_ok=True)

    report = [
        "# Preprocessing Report\n",
        f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
    ]

    for _, row in context.iterrows():
        step_val = row["step"]
        step_df = cast(pd.DataFrame, row["step_data"])

        report.append(f"## Step: {step_val}\n")

        if step_val == StepEnum.READ_DATA.value:
            report_read_data(step_df, report)

        elif step_val == StepEnum.RENAME_COLUMNS.value:
            report_rename_columns(context, step_df, report)

        elif step_val == StepEnum.REMOVE_COLUMNS.value:
            report_remove_columns(context, step_df, report)

        elif step_val == StepEnum.COMBINE_TECH_FILES.value:
            report_combine_tech_files(context, step_df, report)

        elif step_val == StepEnum.COMBINE_FLIGHT_TECHS.value:
            report_combine_flight_techs(context, step_df, report)

        elif step_val == StepEnum.INTERPOLATE_TO_LABEL.value:
            report_interpolate_to_label(context, step_df, report)

        elif step_val == StepEnum.PROJECT_COORDINATES.value:
            report.append(
                "- Projected Latitude/Longitude to x/y coordinates using AEQD projection."
            )
            report.append(
                f"- Base Tower (Origin): {BASE_TOWER.name} ({BASE_TOWER.lat}, {BASE_TOWER.lon})"
            )
            report.append("\n### Projected 3D Trajectories (Colored by RSRP)\n")

            _generate_3d_plot(step_df, report_home, "", "RSRP", "", report)

            report.append("\n")

        elif step_val == StepEnum.MAD_FILTER.value:
            report.append(
                "- Projected Latitude/Longitude to x/y coordinates using AEQD projection."
            )
            report.append(
                f"- Base Tower (Origin): {BASE_TOWER.name} ({BASE_TOWER.lat}, {BASE_TOWER.lon})"
            )
            report.append("\n### Projected 3D Trajectories (Colored by RSRP)\n")

            _generate_3d_plot(step_df, report_home, "mad", "RSRP", "", report)

            report.append("\n")

        elif step_val == StepEnum.CALCULATE_BIN.value:
            report.append(
                "- Partitioned the 3D data points into a spherical grid centered at the Base Station with 0.05 rad bins."
            )
            report.append("\n### Projected 3D Trajectories (Colored by Angular Bin)\n")

            _generate_3d_plot(
                step_df,
                report_home,
                "bin",
                "bin",
                " (Angular Bins)",
                report,
                is_categorical=True,
            )
            report.append("\n")

        elif step_val == StepEnum.CORRELATION_COMPUTATION.value:
            report.append(
                "- Computed pairwise spatial correlation of RSRP for each angular bin and aggregated by 5m radial separation distance bins."
            )

            report.append("\n**Note on Pairwise Point Generation:**")
            report.append(
                "For a bin with `N` points, the correlation is computed between every point and every other point in that *same* bin."
            )
            report.append(
                "Because we pair every point with every other point, the total number of generated correlation pairs will be much larger than the original data points (Quadratic Growth)."
            )
            report.append("\n```text")
            report.append("Visual Example (Bin with 4 points: A, B, C, D):")
            report.append("Pairwise combinations (N=4, so N*(N-1)/2 = 6 pairs):")
            report.append("A <-> B\nA <-> C\nA <-> D\nB <-> C\nB <-> D\nC <-> D")
            report.append("```\n")

            report.append("\n### Spatial Correlation Profiles\n")

            assets_dir = report_home / "report_assets"
            assets_dir.mkdir(exist_ok=True, parents=True)

            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            calc_bin_df = get_step_entry(StepEnum.CALCULATE_BIN, context)
            BIN_SIZE = 0.05
            num_phi_bins = int(np.ceil(2 * np.pi / BIN_SIZE))

            for _, r in cast(pd.DataFrame, step_df).iterrows():
                df: pd.DataFrame = r["data"]
                flight_name = r["flight_name"]
                dataset_id = r["dataset_id"]
                bin_stats = r.get("bin_stats")

                if df.empty:
                    report.append(f"#### Flight: {flight_name}\n")
                    report.append("No correlation data generated.\n")
                    continue

                fig = make_subplots(specs=[[{"secondary_y": True}]])

                fig.add_trace(
                    go.Bar(
                        x=df["radial_bin"],
                        y=df["num_pairs"],
                        name="Num Points",
                        opacity=0.3,
                        marker_color="red",
                    ),
                    secondary_y=True,
                )

                fig.add_trace(
                    go.Scatter(
                        x=df["radial_bin"],
                        y=df["r"],
                        name="Avg Correlation",
                        mode="lines+markers",
                        line=dict(color="blue"),
                    ),
                    secondary_y=False,
                )

                fig.update_layout(
                    title_text=f"Spatial Correlation: Dataset {dataset_id} - {flight_name}",
                    template="plotly_white",
                )

                fig.update_xaxes(title_text="Radial separation distance (m)")
                fig.update_yaxes(
                    title_text="Avg Correlation", secondary_y=False, range=[-1.0, 1.0]
                )
                fig.update_yaxes(
                    title_text="Num Points", secondary_y=True, showgrid=False
                )

                safe_flight_name = flight_name.replace(" ", "_").replace("/", "_")
                base_name = f"dataset_{dataset_id}_{safe_flight_name}_correlation"
                png_filename = f"{base_name}.png"
                html_filename = f"{base_name}.html"

                png_path = assets_dir / png_filename
                html_path = assets_dir / html_filename

                fig.write_html(str(html_path))

                try:
                    fig.write_image(str(png_path))
                    report.append(f"#### Flight: {flight_name}\n")
                    rel_png = f"report_assets/{png_filename}"
                    rel_html = f"report_assets/{html_filename}"
                    report.append(f"[![{flight_name}]({rel_png})]({rel_html})\n")
                except Exception as e:
                    logger.warning(
                        f"Could not save static image for {flight_name}: {e}. Ensure 'kaleido' is installed."
                    )
                    rel_html = f"report_assets/{html_filename}"
                    report.append(f"#### Flight: {flight_name}\n")
                    report.append(f"[Interactive Correlation Plot]({rel_html})\n")

                if bin_stats is not None and not bin_stats.empty:
                    top_corr_bin = bin_stats.loc[bin_stats["r"].idxmax()]["bin"]
                    top_dist_bins = (
                        bin_stats.sort_values("dist", ascending=False)
                        .head(2)["bin"]
                        .tolist()
                    )

                    flight_points_row = calc_bin_df[
                        (calc_bin_df["dataset_id"] == dataset_id)
                        & (calc_bin_df["flight_name"] == flight_name)
                    ]
                    if not flight_points_row.empty:
                        plot_df = flight_points_row.iloc[0]["data"]
                        if (
                            "x" in plot_df.columns
                            and "y" in plot_df.columns
                            and "z" in plot_df.columns
                            and "bin" in plot_df.columns
                        ):
                            plot_df_clean = plot_df.dropna(
                                subset=["x", "y", "z", "bin"]
                            ).copy()
                            plot_df_clean["bin"] = plot_df_clean["bin"].astype(str)

                            rsrp_col = get_label_col(plot_df_clean)

                            fig3d = px.scatter_3d(
                                plot_df_clean,
                                x="x",
                                y="y",
                                z="z",
                                color="bin",
                                title=f"3D trajectory: Dataset {dataset_id} - {flight_name} (Correlation Bins)",
                                template="plotly_dark",
                                color_discrete_sequence=px.colors.qualitative.Alphabet,
                                hover_data=[rsrp_col],
                            )
                            fig3d.add_trace(
                                go.Scatter3d(
                                    x=[0],
                                    y=[0],
                                    z=[10],
                                    mode="markers",
                                    marker=dict(size=8, color="red", symbol="diamond"),
                                    name="Base Tower",
                                )
                            )

                            bins_to_plot = []
                            seen_bins = set()
                            colors = ["yellow", "cyan", "magenta"]
                            names = [
                                "Highest Correlation",
                                "Greatest Distance 1",
                                "Greatest Distance 2",
                            ]

                            bins_list = [str(top_corr_bin)] + [
                                str(b) for b in top_dist_bins
                            ]

                            for i, (b_val, name_desc) in enumerate(
                                zip(bins_list, names)
                            ):
                                if b_val not in seen_bins:
                                    seen_bins.add(b_val)
                                    bins_to_plot.append(
                                        {
                                            "bin_id": b_val,
                                            "color": colors[i],
                                            "name": f"Bin: {name_desc}",
                                        }
                                    )

                            for b_info in bins_to_plot:
                                b_str = b_info["bin_id"]
                                b = int(float(b_str))

                                b_df = plot_df_clean[plot_df_clean["bin"] == b_str]
                                if b_df.empty:
                                    continue
                                r_max = np.sqrt(
                                    b_df["x"] ** 2
                                    + b_df["y"] ** 2
                                    + (b_df["z"] - 10) ** 2
                                ).max()
                                if pd.isna(r_max) or r_max == 0:
                                    r_max = 100

                                bin_theta = b // num_phi_bins
                                bin_phi = b % num_phi_bins

                                theta_min = bin_theta * BIN_SIZE
                                theta_max = (bin_theta + 1) * BIN_SIZE
                                phi_min = bin_phi * BIN_SIZE
                                phi_max = (bin_phi + 1) * BIN_SIZE

                                rays = [
                                    (theta_min, phi_min),
                                    (theta_min, phi_max),
                                    (theta_max, phi_min),
                                    (theta_max, phi_max),
                                ]

                                for idx, (t, p) in enumerate(rays):
                                    fig3d.add_trace(
                                        go.Scatter3d(
                                            x=[0, r_max * np.sin(t) * np.cos(p)],
                                            y=[0, r_max * np.sin(t) * np.sin(p)],
                                            z=[10, 10 + r_max * np.cos(t)],
                                            mode="lines",
                                            line=dict(color=b_info["color"], width=4),
                                            name=b_info["name"]
                                            if idx == 0
                                            else f"{b_info['name']} line {idx + 1}",
                                            showlegend=(idx == 0),
                                        )
                                    )

                            fig3d.update_layout(
                                scene=dict(
                                    xaxis_title="x (m)",
                                    yaxis_title="y (m)",
                                    zaxis_title="z (m)",
                                ),
                                margin=dict(l=0, r=0, b=0, t=30),
                            )

                            base_name3d = (
                                f"dataset_{dataset_id}_{safe_flight_name}_bin_corr_3d"
                            )
                            png_filename3d = f"{base_name3d}.png"
                            html_filename3d = f"{base_name3d}.html"

                            png_path3d = assets_dir / png_filename3d
                            html_path3d = assets_dir / html_filename3d

                            fig3d.write_html(str(html_path3d))

                            try:
                                fig3d.write_image(str(png_path3d))
                                report.append(
                                    f"#### Flight: {flight_name} 3D Correlation Bins\n"
                                )
                                rel_png3d = f"report_assets/{png_filename3d}"
                                rel_html3d = f"report_assets/{html_filename3d}"
                                report.append(
                                    f"[![{flight_name} 3D]({rel_png3d})]({rel_html3d})\n"
                                )
                            except Exception as e:
                                logger.warning(
                                    f"Could not save static image for {flight_name} 3D: {e}. Ensure 'kaleido' is installed."
                                )
                                rel_html3d = f"report_assets/{html_filename3d}"
                                report.append(
                                    f"#### Flight: {flight_name} 3D Correlation Bins\n"
                                )
                                report.append(f"[Interactive 3D Plot]({rel_html3d})\n")

        elif step_val == StepEnum.FAST_FADING_CORRELATION.value:
            report.append(
                "- Computed fast fading factor correlation in spatio-temporal domain (Section II.B.2) using a rolling window of 5 prior samples."
            )
            report.append(
                "- Fast fading `nu` estimated by removing LoS path loss (proportional to -20log10(d3D)) from RSRP."
            )
            report.append("- Aggregated into spatial separation bins of 3m.")

            report.append("\n### Fast Fading Spatio-Temporal Correlation Profiles\n")

            assets_dir = report_home / "report_assets"
            assets_dir.mkdir(exist_ok=True, parents=True)

            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            for _, r in cast(pd.DataFrame, step_df).iterrows():
                df: pd.DataFrame = r["data"]
                flight_name = r["flight_name"]
                dataset_id = r["dataset_id"]

                if df.empty:
                    report.append(f"#### Flight: {flight_name}\n")
                    report.append("No fast fading correlation data generated.\n")
                    continue

                fig = make_subplots(specs=[[{"secondary_y": True}]])

                fig.add_trace(
                    go.Bar(
                        x=df["spatial_bin"],
                        y=df["num_pairs"],
                        name="Num Points",
                        opacity=0.3,
                        marker_color="green",
                    ),
                    secondary_y=True,
                )

                fig.add_trace(
                    go.Scatter(
                        x=df["spatial_bin"],
                        y=df["r"],
                        name="Avg Correlation",
                        mode="lines+markers",
                        line=dict(color="orange"),
                    ),
                    secondary_y=False,
                )

                fig.update_layout(
                    title_text=f"Fast Fading Correlation: Dataset {dataset_id} - {flight_name}",
                    template="plotly_white",
                )

                fig.update_xaxes(title_text="Spatial separation distance (m)")
                fig.update_yaxes(
                    title_text="Avg Correlation", secondary_y=False, range=[-1.0, 1.0]
                )
                fig.update_yaxes(
                    title_text="Num Points", secondary_y=True, showgrid=False
                )

                safe_flight_name = flight_name.replace(" ", "_").replace("/", "_")
                base_name = f"dataset_{dataset_id}_{safe_flight_name}_ff_correlation"
                png_filename = f"{base_name}.png"
                html_filename = f"{base_name}.html"

                png_path = assets_dir / png_filename
                html_path = assets_dir / html_filename

                fig.write_html(str(html_path))

                try:
                    fig.write_image(str(png_path))
                    report.append(f"#### Flight: {flight_name}\n")
                    rel_png = f"report_assets/{png_filename}"
                    rel_html = f"report_assets/{html_filename}"
                    report.append(f"[![{flight_name}]({rel_png})]({rel_html})\n")
                except Exception as e:
                    logger.warning(
                        f"Could not save static image for {flight_name}: {e}. Ensure 'kaleido' is installed."
                    )
                    rel_html = f"report_assets/{html_filename}"
                    report.append(f"#### Flight: {flight_name}\n")
                    report.append(f"[Interactive Fast Fading Plot]({rel_html})\n")

    with open(report_home / "report.md", "w") as f:
        f.write("\n".join(report))
    logger.info(f"Report generated at {report_home}")
