import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from aerpaw_processing.resources.tower_locations import towers
from aerpaw_processing.preprocessing.preprocess_main import process_datasets
from aerpaw_processing.preprocessing.preprocess_utils import (
    verify_time_loc_cols,
    get_dataset_and_flight_from_id,
    get_label_col,
)
from aerpaw_processing.resources.config.config_init import load_env, CONFIG

load_env()

base_tower = towers[0]

"Dataset_24_Nemo_5G_30m_Flight_1" "Dataset_24_Nemo_5G_30m_Flight_2" "Dataset_24_PawPrints_4G_30m_Flight_1" "Dataset_24_PawPrints_4G_30m_Flight_2" "Dataset_24_PawPrints_4G_50m_Flight" "Dataset_24_PawPrints_5G_30m_Flight_1" "Dataset_24_PawPrints_5G_30m_Flight_2" "Dataset_24_PawPrints_5G_50m_Flight"


def preprocess_paper_optimized(df: pd.DataFrame) -> pd.DataFrame:

    # Make sure Timestamp, Latitude, Longitude, and Altitude columns are present
    verify_time_loc_cols(df)

    # l^{bs} = location of base station, given by {\phi^{bs}, \lambda^{bs}, h^{bs}}
    phi_bs, lambda_bs, h_bs = base_tower.lat, base_tower.lon, base_tower.alt

    # l^{uav} = location of UAV, given by {\phi^{uav}, \lambda^{uav}, h^{uav}}
    phi_uav = df["Latitude"]
    lambda_uav = df["Longitude"]
    h_uav = df["Altitude"]

    # A = 6_378_137 = radius of earth
    A = 6_378_137

    # d_h (l^{bs}, l^{uav}) = A x arccos(sin(\phi^{bs}) x sin(\phi^{uav}) + cos(\phi^{bs}) x cos(\phi^{uav}) x cos(\lambda^{uav} - \lambda^{bs}))
    df["d_h"] = A * np.arccos(
        np.sin(np.radians(phi_bs)) * np.sin(np.radians(phi_uav))
        + np.cos(np.radians(phi_bs))
        * np.cos(np.radians(phi_uav))
        * np.cos(np.radians(lambda_uav - lambda_bs))
    )

    # d_v (l^{bs}, l^{uav}) = |h^{bs} - h^{uav}|
    df["d_v"] = np.abs(h_bs - h_uav)

    # d_{3D} (l^{bs}, l^{uav}) = sqrt(d_h^2 + d_v^2)
    df["d_3D"] = np.sqrt(df["d_h"] ** 2 + df["d_v"] ** 2)

    # Diagram:
    #               |--\  \theta_1     UAV
    #               |   -\          /-- |
    #               |     |   /--       | d_v
    #               |    /--   d_3D     |
    #               |/--                |
    #  receiver ->  |  -----------------
    #               |         d_h
    #              /|\
    #              Base
    df["theta_1"] = np.arctan2(df["d_h"], df["d_v"])

    # Azimuth angle ph_1
    df["phi_1"] = np.arctan2(lambda_uav - lambda_bs, phi_uav - phi_bs)

    # SECTION II B: Spatial Signal Correlation Properties

    # ANGULAR BINNING
    # The 3D environment was partitioned into a spherical grid centered at the BS with elevation and azimuth bins of size 0.05 radians
    bin_size = 0.05
    df["elevation_bin"] = np.floor(df["theta_1"] / bin_size).astype(int)
    df["azimuth_bin"] = np.floor(df["phi_1"] / bin_size).astype(int)

    # CORRELATION COMPUTATION

    # r[m, n] = ((omega_m - omega_bar) * (omega_n - omega_bar)) / sigma_sq
    rsrp_col = get_label_col(df, "RSRP_NR_5G")
    omega_bar = df[rsrp_col].mean()
    sigma_sq = df[rsrp_col].var()

    df["rsrp_norm"] = df[rsrp_col] - omega_bar

    r_mn_list = []
    radial_bin_list = []

    binned_df = df.groupby(["elevation_bin", "azimuth_bin"])

    for _, group_data in binned_df:
        rsrp_norm_arr = group_data["rsrp_norm"].values
        d_3D_arr = group_data["d_3D"].values

        # Calculate (omega_m - omega_bar) * (omega_n - omega_bar) for all pairs instantly
        r_mn_matrix = np.outer(rsrp_norm_arr, rsrp_norm_arr) / sigma_sq  # type: ignore

        # Create an N x N matrix of distance differences
        d_3D_mn_matrix = np.abs(d_3D_arr[:, None] - d_3D_arr[None, :])

        # Convert to 5m bins
        radial_bins_matrix = (d_3D_mn_matrix // 5).astype(int)

        # Flatten the matrices and store them
        r_mn_list.append(r_mn_matrix.ravel())
        radial_bin_list.append(radial_bins_matrix.ravel())

    # Concatenate all flattened arrays into two large 1D arrays
    all_r_mn = np.concatenate(r_mn_list)
    all_radial_bins = np.concatenate(radial_bin_list)

    # Put aggregated results into a temporary DataFrame for rapid grouped calculation
    agg_df = pd.DataFrame({"radial_bin": all_radial_bins, "r_mn": all_r_mn})

    # Use native pandas aggregation instead of the Python dictionary loop
    final_profile_df = (
        agg_df.groupby("radial_bin")
        .agg(
            average_correlation=("r_mn", "mean"),
            number_of_pairs=("r_mn", "size"),  # 'size' counts elements
        )
        .reset_index()
    )

    # Create the distance_range label cleanly via vectorization
    final_profile_df["distance_range"] = (
        (final_profile_df["radial_bin"] * 5).astype(str)
        + "-"
        + (final_profile_df["radial_bin"] * 5 + 5).astype(str)
        + "m"
    )

    # Sort and re-order columns to match original output
    final_profile_df = (
        final_profile_df[
            ["radial_bin", "distance_range", "average_correlation", "number_of_pairs"]
        ]
        .sort_values(by="radial_bin")
        .reset_index(drop=True)
    )

    return final_profile_df


_DATASET_CONFIGS = [
    {
        "key": "nemo",
        "xlabel_suffix": "a) Nemo data",
        "line_color": "#d62728",  # red
        "bar_color": "lightpink",
        "bar_alpha": 0.6,
    },
    {
        "key": "paw_prints",
        "xlabel_suffix": "b) PawPrints data",
        "line_color": "#000080",  # navy
        "bar_color": "lightblue",
        "bar_alpha": 0.5,
    },
    {
        "key": "quectel",
        "xlabel_suffix": "c) Quectel data",
        "line_color": "#2ca02c",  # green
        "bar_color": "lightgreen",
        "bar_alpha": 0.5,
    },
]


def graph_paper(
    nemo_dfs: list[pd.DataFrame],
    paw_prints_dfs: list[pd.DataFrame],
    quectel_dfs: list[pd.DataFrame],
) -> None:
    def _combine_and_profile(dfs: list[pd.DataFrame]) -> pd.DataFrame:
        if not dfs:
            return pd.DataFrame()
        combined = pd.concat(dfs, ignore_index=True)
        if combined.empty:
            return pd.DataFrame()
        return preprocess_paper_optimized(combined)

    profiles = {
        "nemo": _combine_and_profile(nemo_dfs),
        "paw_prints": _combine_and_profile(paw_prints_dfs),
        "quectel": _combine_and_profile(quectel_dfs),
    }

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 12), sharex=False)

    for ax_primary, cfg in zip(axes, _DATASET_CONFIGS):
        profile = profiles[cfg["key"]]

        if profile.empty:
            ax_primary.set_visible(False)
            continue

        ax_secondary = ax_primary.twinx()

        x = profile["radial_bin"] * 5
        y_corr = profile["average_correlation"]
        y_count = profile["number_of_pairs"]

        ax_secondary.bar(
            x,
            y_count,
            width=3.5,
            color=cfg["bar_color"],
            alpha=cfg["bar_alpha"],
            label="Num Points",
        )
        ax_secondary.set_ylim(0, 5000)
        ax_secondary.set_ylabel("Num Points", fontsize=12)

        ax_primary.plot(x, y_corr, color=cfg["line_color"], label="Avg Correlation")
        ax_primary.set_ylim(-1.0, 1.0)
        ax_primary.set_xlim(0, 200)
        ax_primary.set_ylabel("Avg Correlation", fontsize=12)
        ax_primary.set_xlabel(
            f"Radial separation distance (m)\n{cfg['xlabel_suffix']}", fontsize=12
        )
        ax_primary.grid(True, linestyle="--", alpha=0.7)

        lines, labels = ax_primary.get_legend_handles_labels()
        lines2, labels2 = ax_secondary.get_legend_handles_labels()
        ax_primary.legend(
            lines + lines2,
            labels + labels2,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.15),
            ncol=2,
            frameon=False,
        )

    plt.tight_layout()
    plt.show()


def main():
    nemo_flight_ids = [
        "Dataset_24_Nemo_5G_30m_Flight_1",
        "Dataset_24_Nemo_5G_30m_Flight_2",
    ]
    paw_prints_flight_ids = [
        # "Dataset_24_PawPrints_4G_30m_Flight_1",
        # "Dataset_24_PawPrints_4G_30m_Flight_2",
        # "Dataset_24_PawPrints_4G_50m_Flight",
        "Dataset_24_PawPrints_5G_30m_Flight_1",
        "Dataset_24_PawPrints_5G_30m_Flight_2",
        "Dataset_24_PawPrints_5G_50m_Flight",
    ]
    quectel_flight_ids = [
        "Dataset_18_Yaw_45_Flight",
        "Dataset_18_Yaw_315_Flight",
    ]

    data_dict = process_datasets(
        filter_features_bool=True,
        relative_time=False,
        project_coords=False,
        alt_median_abs_deviation=True,
        fill=False,
        save_cleaned_data=False,
        add_spherical=False,
    )

    nemo_dfs: list[pd.DataFrame] = []
    for flight_id in nemo_flight_ids:
        dataset, flight = get_dataset_and_flight_from_id(flight_id)
        nemo_dfs.append(data_dict[dataset][flight])

    paw_prints_dfs: list[pd.DataFrame] = []
    for flight_id in paw_prints_flight_ids:
        dataset, flight = get_dataset_and_flight_from_id(flight_id)
        paw_prints_dfs.append(data_dict[dataset][flight])

    quectel_dfs: list[pd.DataFrame] = []
    for flight_id in quectel_flight_ids:
        dataset, flight = get_dataset_and_flight_from_id(flight_id)
        quectel_dfs.append(data_dict[dataset][flight])

    graph_paper(nemo_dfs, paw_prints_dfs, quectel_dfs)


if __name__ == "__main__":
    main()
