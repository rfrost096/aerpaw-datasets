"""
graph_figure4.py
----------------
Reproduces Figure 4 from:
  "AI-Enabled Wireless Propagation Modeling and Radio Environment Maps
   for 5G Aerial Wireless Networks"

Figure 4 shows the pairwise spatial correlation of the fast-fading factor
(ν) obtained from a rolling window of five samples, binned by spatial
separation distance (3 m bins, 0–50 m range).

Paper reference – Section II-B-2:
  r[i, m] = (ν_i)(ν_{i-m}) / σ²_ν   for m ∈ {1, 2, 3, 4, 5}
  grouped by d3D(l^uav_i, l^uav_{i-m}) with a bin step of 3 m.

The fast-fading factor is obtained by removing the large-scale LoS path
loss from the measured RSRP:
  ν_i = Ω_i − Ω̂_LoS(d3D_i)

The large-scale LoS component is estimated via a two-parameter OLS fit:
    Ω̂ = a + b·log10(d3D)
Both slope AND intercept are free, so the residuals are properly
zero-mean, irrespective of P_TX_SS or site-specific antenna gain offsets.

Changes vs. v1
--------------
1. Two-parameter OLS (free slope + intercept) replaces the fixed slope=−20
   fit.  The fixed slope left a systematic trend in ν, causing correlations
   to remain artificially high across all separations.
2. RSRP column resolution now:
     (a) tries aerpaw_processing.preprocess_utils.get_label_col first,
     (b) falls back to the RSRP-like column with the most non-NaN values,
     (c) logs clearly which column was chosen — making the Nemo missing-
         line bug visible immediately.
3. NaN rows are dropped before any computation.
4. Y-axis fixed to [−1, 1] to match the paper's Figure 4.
5. Per-dataset rsrp_col overrides added to graph_figure4() signature.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_rsrp_col(df: pd.DataFrame, rsrp_col: Optional[str], label: str) -> str:
    """
    Return the best RSRP column name for `df`.

    Priority:
    1. Caller-supplied `rsrp_col` (validated).
    2. aerpaw_processing.preprocess_utils.get_label_col, if importable.
    3. The RSRP-like column with the most finite non-NaN numeric values.
    """
    if rsrp_col is not None:
        if rsrp_col not in df.columns:
            raise KeyError(
                f"[{label}] Requested rsrp_col '{rsrp_col}' not in df. "
                f"Available: {list(df.columns)}"
            )
        return rsrp_col

    # Try the project utility first
    try:
        from aerpaw_processing.preprocessing.preprocess_utils import get_label_col

        col = get_label_col(df, "RSRP_NR_5G")
        print(f"[{label}] RSRP column (via get_label_col): '{col}'")
        return col
    except Exception:
        pass

    # Fall back: all RSRP-like columns; pick the one with the most numeric data
    candidates = [c for c in df.columns if "RSRP" in c.upper()]
    if not candidates:
        raise ValueError(
            f"[{label}] No RSRP column found. " f"Available columns: {list(df.columns)}"
        )

    best_col = max(
        candidates, key=lambda c: pd.to_numeric(df[c], errors="coerce").notna().sum()
    )
    n_valid = pd.to_numeric(df[best_col], errors="coerce").notna().sum()
    print(
        f"[{label}] Auto-selected RSRP column: '{best_col}' "
        f"({n_valid} valid rows). Candidates: {candidates}"
    )
    return best_col


def _compute_d3D_bs(df: pd.DataFrame, tower) -> np.ndarray:
    """3-D UAV ↔ base-station separation distance (metres), per Eq. 1."""
    A = 6_378_137  # Earth radius (m)
    phi_bs = np.radians(tower.lat)
    phi_uav = np.radians(df["Latitude"].values)
    lam_diff = np.radians(df["Longitude"].values - tower.lon)

    d_h = A * np.arccos(
        np.clip(
            np.sin(phi_bs) * np.sin(phi_uav)
            + np.cos(phi_bs) * np.cos(phi_uav) * np.cos(lam_diff),
            -1.0,
            1.0,
        )
    )
    d_v = np.abs(tower.alt - df["Altitude"].values)
    return np.sqrt(d_h**2 + d_v**2)


def _fit_los_component(rsrp: np.ndarray, d3d: np.ndarray) -> np.ndarray:
    """
    Two-parameter OLS log-distance fit:
        Ω̂ = a + b · log10(d3D)

    Both the intercept `a` (absorbs P_TX_SS, antenna gains) and slope `b`
    (path-loss exponent × 10) are estimated from the data, ensuring the
    residual ν = Ω − Ω̂ is exactly zero-mean and trend-free.
    """
    log_d = np.log10(np.maximum(d3d, 1e-3))
    X = np.column_stack([np.ones_like(log_d), log_d])
    coeffs, _, _, _ = np.linalg.lstsq(X, rsrp, rcond=None)
    a, b = coeffs
    return a + b * log_d


def _uav_separation(
    lat_rad: np.ndarray,
    lon_rad: np.ndarray,
    alt: np.ndarray,
    m: int,
) -> np.ndarray:
    """
    Haversine distance between UAV positions i and i−m (metres).
    Returns array of length len − m.
    """
    A = 6_378_137
    n = len(lat_rad) - m
    lam_diff = lon_rad[m:] - lon_rad[:n]

    d_h = A * np.arccos(
        np.clip(
            np.sin(lat_rad[m:]) * np.sin(lat_rad[:n])
            + np.cos(lat_rad[m:]) * np.cos(lat_rad[:n]) * np.cos(lam_diff),
            -1.0,
            1.0,
        )
    )
    d_v = np.abs(alt[m:] - alt[:n])
    return np.sqrt(d_h**2 + d_v**2)


# ---------------------------------------------------------------------------
# Core correlation function
# ---------------------------------------------------------------------------


def compute_fast_fading_correlation(
    df: pd.DataFrame,
    tower,
    rsrp_col: Optional[str] = None,
    label: str = "dataset",
    window: int = 5,
    bin_size_m: float = 3.0,
    max_sep_m: float = 50.0,
) -> pd.DataFrame:
    """
    Compute the pairwise spatial correlation of the fast-fading factor
    following Section II-B-2 of the paper.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain Latitude, Longitude, Altitude, and an RSRP column.
    tower : object
        Base-station location with .lat, .lon, .alt attributes.
    rsrp_col : str, optional
        RSRP column name; auto-detected if None.
    label : str
        Name shown in diagnostic prints.
    window : int
        Rolling window size (default 5, per the paper).
    bin_size_m : float
        Spatial separation bin width in metres (default 3 m).
    max_sep_m : float
        Maximum UAV-to-UAV separation to include (default 50 m).

    Returns
    -------
    pd.DataFrame
        Columns: separation_m (bin centre), avg_correlation, num_pairs.
    """
    df = df.copy().reset_index(drop=True)

    # ---- resolve RSRP column -----------------------------------------------
    col = _resolve_rsrp_col(df, rsrp_col, label)
    df[col] = pd.to_numeric(df[col], errors="coerce")

    # ---- drop NaN rows in required columns ---------------------------------
    required = ["Latitude", "Longitude", "Altitude", col]
    n_before = len(df)
    df = df.dropna(subset=required).reset_index(drop=True)
    n_after = len(df)
    if n_before != n_after:
        print(f"[{label}] Dropped {n_before - n_after} NaN rows; {n_after} remain.")
    if n_after < window + 2:
        raise ValueError(
            f"[{label}] Only {n_after} valid rows after NaN drop — "
            f"need at least {window + 2}."
        )

    rsrp = df[col].values.astype(float)
    d3d = _compute_d3D_bs(df, tower)

    # ---- fast-fading factor  ν = Ω − Ω̂_LoS  (should be ≈ zero-mean) -----
    los_hat = _fit_los_component(rsrp, d3d)
    nu = rsrp - los_hat
    sigma2_nu = np.var(nu, ddof=0)
    if sigma2_nu < 1e-12:
        raise ValueError(f"[{label}] Fast-fading variance ≈ 0; cannot normalise.")

    print(
        f"[{label}] ν — mean={nu.mean():.4f} dB, " f"std={nu.std():.4f} dB, N={n_after}"
    )

    # ---- UAV coordinate arrays for inter-sample distances ------------------
    lat_rad = np.radians(df["Latitude"].values)
    lon_rad = np.radians(df["Longitude"].values)
    alt = df["Altitude"].values

    # ---- sliding window  r[i,m] = (ν_i · ν_{i-m}) / σ²_ν  ----------------
    r_list = []
    sep_list = []

    for m in range(1, window + 1):
        n = len(df) - m
        r_im = (nu[m:] * nu[:n]) / sigma2_nu
        sep_m = _uav_separation(lat_rad, lon_rad, alt, m)
        r_list.append(r_im)
        sep_list.append(sep_m)

    all_r = np.concatenate(r_list)
    all_sep = np.concatenate(sep_list)

    # ---- filter to max_sep_m and bin ---------------------------------------
    mask = all_sep <= max_sep_m
    all_r = all_r[mask]
    all_sep = all_sep[mask]

    if len(all_r) == 0:
        warnings.warn(f"[{label}] No pairs within {max_sep_m} m — check flight speed.")
        return pd.DataFrame(columns=["separation_m", "avg_correlation", "num_pairs"])

    bin_idx = (all_sep // bin_size_m).astype(int)

    agg = (
        pd.DataFrame({"bin": bin_idx, "r": all_r})
        .groupby("bin", sort=True)
        .agg(avg_correlation=("r", "mean"), num_pairs=("r", "size"))
        .reset_index()
    )
    agg["separation_m"] = agg["bin"] * bin_size_m + bin_size_m / 2.0  # bin centre

    return agg[["separation_m", "avg_correlation", "num_pairs"]]


# ---------------------------------------------------------------------------
# Plotting function
# ---------------------------------------------------------------------------


def graph_figure4(
    nemo_dfs: list,
    paw_prints_dfs: list,
    quectel_dfs: list,
    tower,
    rsrp_col_nemo: Optional[str] = None,
    rsrp_col_paw_prints: Optional[str] = None,
    rsrp_col_quectel: Optional[str] = None,
    window: int = 5,
    bin_size_m: float = 3.0,
    max_sep_m: float = 50.0,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Reproduce Figure 4 from the paper.

    Parameters
    ----------
    nemo_dfs, paw_prints_dfs, quectel_dfs : list of pd.DataFrame
        Measurement dataframes per device; multiple flights are concatenated.
    tower : object
        Base-station location with .lat, .lon, .alt attributes.
    rsrp_col_nemo, rsrp_col_paw_prints, rsrp_col_quectel : str, optional
        Explicit RSRP column names per device; auto-detected if None.
    window : int
        Rolling window size (default 5).
    bin_size_m : float
        Spatial bin width in metres (default 3 m).
    max_sep_m : float
        X-axis upper limit in metres (default 50 m).
    save_path : str, optional
        If given, the figure is saved here.

    Returns
    -------
    matplotlib.figure.Figure
    """
    dataset_configs = [
        {
            "label": "Nemo data",
            "dfs": nemo_dfs,
            "rsrp_col": rsrp_col_nemo,
            "color": "#d62728",  # red  – matches paper
            "marker": "x",
            "ms": 7,
        },
        {
            "label": "PawPrints data",
            "dfs": paw_prints_dfs,
            "rsrp_col": rsrp_col_paw_prints,
            "color": "#1f77b4",  # blue
            "marker": "D",
            "ms": 5,
        },
        {
            "label": "Quectel data",
            "dfs": quectel_dfs,
            "rsrp_col": rsrp_col_quectel,
            "color": "#2ca02c",  # green
            "marker": "s",
            "ms": 5,
        },
    ]

    fig, ax = plt.subplots(figsize=(7, 4))

    for cfg in dataset_configs:
        valid_dfs = [d for d in cfg["dfs"] if d is not None and not d.empty]
        if not valid_dfs:
            print(f"[graph_figure4] No data for '{cfg['label']}', skipping.")
            continue

        combined = pd.concat(valid_dfs, ignore_index=True)

        try:
            profile = compute_fast_fading_correlation(
                combined,
                tower,
                rsrp_col=cfg["rsrp_col"],
                label=cfg["label"],
                window=window,
                bin_size_m=bin_size_m,
                max_sep_m=max_sep_m,
            )
        except Exception as exc:
            print(f"[graph_figure4] ERROR for '{cfg['label']}': {exc}")
            continue

        if profile.empty:
            print(f"[graph_figure4] Empty profile for '{cfg['label']}', skipping.")
            continue

        ax.plot(
            profile["separation_m"],
            profile["avg_correlation"],
            color=cfg["color"],
            marker=cfg["marker"],
            markersize=cfg["ms"],
            linewidth=1.8,
            label=cfg["label"],
        )

    # ---- axis formatting to match paper Figure 4 ---------------------------
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlim(0, max_sep_m)
    ax.set_ylim(-1.0, 1.0)  # matches paper y-range
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_xlabel("Spatial separation distance (m)", fontsize=12)
    ax.set_ylabel("Avg Correlation", fontsize=12)
    ax.set_title(
        "Fig. 4 – Fast-fading factor correlation\n"
        "(rolling window of 5 samples, spatial separation bins of 3 m)",
        fontsize=11,
    )
    ax.legend(frameon=False, fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Stand-alone entry point — mirrors preprocess_paper.py's main()
# ---------------------------------------------------------------------------


def main():
    from aerpaw_processing.resources.tower_locations import towers
    from aerpaw_processing.preprocessing.preprocess_main import process_datasets
    from aerpaw_processing.preprocessing.preprocess_utils import (
        get_dataset_and_flight_from_id,
    )
    from aerpaw_processing.resources.config.config_init import load_env

    load_env()
    base_tower = towers[0]

    nemo_flight_ids = [
        "Dataset_24_Nemo_5G_30m_Flight_1",
        "Dataset_24_Nemo_5G_30m_Flight_2",
    ]
    paw_prints_flight_ids = [
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

    def _load(flight_ids):
        dfs = []
        for fid in flight_ids:
            dataset, flight = get_dataset_and_flight_from_id(fid)
            dfs.append(data_dict[dataset][flight])
        return dfs

    fig = graph_figure4(
        nemo_dfs=_load(nemo_flight_ids),
        paw_prints_dfs=_load(paw_prints_flight_ids),
        quectel_dfs=_load(quectel_flight_ids),
        tower=base_tower,
        save_path="figure4.png",
    )
    plt.show()


if __name__ == "__main__":
    main()
