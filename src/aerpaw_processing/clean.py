"""Clean AERPAW datasets by removing unimportant features."""

import pandas as pd
from pyproj import Proj
import os
from aerpaw_processing.tower_locations import towers
from aerpaw_processing.details import get_all_flight_details
from aerpaw_processing.resources.config.config_init import CONFIG, load_env

load_env()

REQUESTED_FEATURES = [
    "Timestamp",
    "Latitude",
    "Longitude",
    "Altitude",
    "RSRP",
    "RSRQ",
    "SINR",
]


def clean_datasets(project_coords: bool = False, relative_time: bool = False):
    flights = get_all_flight_details()
    if flights is None:
        print("No flight details found.")
        return
    flights.load_all_analysis_data()

    for dataset in CONFIG.datasets:
        for flight_name, flight in flights.flights[dataset.num].items():
            df = flight.analysis_data

            ordered_cols: list[str] = []

            for col in df.columns:
                for requested in REQUESTED_FEATURES:
                    if requested in col:
                        ordered_cols.append(col)
                        break
            df = df[ordered_cols].copy()

            signal_cols = [
                c
                for c in df.columns
                if any(req in c for req in ["RSRP", "RSRQ", "SINR"])
            ]

            if signal_cols:
                df = df.dropna(subset=signal_cols, how="all")

            if relative_time:
                timestamp_col = next((c for c in df.columns if "Timestamp" in c), None)
                if timestamp_col and not df.empty:
                    dt_series: pd.Series[pd.Timestamp] = df[timestamp_col]

                    rel_timedelta = dt_series - dt_series.iloc[0]

                    tot_sec = rel_timedelta.dt.total_seconds()
                    mins = (tot_sec // 60).astype(int).astype(str).str.zfill(2)
                    secs = (tot_sec % 60).astype(int).astype(str).str.zfill(2)
                    milli = (
                        (rel_timedelta.dt.microseconds // 1000).astype(str).str.zfill(3)
                    )

                    df["Relative_Time"] = mins + ":" + secs + "." + milli
                    ordered_cols.insert(1, "Relative_Time")
                    ordered_cols.remove(timestamp_col)
                    df.drop(columns=[timestamp_col], inplace=True)
                    df = df[ordered_cols]

            if project_coords:
                base_tower = towers[0]

                local_proj = Proj(
                    proj="aeqd",
                    lat_0=base_tower.lat,
                    lon_0=base_tower.lon,
                    datum="WGS84",
                )

                df["x"], df["y"] = local_proj(
                    df["Longitude"].values, df["Latitude"].values
                )

                df["z"] = df["Altitude"]

                df[["x", "y", "z"]] = df[["x", "y", "z"]].round(3)

                final_order = []
                for col in ordered_cols:
                    if col == "Latitude":
                        final_order.append("y")
                    elif col == "Longitude":
                        final_order.append("x")
                    elif col == "Altitude":
                        final_order.append("z")
                    else:
                        final_order.append(col)

                df = df[final_order]

            output_path = str(os.path.join(os.getenv("DATASET_CLEAN_HOME"), f"dataset_{dataset.num}_{flight_name}.csv"))  # type: ignore
            df.to_csv(output_path, index=False)


def main():
    clean_datasets(project_coords=True, relative_time=True)


if __name__ == "__main__":
    main()
