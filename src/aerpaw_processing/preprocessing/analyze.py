import pandas as pd
from tabulate import tabulate
from aerpaw_processing.preprocessing.main import process_datasets
from aerpaw_processing.preprocessing.utils import (
    get_flight_id,
    remove_duplicate_timestamps,
    get_timestamp_col,
)
from aerpaw_processing.resources.config.config_init import CONFIG, load_env

load_env()


def analyze():
    data_dict = process_datasets(
        save_cleaned_data=False, relative_time=True, project_coords=True
    )

    all_flight_details: list[dict[str, str]] = []

    for dataset in CONFIG.datasets:
        for flight in dataset.flights:
            flight_details: dict[str, str] = {
                "flight_id": get_flight_id(dataset.num, flight.name),
            }

            flight_data = data_dict[dataset.num][flight.name]

            flight_details["columns"] = ", ".join(flight_data.columns)
            flight_details["num_rows"] = str(len(flight_data))

            flight_data = remove_duplicate_timestamps(flight_data)

            timestamp_diffs: pd.Series[pd.Timedelta] = flight_data[
                get_timestamp_col()
            ].diff()
            timestamp_mean = timestamp_diffs.mean().total_seconds() * 1_000_000
            timestamp_std = timestamp_diffs.std().total_seconds() * 1_000_000

            flight_details["timestamp_mean_std_ns"] = (
                f"{timestamp_mean:.2f} ± {timestamp_std:.2f}"
            )

            distance_x = flight_data["x"].diff()
            distance_y = flight_data["y"].diff()
            distance_z = flight_data["z"].diff()

            distance = (distance_x**2 + distance_y**2 + distance_z**2) ** 0.5

            distance_mean = distance.mean()
            distance_std = distance.std()

            flight_details["distance_mean_std"] = (
                f"{distance_mean:.2f} ± {distance_std:.2f}"
            )

            all_flight_details.append(flight_details)

    df = pd.DataFrame(all_flight_details)

    print(tabulate(df, headers="keys", tablefmt="grid", showindex=False))  # type: ignore


if __name__ == "__main__":
    analyze()
