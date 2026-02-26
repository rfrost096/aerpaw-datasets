import argparse
import pandas as pd
from pathlib import Path
import os
import logging
from enum import Enum
from aerpaw_processing.preprocessing.preprocess_main import process_datasets
from aerpaw_processing.preprocessing.preprocess_utils import get_flight_id
from aerpaw_processing.analysis.analysis_utils import (
    get_columns,
    get_num_rows,
    get_timestamp_mean_std,
    get_distance_mean_std,
)
from aerpaw_processing.resources.config.config_init import load_env

load_env()

logger = logging.getLogger(__name__)


class FlightCharacteristic(Enum):
    COLUMNS = "columns"
    UNIQUE_COLUMNS = "unique_columns"
    NUM_ROWS = "num_rows"
    TIMESTAMP_MEAN_STD = "timestamp_mean_std_s"
    DISTANCE_MEAN_STD = "distance_mean_std"


class DatasetFlightDetails:
    flight_dict: dict[str, pd.DataFrame]

    def __init__(
        self,
        save_cleaned_data: bool = False,
        relative_time: bool = True,
        project_coords: bool = True,
        alt_median_deviation: bool = False,
        fill: bool = True,
    ):
        data_dict = process_datasets(
            save_cleaned_data=save_cleaned_data,
            relative_time=relative_time,
            project_coords=project_coords,
            alt_median_abs_deviation=alt_median_deviation,
            fill=fill,
        )
        self.flight_dict = {
            get_flight_id(dataset_num, flight_name): flight_data
            for dataset_num, flights in data_dict.items()
            for flight_name, flight_data in flights.items()
        }

    def get_characteristics(
        self, characteristics: list[FlightCharacteristic]
    ) -> pd.DataFrame:
        df = pd.DataFrame({"flight_id": list(self.flight_dict.keys())})

        for characteristic in characteristics:
            if characteristic == FlightCharacteristic.COLUMNS:
                df["columns"] = get_columns(self.flight_dict)
            elif characteristic == FlightCharacteristic.UNIQUE_COLUMNS:
                df["unique_columns"] = get_columns(self.flight_dict, unique=True)
            elif characteristic == FlightCharacteristic.NUM_ROWS:
                df["num_rows"] = get_num_rows(self.flight_dict)
            elif characteristic == FlightCharacteristic.TIMESTAMP_MEAN_STD:
                df["timestamp_mean_std_s"] = get_timestamp_mean_std(self.flight_dict)
            elif characteristic == FlightCharacteristic.DISTANCE_MEAN_STD:
                df["distance_mean_std"] = get_distance_mean_std(self.flight_dict)

        return df


def analyze() -> pd.DataFrame | None:
    valid_values = ", ".join(f.value for f in FlightCharacteristic)

    parser = argparse.ArgumentParser(
        description="Analyze flight dataset characteristics.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "characteristics",
        type=str,
        help=(
            "Comma-separated list of characteristics to compute.\n"
            f"Valid values: {valid_values}\n"
            "Example: columns,num_rows,timestamp_mean_std_s"
        ),
    )
    parser.add_argument(
        "--alt-median-deviation",
        action="store_true",
        default=False,
        help="Use median absolute deviation for altitude filtering (default: False).",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=False,
        help="Save the results to a CSV file in the output directory.",
    )

    args = parser.parse_args()

    valid_map = {f.value: f for f in FlightCharacteristic}
    raw_values = [v.strip() for v in args.characteristics.split(",")]

    invalid = [v for v in raw_values if v not in valid_map]
    if invalid:
        parser.error(
            f"Invalid characteristic(s): {', '.join(invalid)}. "
            f"Valid values are: {valid_values}"
        )
        return None

    characteristics = [valid_map[v] for v in raw_values]
    if not characteristics:
        parser.error("No valid characteristics provided.")
        return None

    details = DatasetFlightDetails(alt_median_deviation=args.alt_median_deviation)
    result_df = details.get_characteristics(characteristics)

    logger.info("Analysis output:\n%s", result_df)

    if args.save:
        characteristic_slug = "_".join(c.value for c in characteristics)
        filename = f"analysis_{characteristic_slug}.csv"
        output_dir = os.path.join(Path(__file__).resolve().parent, "output")
        os.makedirs(output_dir, exist_ok=True)
        result_df.to_csv(os.path.join(output_dir, filename), index=False)
        logger.info("Analysis results saved to %s/%s", output_dir, filename)

    return result_df
