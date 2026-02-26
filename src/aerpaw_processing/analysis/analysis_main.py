import pandas as pd
from pathlib import Path
import os
import logging
from enum import Enum
from aerpaw_processing.preprocessing.preprocess_main import process_datasets
from aerpaw_processing.preprocessing.preprocess_utils import (
    get_flight_id,
)
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

    def __init__(self):
        data_dict = process_datasets(
            save_cleaned_data=False, relative_time=True, project_coords=True
        )
        self.flight_dict = {}
        for dataset_num, flights in data_dict.items():
            for flight_name, flight_data in flights.items():
                flight_id = get_flight_id(dataset_num, flight_name)
                self.flight_dict[flight_id] = flight_data

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
            if characteristic == FlightCharacteristic.TIMESTAMP_MEAN_STD:
                df["timestamp_mean_std_s"] = get_timestamp_mean_std(self.flight_dict)
            elif characteristic == FlightCharacteristic.DISTANCE_MEAN_STD:
                df["distance_mean_std"] = get_distance_mean_std(self.flight_dict)

        return df


def main():
    details = DatasetFlightDetails()
    characteristics = [
        FlightCharacteristic.COLUMNS,
        FlightCharacteristic.NUM_ROWS,
        FlightCharacteristic.TIMESTAMP_MEAN_STD,
        FlightCharacteristic.DISTANCE_MEAN_STD,
    ]
    df = details.get_characteristics(characteristics)

    logger.info("Analysis output:\n%s", df)

    current_dir = Path(__file__).resolve().parent
    output_dir = os.path.join(current_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "analysis.csv"), index=False)

    logger.info("Analysis results saved to %s", output_dir)

    return df


if __name__ == "__main__":
    main()
