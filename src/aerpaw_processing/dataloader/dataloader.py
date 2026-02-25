import logging
import pandas as pd
import torch
from torch.utils.data import Dataset
import os
import re
from aerpaw_processing.preprocessing.utils import (
    get_flight_id,
    load_data,
    get_timestamp_col,
)
from aerpaw_processing.resources.config.config_init import (
    load_config,
    TIMESTAMP_PATTERN,
    TIMEDELTA_PATTERN,
)

load_config()


logger = logging.getLogger(__name__)


class SignalDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, dataset_num: int, flight_name: str, label_col: str):

        flight_id = get_flight_id(dataset_num, flight_name) + ".csv"

        clean_dataset_dir = os.getenv("DATASET_CLEAN_HOME")

        if not clean_dataset_dir:
            raise EnvironmentError(
                "Environment variable 'DATASET_CLEAN_HOME' is not set. "
                "Please set it to the path of the cleaned dataset directory."
            )

        data_path = os.path.join(clean_dataset_dir, flight_id)

        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Data path '{data_path}' does not exist. "
                "Run aerpaw_processing.preprocessing.main with desired settings to generate the cleaned dataset."
            )

        data = load_data(data_path)

        label_candidates: list[str] = [
            col
            for col in data.columns
            if col.lower() == label_col.lower()
            or col.lower().startswith(label_col.lower() + "_")
        ]

        if not label_candidates:
            raise ValueError(f"No label column found for '{label_col}'.")

        if len(label_candidates) > 1:
            logging.info(
                f"Multiple label columns found for '{label_col}': {label_candidates}. "
                "Using the first one."
            )

        self.label_col = label_candidates[0]

        self.feature_cols = [col for col in data.columns if col != self.label_col]

        timestamp_col = get_timestamp_col()
        if timestamp_col in self.feature_cols:
            valid_time = str(data[timestamp_col].iloc[data[timestamp_col].first_valid_index()])  # type: ignore
            if re.match(TIMESTAMP_PATTERN, valid_time):
                data[timestamp_col] = pd.to_datetime(
                    data[timestamp_col], format="%Y-%m-%d %H:%M:%S.%f"
                )
                data[timestamp_col] = data[timestamp_col].astype("int64") // 10**9
            elif re.match(TIMEDELTA_PATTERN, valid_time):
                data[timestamp_col] = pd.to_timedelta(data[timestamp_col])
                data[timestamp_col] = data[timestamp_col].dt.total_seconds()
            else:
                raise ValueError(
                    f"Timestamp column '{timestamp_col}' has an unrecognized format: {valid_time}"
                )

        data = data.dropna(subset=[self.label_col] + self.feature_cols)

        self.features = torch.tensor(
            data[self.feature_cols].values, dtype=torch.float32
        )

        self.labels = torch.tensor(data[self.label_col].values, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]
