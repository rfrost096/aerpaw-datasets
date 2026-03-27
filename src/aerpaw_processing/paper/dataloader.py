import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from aerpaw_processing.paper.preprocess_steps import process
from aerpaw_processing.paper.preprocess_utils import DatasetConfig, get_env_var
from aerpaw_processing.resources.config.config_init import load_env

load_env()


class UAVFlightDataset(Dataset):
    def __init__(
        self,
        config: DatasetConfig,
        target: str,
        inception_dim: int = 1,
        flag_all_features: bool = True,
        dataset_filenames: list[str] | None = None,
    ):
        """
        Args:
            config: Dataset configuration
            target (str): The column name to predict (e.g., 'RSRP_LTE_4G' or 'RSRP_NR_5G').
                          If False, drop datasets missing the target.
            inception_dim: Dimensionality for inception model (1, 2, or 3)
            flag_all_features (bool): If True, keep inconsistent features and add flag columns
                                      If False, keep only features present in ALL datasets.
        """
        clean_home = Path(str(get_env_var("DATASET_CLEAN_HOME")))
        dataset_dir = clean_home / "data" / config.get_id()
        self.dataset_dir = dataset_dir
        self.target = target
        self.inception_dim = inception_dim
        self.flag_all_features = flag_all_features
        self.dataset_filenames = dataset_filenames

        # Check if the processed datasets already exist. If they are missing,
        # generate them using the specified config.
        if not self.dataset_dir.exists() or not any(self.dataset_dir.iterdir()):
            process(config)

        # Get a list of csv files. Default to all files if no specific filenames
        # were defined.
        if self.dataset_filenames is None:
            csv_files = list(dataset_dir.glob("*.csv"))
        else:
            csv_files = [
                self.dataset_dir / filename for filename in self.dataset_filenames
            ]

        raw_dfs = [pd.read_csv(f) for f in csv_files]

        processed_dfs = []

        for df in raw_dfs:
            # All datasets have these columns. Every dataset has x, y, z columns based
            # on the same base tower. Every dataset has RelativeTime column based on
            # the first measurement in each dataset. These columns are now redundant.
            df = df.drop(
                columns=["Timestamp", "Latitude", "Longitude", "Altitude"],
                axis=0,
                errors="ignore",
            )

            # Convert pandas RelativeTime timedelta column to float value for processing.
            if "RelativeTime" in df.columns:
                df["RelativeTime"] = pd.to_timedelta(
                    df["RelativeTime"]
                ).dt.total_seconds()
                df = df.sort_values(by="RelativeTime")

            # Some datasets have RSRP_LTE_4G and some datasets have RSRP_NR_5G. We only
            # want datasets with the specific RSRP type.
            if self.target not in df.columns:
                continue

            processed_dfs.append(df)

        if not processed_dfs:
            raise ValueError(
                "No datasets remaining after applying target filtering logic."
            )

        # Get a set of all columns across all datasets and
        # a set of columns that are common between all datasets.
        all_cols = set()
        common_cols = set(processed_dfs[0].columns)

        for df in processed_dfs:
            all_cols.update(df.columns)
            common_cols.intersection_update(df.columns)

        # Remove target column to get just the feature columns
        feature_cols_all = all_cols - {self.target}
        feature_cols_common = common_cols - {self.target}

        final_feature_cols: list[str]
        if self.flag_all_features:
            # If this value is true, we want to keep feature data that may only be
            # present on some of the datasets. To train with some of the data missing,
            # we add a "flag" column that is set to 1 if the column is present in that
            # dataset or 0 if the column is missing. This way, the model will learn
            # that if a column is missing, it's actual value (0.0) can be ignored.
            final_feature_cols = sorted(list(feature_cols_all))
            flag_cols = [
                f"{col}_present" for col in feature_cols_all - feature_cols_common
            ]
            final_feature_cols = sorted(final_feature_cols + flag_cols)
        else:
            # If the value is false, just remove columns that are not common between
            # all datasets
            final_feature_cols = sorted(list(feature_cols_common))

        all_X_inception = []
        all_X_los = []
        all_y_targets = []

        for df in processed_dfs:
            if self.flag_all_features:
                # Set flag value to 1 for present and 0 for missing
                features_to_flag = feature_cols_all - feature_cols_common
                for col in features_to_flag:
                    if col in df.columns:
                        df[f"{col}_present"] = 1
                    else:
                        df[col] = 0.0
                        df[f"{col}_present"] = 0
            else:
                cols_to_keep = list(feature_cols_common | {self.target})
                df = df[cols_to_keep]

            df.fillna(0.0, inplace=True)

            # Use sorted final_feature_cols to match up columns between datasets.
            flight_X = df[final_feature_cols].values
            flight_y = df[self.target].values

            # Compute 3D distance to the tower for Line of Sight Path Loss model
            # Base station is at x=0, y=0, z=10
            x_idx = final_feature_cols.index("x") if "x" in final_feature_cols else -1
            y_idx = final_feature_cols.index("y") if "y" in final_feature_cols else -1
            z_idx = final_feature_cols.index("z") if "z" in final_feature_cols else -1

            for i in range(len(flight_X)):
                point = flight_X[i]
                target_val = flight_y[i]

                if x_idx != -1 and y_idx != -1 and z_idx != -1:
                    x_val, y_val, z_val = point[x_idx], point[y_idx], point[z_idx]
                    d3D = np.sqrt(x_val**2 + y_val**2 + (z_val - 10) ** 2)
                    log_d = np.log10(max(d3D, 1e-5))
                else:
                    log_d = 0.0  # Fallback if spatial features are missing

                # Reshape point feature vector for 1D, 2D, or 3D CNN
                if self.inception_dim == 1:
                    point_t = point.reshape(-1, 1)
                elif self.inception_dim == 2:
                    point_t = point.reshape(-1, 1, 1)
                elif self.inception_dim == 3:
                    point_t = point.reshape(-1, 1, 1, 1)
                else:
                    raise ValueError("inception_dim must be 1, 2, or 3")

                all_X_inception.append(point_t)
                all_X_los.append([log_d])
                all_y_targets.append(target_val)

        self.X_inception = torch.tensor(np.array(all_X_inception), dtype=torch.float32)
        self.X_los = torch.tensor(np.array(all_X_los), dtype=torch.float32)
        self.y = torch.tensor(np.array(all_y_targets), dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_inception[idx], self.X_los[idx], self.y[idx]
