import pandas as pd
import math
from torch.utils.data import Dataset
import torch
from pathlib import Path
from aerpaw_processing.paper.preprocess_steps import process
from aerpaw_processing.paper.preprocess_utils import DatasetConfig, get_env_var
from aerpaw_processing.resources.config.config_init import load_env

load_env()

INCEPTION_COLUMNS = ["d3D", "elevation", "azimuth"]
TARGET = "RSRP_NR_5G"
CENTER_FREQUENCY = 3.4 * 1e9
# https://sites.google.com/ncsu.edu/aerpaw-user-manual/6-sample-experiments-repository/6-1-radio-software/6-1-6-ericsson-experiments
PORTS = 2 * 2  # 2x2 MIMO
WATTS_PER_PORT = 5  # 5 watts per antenna port
WATTS_TO_MILLIWATTS = 1e3
# 100 MHz channel bandwidth (according to paper)
PRB_S = 273  # Assuming 273 Physical Resource Blocks (PRBs) based on channel bandwidth
RESOURCE_ELEMENTS = 12
P_TX = 10 * math.log10(
    ((PORTS * WATTS_PER_PORT) * WATTS_TO_MILLIWATTS) / (PRB_S * RESOURCE_ELEMENTS)
)
R_MAX = 500


class InceptionDataset(Dataset):
    def __init__(
        self,
        config: DatasetConfig,
        target: str = TARGET,
        dataset_filenames: list[str] | None = None,
    ):
        """
        Args:
            config: Dataset configuration
            target (str): The column name to predict (e.g., 'RSRP_LTE_4G' or 'RSRP_NR_5G').
                          If False, drop datasets missing the target.
        """
        clean_home = Path(str(get_env_var("DATASET_CLEAN_HOME")))
        dataset_dir = clean_home / "data" / config.get_id()
        self.dataset_dir = dataset_dir
        self.target = target
        self.dataset_filenames = dataset_filenames
        self.r_max = R_MAX
        self.tx_power = P_TX
        self.fc = CENTER_FREQUENCY

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
            if len(set(INCEPTION_COLUMNS).intersection(set(df.columns))) != len(
                INCEPTION_COLUMNS
            ):
                missing_cols: list[str] = []
                for col in INCEPTION_COLUMNS:
                    if col not in df.columns:
                        missing_cols.append(col)
                raise ValueError(
                    f"Columns are missing for Inception features: {', '.join(missing_cols)}"
                )

            if self.target not in df.columns:
                continue

            cols_to_keep = [target] + INCEPTION_COLUMNS

            df = df[cols_to_keep]

            processed_dfs.append(df)

        if not processed_dfs:
            raise ValueError(
                "No datasets remaining after applying target filtering logic."
            )

        self.data = pd.concat(processed_dfs, ignore_index=True)
        self.theta = torch.tensor(self.data["elevation"].values, dtype=torch.float32)
        self.phi = torch.tensor(self.data["azimuth"].values, dtype=torch.float32)
        self.d3d = torch.tensor(self.data["d3D"].values, dtype=torch.float32)
        self.targets = torch.tensor(self.data[self.target].values, dtype=torch.float32)

        self.d_steps = torch.arange(1, self.r_max + 1, dtype=torch.float32)
        self.log10_d_steps = torch.log10(self.d_steps)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        theta = self.theta[idx]
        phi = self.phi[idx]
        d3d_val = self.d3d[idx]
        y = self.targets[idx]

        p_tx_seq = torch.full((self.r_max,), self.tx_power, dtype=torch.float32)
        log10_fc_seq = torch.full(
            (self.r_max,), math.log10(self.fc), dtype=torch.float32
        )

        log10_d = self.log10_d_steps

        theta_seq = torch.full((self.r_max,), theta.item(), dtype=torch.float32)
        phi_seq = torch.full((self.r_max,), phi.item(), dtype=torch.float32)

        x_seq = self.d_steps * torch.sin(theta) * torch.cos(phi)
        y_seq = self.d_steps * torch.sin(theta) * torch.sin(phi)
        z_seq = self.d_steps * torch.cos(theta)

        x = torch.stack(
            [p_tx_seq, log10_fc_seq, log10_d, theta_seq, phi_seq, x_seq, y_seq, z_seq],
            dim=0,
        )

        j_index = torch.argmin(torch.abs(self.d_steps - d3d_val))
        mask = torch.zeros(self.r_max, dtype=torch.bool)
        mask[j_index] = True
        y_seq = torch.zeros(self.r_max, dtype=torch.float32)
        y_seq[j_index] = y

        return x, y_seq, mask
