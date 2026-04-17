import pandas as pd
import math
import torch
from torch.utils.data import Dataset, Subset, random_split
from pathlib import Path
from aerpaw_processing.paper.preprocess_steps import process
from aerpaw_processing.paper.preprocess_utils import DatasetConfig, get_env_var
from aerpaw_processing.resources.config.config_init import load_env

load_env()

FEATURE_COLUMNS = ["d3D", "elevation", "azimuth"]
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


class UAVDataset(Dataset):
    def __init__(
        self,
        config: DatasetConfig,
        target: str = TARGET,
        dataset_filenames: list[str] | None = None,
        max_samples: int | None = None,
        seed: int = 42,
    ):
        """
        Args:
            config: Dataset configuration
            target (str): The column name to predict (e.g., 'RSRP_NR_5G').
            dataset_filenames (list[str]): Specific CSV filenames to load.
            max_samples (int): If set, limits the dataset to this many samples (shuffled).
            seed (int): Seed for shuffling when max_samples is used.
        """
        clean_home = Path(str(get_env_var("DATASET_CLEAN_HOME")))
        dataset_dir = clean_home / "data" / config.get_id()
        self.dataset_dir = dataset_dir
        self.target = target
        self.dataset_filenames = dataset_filenames
        self.r_max = R_MAX
        self.tx_power = P_TX
        self.fc = CENTER_FREQUENCY

        if not self.dataset_dir.exists() or not any(self.dataset_dir.iterdir()):
            process(config)

        if self.dataset_filenames is None:
            csv_files = sorted(list(dataset_dir.glob("*.csv")))
        else:
            csv_files = [
                self.dataset_dir / filename for filename in self.dataset_filenames
            ]

        raw_dfs = [pd.read_csv(f) for f in csv_files]

        processed_dfs = []

        for df in raw_dfs:
            if not all(col in df.columns for col in FEATURE_COLUMNS):
                missing_cols = [col for col in FEATURE_COLUMNS if col not in df.columns]
                raise ValueError(
                    f"Columns are missing for features: {', '.join(missing_cols)}"
                )

            if self.target not in df.columns:
                continue

            cols_to_keep = [target] + FEATURE_COLUMNS
            df = df[cols_to_keep]
            processed_dfs.append(df)

        if not processed_dfs:
            raise ValueError(
                "No datasets remaining after applying target filtering logic."
            )

        self.data = pd.concat(processed_dfs, ignore_index=True)

        # Apply max_samples if requested
        if max_samples is not None and max_samples < len(self.data):
            self.data = self.data.sample(n=max_samples, random_state=seed).reset_index(
                drop=True
            )

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
        y_target_seq = torch.zeros(self.r_max, dtype=torch.float32)
        y_target_seq[j_index] = y

        return x, y_target_seq, mask


def get_uav_splits(
    dataset: UAVDataset,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
) -> tuple[Subset, Subset, Subset]:
    """
    Splits the dataset into train, validation, and test sets deterministically.
    """
    if not math.isclose(train_split + val_split + test_split, 1.0):
        raise ValueError("Splits must sum to 1.0")

    n_total = len(dataset)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)
    n_test = n_total - n_train - n_val

    return random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed),
    )


def get_sequential_splits(
    dataset: UAVDataset,
    train_frac: float,
    val_frac: float = 0.0,
) -> tuple[Subset, Subset, Subset]:
    """
    Splits the dataset sequentially (temporal split).
    Returns (train_subset, val_subset, test_subset).
    """
    n_total = len(dataset)
    n_train = int(n_total * train_frac)
    n_val = int(n_total * val_frac)
    n_test = n_total - n_train - n_val

    train_indices = list(range(0, n_train))
    val_indices = list(range(n_train, n_train + n_val))
    test_indices = list(range(n_train + n_val, n_total))

    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
        Subset(dataset, test_indices),
    )
