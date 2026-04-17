"""
Extreme Learning Machine (ELM) model for 5G RSRP REM sequence prediction.

This model is designed as a lightweight alternative to InceptionTime,
suitable for edge compute on UAVs. It uses a random, frozen hidden layer
and a trainable output layer solved via a one-shot analytical solution.

Input
-----
  x  : (B, 8, R_max)
Output
------
  pred : (B, R_max)
"""

import math
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from aerpaw_processing.paper.uav_dataloader import UAVDataset, get_uav_splits

# ---------------------------------------------------------------------------
# ELM Network
# ---------------------------------------------------------------------------

class ExtremeLearningMachine(nn.Module):
    """
    1-D ELM adapted to the RSRP sequence prediction task.

    Parameters
    ----------
    in_channels : input feature channels (8 in this work)
    r_max       : sequence length (500 m)
    hidden_size : number of nodes in the random hidden layer
    seed        : seed for random hidden layer initialization
    """
    def __init__(
        self,
        in_channels: int = 8,
        r_max: int = 500,
        hidden_size: int = 128,
        seed: int = 42,
    ):
        super().__init__()
        self.r_max = r_max
        self.hidden_size = hidden_size

        # ── Random hidden layer (frozen) ──────────────────────────────────
        torch.manual_seed(seed)
        self.hidden_layer = nn.Conv1d(in_channels, hidden_size, kernel_size=1, bias=True)
        # Freeze parameters
        for p in self.hidden_layer.parameters():
            p.requires_grad = False

        self.activation = nn.ReLU(inplace=True)

        # ── Output layer ──────────────────────────────────────────────────
        # In analytical ELM, this is solved once. We use a Conv1d(1x1) to store beta.
        self.output_layer = nn.Conv1d(hidden_size, 1, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 8, R_max)

        Returns
        -------
        pred : (B, R_max)  – predicted RSRP at every radial step
        """
        h = self.hidden_layer(x)
        h = self.activation(h)
        pred = self.output_layer(h).squeeze(1)  # (B, R_max)
        return pred

    @torch.no_grad()
    def fit(self, loader: DataLoader, device: torch.device):
        """
        Solve for output weights using the Moore-Penrose pseudo-inverse.
        Since we only have one ground truth per sample (masked), we solve
        for the mapping from hidden features at bin j to the target value.
        """
        self.to(device)
        self.eval()
        
        h_list = []
        y_list = []

        for x, y_seq, mask in loader:
            x = x.to(device)
            mask = mask.to(device)
            y_seq = y_seq.to(device)

            # Hidden representations: (B, hidden_size, R_max)
            h = self.hidden_layer(x)
            h = self.activation(h)

            # Extract hidden state and target at the masked bin j
            # h[mask] returns (N_masked, hidden_size) because mask is (B, R_max)
            # and h is (B, H, R_max). PyTorch's boolean indexing on (B, H, L) 
            # with (B, L) mask requires care.
            
            # Alternative: transpose h to (B, R_max, H) then apply mask
            h_permuted = h.permute(0, 2, 1) # (B, R_max, H)
            h_at_j = h_permuted[mask] # (B, H)
            y_at_j = y_seq[mask] # (B,)

            h_list.append(h_at_j)
            y_list.append(y_at_j)

        H = torch.cat(h_list, dim=0) # (N, H)
        Y = torch.cat(y_list, dim=0).unsqueeze(1) # (N, 1)

        # Solve H * beta = Y  => beta = pinv(H) * Y
        # beta will be (H, 1)
        beta = torch.linalg.pinv(H) @ Y
        
        # Load beta into output_layer weight: (1, H, 1)
        self.output_layer.weight.data = beta.t().unsqueeze(2)

# ---------------------------------------------------------------------------
# Evaluation utility
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    preds_list, targets_list = [], []

    for x, y_seq, mask in loader:
        x = x.to(device)
        y_seq = y_seq.to(device).float()
        mask = mask.to(device)

        pred = model(x)
        preds_list.append(pred[mask].cpu())
        targets_list.append(y_seq[mask].cpu())

    if not preds_list:
        return {"RMSE": 0.0, "MAE": 0.0, "R2": 0.0}

    preds = torch.cat(preds_list)
    targets = torch.cat(targets_list)

    residuals = preds - targets
    rmse = residuals.pow(2).mean().sqrt().item()
    mae = residuals.abs().mean().item()

    ss_res = residuals.pow(2).sum().item()
    ss_tot = (targets - targets.mean()).pow(2).sum().item()
    r2 = 1.0 - ss_res / (ss_tot + 1e-8)

    return {"RMSE": rmse, "MAE": mae, "R2": r2}

# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fit Extreme Learning Machine (ELM) for 5G RSRP REM prediction."
    )
    p.add_argument(
        "--target",
        default="RSRP_NR_5G",
        help="Column name to predict (default: RSRP_NR_5G)",
    )
    p.add_argument(
        "--dataset_filenames",
        nargs="*",
        default=None,
        help="Specific CSV filenames inside the dataset dir; defaults to all CSVs found",
    )
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--val_split", type=float, default=0.15)
    p.add_argument("--test_split", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--hidden_size",
        type=int,
        default=1024,
        help="Number of nodes in the random hidden layer (default: 1024)",
    )
    p.add_argument(
        "--device",
        default=None,
        help="'cuda', 'mps', or 'cpu'; auto-detected if omitted",
    )
    return p.parse_args()

def main() -> None:
    args = parse_args()
    from aerpaw_processing.paper.preprocess_utils import DatasetConfig

    config = DatasetConfig()
    dataset = UAVDataset(
        config=config,
        target=args.target,
        dataset_filenames=args.dataset_filenames,
        seed=args.seed
    )

    train_ds, val_ds, test_ds = get_uav_splits(
        dataset, 
        train_split=1.0 - args.val_split - args.test_split,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    if args.device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = args.device
    dev = torch.device(device)

    model = ExtremeLearningMachine(
        in_channels=8,
        r_max=dataset.r_max,
        hidden_size=args.hidden_size,
        seed=args.seed
    )

    t0 = time.time()
    model.fit(train_loader, dev)
    fit_time = time.time() - t0
    
    val_metrics = evaluate(model, val_loader, dev)
    test_metrics = evaluate(model, test_loader, dev)

    print(f"ELM Fit Time: {fit_time:.4f}s")
    print(f"Val Metrics:  RMSE={val_metrics['RMSE']:.4f}, MAE={val_metrics['MAE']:.4f}, R2={val_metrics['R2']:.4f}")
    print(f"Test Metrics: RMSE={test_metrics['RMSE']:.4f}, MAE={test_metrics['MAE']:.4f}, R2={test_metrics['R2']:.4f}")

if __name__ == "__main__":
    main()
