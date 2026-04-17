import math
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from aerpaw_processing.paper.inception_dataloader import InceptionDataset
from aerpaw_processing.paper.preprocess_utils import DatasetConfig

class FSPLModel(nn.Module):
    def __init__(self):
        super().__init__()
        # c = 299792458
        # 20 * log10(4 * pi / c) = -147.55
        self.constant = 20 * math.log10(4 * math.pi / 299792458)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (B, 8, R_max)
        # x[:, 0, :] is p_tx
        # x[:, 1, :] is log10(fc)
        # x[:, 2, :] is log10(d)
        
        p_tx = x[:, 0, :]
        log10_fc = x[:, 1, :]
        log10_d = x[:, 2, :]
        
        fspl = 20 * log10_d + 20 * log10_fc + self.constant
        rsrp = p_tx - fspl
        return rsrp

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """
    Evaluate RMSE, MAE, and R² on *loader* using the masked prediction
    at each sample's ground-truth radial bin j.
    """
    model.eval()
    preds_list, targets_list = [], []

    for x, y_seq, mask in loader:
        x = x.to(device)
        y_seq = y_seq.to(device).float()
        mask = mask.to(device)

        pred = model(x)
        preds_list.append(pred[mask].cpu())
        targets_list.append(y_seq[mask].cpu())

    if len(preds_list) == 0:
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


def evaluate_and_print(
    dataset_name: str,
    dataset,
    model: nn.Module,
    val_split: float,
    batch_size: int,
    device: torch.device,
    seed: int,
):
    torch.manual_seed(seed)
    n_total = len(dataset)
    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    t0 = time.time()
    val_metrics = evaluate(model, val_loader, device)
    elapsed = time.time() - t0

    name_str = dataset_name
    if len(name_str) > 25:
        name_str = name_str[:22] + "..."

    print(
        f"{name_str:>25}  "
        f"{val_metrics['RMSE']:>10.4f}  "
        f"{val_metrics['MAE']:>9.4f}  "
        f"{val_metrics['R2']:>8.4f}  "
        f"{elapsed:>6.1f}s"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate FSPL for 5G RSRP REM prediction."
    )
    # Dataset / config
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
    p.add_argument("--val_split", type=float, default=0.138)
    p.add_argument("--seed", type=int, default=42)
    # I/O
    p.add_argument(
        "--device",
        default=None,
        help="'cuda', 'mps', or 'cpu'; auto-detected if omitted",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    config = DatasetConfig()

    if args.device is None:
        device_str = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device_str = args.device
    dev = torch.device(device_str)
    print(f"[eval] using device: {dev}")

    model = FSPLModel().to(dev)

    dataset_all = InceptionDataset(
        config=config,
        target=args.target,
        dataset_filenames=args.dataset_filenames,
    )

    if args.dataset_filenames is None:
        dataset_dir = dataset_all.dataset_dir
        csv_files = list(dataset_dir.glob("*.csv"))
        filenames = sorted([f.name for f in csv_files])
    else:
        filenames = args.dataset_filenames

    print(
        f"\n{'Dataset':>25}  {'Val RMSE':>10}  "
        f"{'Val MAE':>9}  {'Val R2':>8}  {'Time':>7}"
    )
    print("-" * 67)

    for filename in filenames:
        try:
            single_dataset = InceptionDataset(
                config=config,
                target=args.target,
                dataset_filenames=[filename],
            )
            evaluate_and_print(
                filename, single_dataset, model, args.val_split, args.batch_size, dev, args.seed
            )
        except ValueError:
            # Skip datasets that don't have the target column or valid rows
            pass

    evaluate_and_print(
        "All", dataset_all, model, args.val_split, args.batch_size, dev, args.seed
    )

if __name__ == "__main__":
    main()
