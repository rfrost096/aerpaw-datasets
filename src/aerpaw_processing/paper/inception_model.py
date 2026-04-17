"""
Inception (InceptionTime) model for 5G RSRP REM sequence prediction.

Architecture follows the InceptionTime design [Fawaz et al., 2020] adapted
for the 1-D sequence-to-sequence REM reconstruction task described in:

  "AI-Enabled Wireless Propagation Modeling and Radio Environment Maps
   for 5G Aerial Wireless Networks"  (Reddy et al., IEEE JSTEAP)

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
# Inception module
# ---------------------------------------------------------------------------


class InceptionModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        nb_filters: int = 32,
        kernel_sizes: tuple[int, int, int] = (39, 19, 9),
        use_bottleneck: bool = True,
    ):
        super().__init__()
        self.use_bottleneck = use_bottleneck

        bottleneck_channels = nb_filters if use_bottleneck else in_channels
        if use_bottleneck:
            self.bottleneck = nn.Conv1d(
                in_channels, bottleneck_channels, kernel_size=1, bias=False
            )

        self.conv_branches = nn.ModuleList(
            [
                nn.Conv1d(
                    bottleneck_channels,
                    nb_filters,
                    kernel_size=k,
                    padding=k // 2,
                    bias=False,
                )
                for k in kernel_sizes
            ]
        )

        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.mp_conv = nn.Conv1d(in_channels, nb_filters, kernel_size=1, bias=False)

        self.bn = nn.BatchNorm1d(4 * nb_filters)
        self.relu = nn.ReLU(inplace=True)

        self.out_channels = 4 * nb_filters

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck_out = self.bottleneck(x) if self.use_bottleneck else x
        branch_outs = [conv(bottleneck_out) for conv in self.conv_branches]
        mp_out = self.mp_conv(self.maxpool(x))
        out = torch.cat(branch_outs + [mp_out], dim=1)
        return self.relu(self.bn(out))


class ResidualShortcut(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, shortcut: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.bn(self.conv(shortcut)))


class InceptionTime(nn.Module):
    def __init__(
        self,
        in_channels: int = 8,
        r_max: int = 500,
        nb_filters: int = 32,
        nb_modules: int = 6,
        kernel_sizes: tuple[int, int, int] = (39, 19, 9),
        depth_residual: int = 3,
    ):
        super().__init__()
        self.r_max = r_max
        self.depth_residual = depth_residual

        self.inception_modules = nn.ModuleList()
        self.residual_shortcuts = nn.ModuleList()

        current_channels = in_channels
        residual_in_channels = in_channels

        for i in range(nb_modules):
            use_bottleneck = current_channels > nb_filters
            module = InceptionModule(
                in_channels=current_channels,
                nb_filters=nb_filters,
                kernel_sizes=kernel_sizes,
                use_bottleneck=use_bottleneck,
            )
            self.inception_modules.append(module)
            current_channels = module.out_channels

            if (i + 1) % depth_residual == 0:
                shortcut = ResidualShortcut(residual_in_channels, current_channels)
                self.residual_shortcuts.append(shortcut)
                residual_in_channels = current_channels

        self.output_conv = nn.Conv1d(current_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual_input = x
        shortcut_idx = 0

        for i, module in enumerate(self.inception_modules):
            x = module(x)
            if (i + 1) % self.depth_residual == 0:
                x = self.residual_shortcuts[shortcut_idx](residual_input, x)
                residual_input = x
                shortcut_idx += 1

        pred = self.output_conv(x).squeeze(1)
        return pred


def masked_smooth_l1_loss(
    pred: torch.Tensor,
    target_seq: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    pred_at_j = pred[mask]
    target_at_j = target_seq[mask]
    return nn.functional.smooth_l1_loss(pred_at_j, target_at_j.float())


def build_optimizer_and_scheduler(
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    warmup_steps: int = 500,
    total_steps: int = 10_000,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        return math.sqrt(warmup_steps) / math.sqrt(max(step, warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for x, y_seq, mask in loader:
        x = x.to(device)
        y_seq = y_seq.to(device).float()
        mask = mask.to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss = masked_smooth_l1_loss(pred, y_seq, mask)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


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


def train(
    dataset,
    *,
    val_split: float = 0.15,
    test_split: float = 0.15,
    batch_size: int = 64,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    warmup_steps: int = 500,
    nb_filters: int = 32,
    nb_modules: int = 6,
    depth_residual: int = 3,
    kernel_sizes: tuple[int, int, int] = (39, 19, 9),
    checkpoint_dir: str | None = None,
    device: str | None = None,
    seed: int = 42,
    target_rmse: float | None = None,
) -> tuple[InceptionTime, int]:
    """
    Returns (best_model, total_epochs_run).
    """
    torch.manual_seed(seed)

    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    dev = torch.device(device)
    print(f"[train] using device: {dev}")

    train_ds, val_ds, test_ds = get_uav_splits(
        dataset,
        train_split=1.0 - val_split - test_split,
        val_split=val_split,
        test_split=test_split,
        seed=seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(dev.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(dev.type == "cuda"),
    )
    print(f"[train] samples  train={len(train_ds)}  val={len(val_ds)}")

    # Ensure deterministic model initialization
    torch.manual_seed(seed)
    model = InceptionTime(
        in_channels=8,
        r_max=dataset.r_max,
        nb_filters=nb_filters,
        nb_modules=nb_modules,
        kernel_sizes=kernel_sizes,
        depth_residual=depth_residual,
    ).to(dev)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] model params: {n_params:,}")

    total_steps = epochs * len(train_loader)
    optimizer, scheduler = build_optimizer_and_scheduler(
        model,
        lr=lr,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )

    ckpt_path = None
    if checkpoint_dir is not None:
        ckpt_path = Path(checkpoint_dir) / "inception_best.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    best_rmse = float("inf")
    best_state = None
    epochs_run = 0

    print(
        f"\n{'Epoch':>6}  {'Train Loss':>12}  {'Val RMSE':>10}  "
        f"{'Val MAE':>9}  {'Val R2':>8}  {'LR':>10}  {'Time':>7}"
    )
    print("-" * 72)

    for epoch in range(1, epochs + 1):
        epochs_run = epoch
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, dev)
        val_metrics = evaluate(model, val_loader, dev)

        current_lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        print(
            f"{epoch:>6}  {train_loss:>12.4f}  "
            f"{val_metrics['RMSE']:>10.4f}  "
            f"{val_metrics['MAE']:>9.4f}  "
            f"{val_metrics['R2']:>8.4f}  "
            f"{current_lr:>10.2e}  "
            f"{elapsed:>6.1f}s"
        )

        if val_metrics["RMSE"] < best_rmse:
            best_rmse = val_metrics["RMSE"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if ckpt_path is not None:
                torch.save(best_state, ckpt_path)

        # Early stopping based on target_rmse
        if target_rmse is not None and val_metrics["RMSE"] <= target_rmse:
            print(f"\n[train] Early stopping: Target RMSE {target_rmse} reached.")
            break

    print(f"\n[train] best val RMSE: {best_rmse:.4f} dB")
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, epochs_run


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train InceptionTime for 5G RSRP REM prediction."
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
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--val_split", type=float, default=0.15)
    p.add_argument("--test_split", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--nb_filters", type=int, default=32)
    p.add_argument("--nb_modules", type=int, default=6)
    p.add_argument("--depth_residual", type=int, default=3)
    p.add_argument("--kernel_sizes", type=int, nargs=3, default=[39, 19, 9])
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--device", default=None)
    p.add_argument("--target_rmse", type=float, default=None)
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

    train(
        dataset,
        val_split=args.val_split,
        test_split=args.test_split,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        nb_filters=args.nb_filters,
        nb_modules=args.nb_modules,
        depth_residual=args.depth_residual,
        kernel_sizes=tuple(args.kernel_sizes),
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        seed=args.seed,
        target_rmse=args.target_rmse,
    )


if __name__ == "__main__":
    main()
