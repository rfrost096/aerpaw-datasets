"""
Extreme Learning Machine (ELM) model for 5G RSRP REM sequence prediction.

This model is designed as a lightweight alternative to InceptionTime,
suitable for edge compute on UAVs. It uses a random, frozen hidden layer
and a trainable output layer.

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
from torch.utils.data import DataLoader, random_split

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
    """
    def __init__(
        self,
        in_channels: int = 8,
        r_max: int = 500,
        hidden_size: int = 128,
    ):
        super().__init__()
        self.r_max = r_max
        self.hidden_size = hidden_size

        # ── Random hidden layer (frozen) ──────────────────────────────────
        self.hidden_layer = nn.Conv1d(in_channels, hidden_size, kernel_size=1, bias=True)
        # Freeze parameters
        self.hidden_layer.weight.requires_grad = False
        self.hidden_layer.bias.requires_grad = False

        self.activation = nn.ReLU(inplace=True)

        # ── Trainable output layer ────────────────────────────────────────
        self.output_layer = nn.Conv1d(hidden_size, 1, kernel_size=1, bias=True)

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

# ---------------------------------------------------------------------------
# Masked Smooth-L1 loss
# ---------------------------------------------------------------------------

def masked_smooth_l1_loss(
    pred: torch.Tensor,
    target_seq: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    pred_at_j = pred[mask]
    target_at_j = target_seq[mask]
    return nn.functional.smooth_l1_loss(pred_at_j, target_at_j.float())

# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def build_optimizer_and_scheduler(
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    warmup_steps: int = 500,
    total_steps: int = 10_000,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )

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

# ---------------------------------------------------------------------------
# Training entry-point
# ---------------------------------------------------------------------------

def train(
    dataset,
    *,
    val_split: float = 0.138,
    batch_size: int = 64,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    warmup_steps: int = 500,
    hidden_size: int = 128,
    checkpoint_dir: str | None = None,
    device: str | None = None,
    seed: int = 42,
) -> ExtremeLearningMachine:
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

    n_total = len(dataset)
    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
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
    print(f"[train] samples  train={n_train}  val={n_val}")

    model = ExtremeLearningMachine(
        in_channels=8,
        r_max=dataset.r_max,
        hidden_size=hidden_size,
    ).to(dev)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] model trainable params: {n_params:,}")

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
        ckpt_path = Path(checkpoint_dir) / "elm_best.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    best_rmse = float("inf")
    best_state = None

    print(
        f"\n{'Epoch':>6}  {'Train Loss':>12}  {'Val RMSE':>10}  "
        f"{'Val MAE':>9}  {'Val R2':>8}  {'LR':>10}  {'Time':>7}"
    )
    print("-" * 72)

    for epoch in range(1, epochs + 1):
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

    print(f"\n[train] best val RMSE: {best_rmse:.4f} dB")
    if ckpt_path is not None:
        print(f"[train] checkpoint saved → {ckpt_path}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model

# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Extreme Learning Machine (ELM) for 5G RSRP REM prediction."
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
    p.add_argument("--val_split", type=float, default=0.138)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--hidden_size",
        type=int,
        default=128,
        help="Number of nodes in the random hidden layer (default: 128)",
    )
    p.add_argument(
        "--checkpoint_dir",
        default="checkpoints",
        help="Directory to write elm_best.pt (default: checkpoints)",
    )
    p.add_argument(
        "--device",
        default=None,
        help="'cuda', 'mps', or 'cpu'; auto-detected if omitted",
    )
    return p.parse_args()

def main() -> None:
    args = parse_args()

    from aerpaw_processing.paper.inception_dataloader import InceptionDataset
    from aerpaw_processing.paper.preprocess_utils import DatasetConfig

    config = DatasetConfig()

    dataset = InceptionDataset(
        config=config,
        target=args.target,
        dataset_filenames=args.dataset_filenames,
    )

    train(
        dataset,
        val_split=args.val_split,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        hidden_size=args.hidden_size,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        seed=args.seed,
    )

if __name__ == "__main__":
    main()
