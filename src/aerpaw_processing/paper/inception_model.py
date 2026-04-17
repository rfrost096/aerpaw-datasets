"""
Inception (InceptionTime) model for 5G RSRP REM sequence prediction.

Architecture follows the InceptionTime design [Fawaz et al., 2020] adapted
for the 1-D sequence-to-sequence REM reconstruction task described in:

  "AI-Enabled Wireless Propagation Modeling and Radio Environment Maps
   for 5G Aerial Wireless Networks"  (Reddy et al., IEEE JSTEAP)

Input
-----
  x  : (B, 8, R_max)  – physics-aligned feature matrix Γ_i (Section III-A)
       Row 0 : P_TX_SS   (dB)            – constant along radial axis
       Row 1 : log10(f_c)                – constant along radial axis
       Row 2 : log10(d_j)               – varies with radial step
       Row 3 : θ_i  (elevation, rad)    – constant along radial axis
       Row 4 : φ_i  (azimuth, rad)      – constant along radial axis
       Row 5 : x^j_i  (Cartesian x, m)  – varies with radial step
       Row 6 : y^j_i  (Cartesian y, m)  – varies with radial step
       Row 7 : z^j_i  (Cartesian z, m)  – varies with radial step

Output
------
  pred : (B, R_max)  – predicted RSRP (dBm) at every radial step

Training
--------
  For each sample only one radial bin j carries the measured RSRP Ω_i.
  The mask tensor (B, R_max) is True only at position j, so the Smooth-L1
  loss is computed exclusively at that bin (masking strategy, Fig. 6).
"""

import math
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# ---------------------------------------------------------------------------
# Inception module
# ---------------------------------------------------------------------------


class InceptionModule(nn.Module):
    """
    Single Inception module from InceptionTime.

    Architecture (per module):
      - Bottleneck conv  : 1×1  → reduces in_channels to nb_filters
      - Three parallel 1-D convolutions on the bottleneck output
        with kernel sizes k1, k2, k3 (padded to preserve length)
      - MaxPool branch   : MaxPool1d on the raw input + 1×1 conv
      - All four branches are concatenated → 4 * nb_filters channels
      - BatchNorm + ReLU

    Parameters
    ----------
    in_channels  : number of input feature channels
    nb_filters   : number of filters per branch (output = 4 * nb_filters)
    kernel_sizes : tuple of three odd kernel sizes for the conv branches
    use_bottleneck: whether to apply the 1×1 bottleneck (disabled for 1st
                   layer when in_channels is small)
    """

    def __init__(
        self,
        in_channels: int,
        nb_filters: int = 32,
        kernel_sizes: tuple[int, int, int] = (39, 19, 9),
        use_bottleneck: bool = True,
    ):
        super().__init__()
        self.use_bottleneck = use_bottleneck

        # ── bottleneck ───────────────────────────────────────────────────
        bottleneck_channels = nb_filters if use_bottleneck else in_channels
        if use_bottleneck:
            self.bottleneck = nn.Conv1d(
                in_channels, bottleneck_channels, kernel_size=1, bias=False
            )

        # ── three parallel convolutions on bottleneck output ─────────────
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

        # ── maxpool branch on raw input ───────────────────────────────────
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.mp_conv = nn.Conv1d(in_channels, nb_filters, kernel_size=1, bias=False)

        # ── post-concat normalisation ─────────────────────────────────────
        self.bn = nn.BatchNorm1d(4 * nb_filters)
        self.relu = nn.ReLU(inplace=True)

        self.out_channels = 4 * nb_filters

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C_in, L)

        Returns
        -------
        out : (B, 4*nb_filters, L)
        """
        bottleneck_out = self.bottleneck(x) if self.use_bottleneck else x

        branch_outs = [conv(bottleneck_out) for conv in self.conv_branches]
        mp_out = self.mp_conv(self.maxpool(x))

        out = torch.cat(branch_outs + [mp_out], dim=1)  # (B, 4*F, L)
        return self.relu(self.bn(out))


# ---------------------------------------------------------------------------
# Residual shortcut (used between every two Inception blocks)
# ---------------------------------------------------------------------------


class ResidualShortcut(nn.Module):
    """
    1×1 conv + BN shortcut added to the output of every *depth_residual*-th
    Inception module.  Matches InceptionTime Fig. 2 in the original paper.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, shortcut: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        shortcut : input to the residual block  (B, C_in, L)
        x        : output of the residual block (B, C_out, L)
        """
        return self.relu(x + self.bn(self.conv(shortcut)))


# ---------------------------------------------------------------------------
# Full InceptionTime network
# ---------------------------------------------------------------------------


class InceptionTime(nn.Module):
    """
    InceptionTime encoder adapted to the 1-D RSRP sequence prediction task.

    The network stacks *nb_modules* Inception modules.  A residual shortcut
    is applied every *depth_residual* modules.  A final pointwise convolution
    maps the feature sequence to a scalar RSRP prediction at every radial
    step.

    Parameters
    ----------
    in_channels   : input feature channels (8 in this work)
    r_max         : sequence length (500 m)
    nb_filters    : filters per Inception branch
    nb_modules    : total number of Inception modules (depth)
    kernel_sizes  : three kernel sizes for parallel conv branches
    depth_residual: apply residual shortcut every N modules
    """

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

        # ── build Inception modules ───────────────────────────────────────
        self.inception_modules = nn.ModuleList()
        self.residual_shortcuts = nn.ModuleList()

        current_channels = in_channels
        residual_in_channels = in_channels  # track input channel to residual block

        for i in range(nb_modules):
            use_bottleneck = current_channels > nb_filters
            module = InceptionModule(
                in_channels=current_channels,
                nb_filters=nb_filters,
                kernel_sizes=kernel_sizes,
                use_bottleneck=use_bottleneck,
            )
            self.inception_modules.append(module)
            current_channels = module.out_channels  # 4 * nb_filters

            # ── residual shortcut at end of every depth_residual block ────
            if (i + 1) % depth_residual == 0:
                shortcut = ResidualShortcut(residual_in_channels, current_channels)
                self.residual_shortcuts.append(shortcut)
                residual_in_channels = current_channels  # reset for next block

        # ── final pointwise conv: feature channels → 1 scalar per step ───
        self.output_conv = nn.Conv1d(current_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 8, R_max)

        Returns
        -------
        pred : (B, R_max)  – predicted RSRP at every radial step
        """
        residual_input = x
        shortcut_idx = 0

        for i, module in enumerate(self.inception_modules):
            x = module(x)

            # apply residual shortcut at the end of every block
            if (i + 1) % self.depth_residual == 0:
                x = self.residual_shortcuts[shortcut_idx](residual_input, x)
                residual_input = x
                shortcut_idx += 1

        pred = self.output_conv(x).squeeze(1)  # (B, R_max)
        return pred


# ---------------------------------------------------------------------------
# Masked Smooth-L1 loss
# ---------------------------------------------------------------------------


def masked_smooth_l1_loss(
    pred: torch.Tensor,
    target_seq: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Smooth-L1 loss only at the masked radial bin j.

    Parameters
    ----------
    pred       : (B, R_max)  – model output
    target_seq : (B, R_max)  – zero everywhere except bin j (holds Ω_i)
    mask       : (B, R_max)  – bool, True only at bin j

    Returns
    -------
    scalar loss averaged over the batch
    """
    # Extract predicted and target values at the single active bin per sample
    pred_at_j = pred[mask]  # (B,) – one value per sample
    target_at_j = target_seq[mask]  # (B,) – Ω_i for each sample
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
    """
    AdamW optimiser with a linear-warmup / square-root-decay schedule,
    matching the LWSRD strategy described in the paper (Section III-C, Stage-2).
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        # square-root decay after warmup
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
    """Run one full pass over the training set. Returns mean batch loss."""
    model.train()
    total_loss = 0.0

    for x, y_seq, mask in loader:
        x = x.to(device)  # (B, 8, R_max)
        y_seq = y_seq.to(device).float()  # (B, R_max)
        mask = mask.to(device)  # (B, R_max) bool

        optimizer.zero_grad()
        pred = model(x)  # (B, R_max)
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
    val_split: float = 0.138,  # ~644 samples mirrors paper's test split
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
) -> InceptionTime:
    """
    Train InceptionTime on an InceptionDataset and return the best model.

    Parameters
    ----------
    dataset        : InceptionDataset instance
    val_split      : fraction of data held out for validation
    batch_size     : training batch size
    epochs         : number of full passes over the training set
    lr             : peak learning rate for AdamW
    weight_decay   : AdamW weight decay
    warmup_steps   : number of LR warmup steps
    nb_filters     : Inception filters per branch  (paper uses 32)
    nb_modules     : number of stacked Inception modules (paper uses 6 blocks)
    depth_residual : residual shortcut every N modules
    kernel_sizes   : parallel conv kernel sizes (must be odd for symmetric pad)
    checkpoint_dir : directory to save 'inception_best.pt'; skipped if None
    device         : 'cuda', 'mps', or 'cpu'; auto-detected if None
    seed           : random seed for reproducibility

    Returns
    -------
    model : trained InceptionTime (loaded from best checkpoint if saved)
    """
    torch.manual_seed(seed)

    # ── device ───────────────────────────────────────────────────────────
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

    # ── train / val split ────────────────────────────────────────────────
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

    # ── model ─────────────────────────────────────────────────────────────
    model = InceptionTime(
        in_channels=8,  # fixed by Γ_i feature matrix
        r_max=dataset.r_max,  # R_max = 500
        nb_filters=nb_filters,
        nb_modules=nb_modules,
        kernel_sizes=kernel_sizes,
        depth_residual=depth_residual,
    ).to(dev)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] model params: {n_params:,}")

    # ── optimiser & scheduler ─────────────────────────────────────────────
    total_steps = epochs * len(train_loader)
    optimizer, scheduler = build_optimizer_and_scheduler(
        model,
        lr=lr,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )

    # ── checkpoint setup ──────────────────────────────────────────────────
    ckpt_path = None
    if checkpoint_dir is not None:
        ckpt_path = Path(checkpoint_dir) / "inception_best.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    best_rmse = float("inf")
    best_state = None

    # ── training loop ─────────────────────────────────────────────────────
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

        # ── checkpoint best model ─────────────────────────────────────────
        if val_metrics["RMSE"] < best_rmse:
            best_rmse = val_metrics["RMSE"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if ckpt_path is not None:
                torch.save(best_state, ckpt_path)

    print(f"\n[train] best val RMSE: {best_rmse:.4f} dB")
    if ckpt_path is not None:
        print(f"[train] checkpoint saved → {ckpt_path}")

    # restore best weights before returning
    if best_state is not None:
        model.load_state_dict(best_state)

    return model


# ---------------------------------------------------------------------------
# CLI entry-point  (python inception_model.py --help)
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train InceptionTime for 5G RSRP REM prediction."
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
        help="Specific CSV filenames inside the dataset dir; "
        "defaults to all CSVs found",
    )
    # Training hyper-parameters
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--val_split", type=float, default=0.138)
    p.add_argument("--seed", type=int, default=42)
    # Architecture hyper-parameters
    p.add_argument(
        "--nb_filters",
        type=int,
        default=32,
        help="Filters per Inception branch (default: 32)",
    )
    p.add_argument(
        "--nb_modules",
        type=int,
        default=6,
        help="Number of stacked Inception modules (default: 6)",
    )
    p.add_argument(
        "--depth_residual",
        type=int,
        default=3,
        help="Residual shortcut every N modules (default: 3)",
    )
    p.add_argument(
        "--kernel_sizes",
        type=int,
        nargs=3,
        default=[39, 19, 9],
        metavar=("K1", "K2", "K3"),
        help="Three kernel sizes for parallel conv branches "
        "(must be odd; default: 39 19 9)",
    )
    # I/O
    p.add_argument(
        "--checkpoint_dir",
        default="checkpoints",
        help="Directory to write inception_best.pt (default: checkpoints)",
    )
    p.add_argument(
        "--device",
        default=None,
        help="'cuda', 'mps', or 'cpu'; auto-detected if omitted",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── lazy import to keep the file usable as a library ─────────────────
    from aerpaw_processing.paper.inception_dataloader import InceptionDataset
    from aerpaw_processing.paper.preprocess_utils import DatasetConfig

    # Build a default DatasetConfig – adjust as needed for your environment
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
        nb_filters=args.nb_filters,
        nb_modules=args.nb_modules,
        depth_residual=args.depth_residual,
        kernel_sizes=tuple(args.kernel_sizes),
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
