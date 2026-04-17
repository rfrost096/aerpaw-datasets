import argparse
import math
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from aerpaw_processing.paper.uav_dataloader import UAVDataset, get_uav_splits, get_sequential_splits
from aerpaw_processing.paper.fspl_model import FSPLModel, evaluate as evaluate_fspl
from aerpaw_processing.paper.inception_model import InceptionTime, train as train_inception, evaluate as evaluate_inception
from aerpaw_processing.paper.elm_model import ExtremeLearningMachine, evaluate as evaluate_elm
from aerpaw_processing.paper.preprocess_utils import DatasetConfig

# Constants
TARGET = "RSRP_NR_5G"
BATCH_SIZE = 64
DEFAULT_SEED = 42
DEFAULT_FLIGHT = "Dataset_24_PawPrints_5G_30m_Flight_2.csv"
TARGET_RMSE = 5.0

def run_experiment(args):
    # Set master seed
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device_str = (
        args.device if args.device else
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    device = torch.device(device_str)
    print(f"Using device: {device}")

    config = DatasetConfig()
    
    # 1. Define Fixed Test Set (Unseen data) from ALL flights
    full_dataset = UAVDataset(config=config, target=TARGET, seed=seed)
    train_ds_full, val_ds_full, test_ds_full = get_uav_splits(
        full_dataset, train_split=0.7, val_split=0.15, test_split=0.15, seed=seed
    )
    
    test_loader = DataLoader(test_ds_full, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Total samples (all flights): {len(full_dataset)}, Test samples: {len(test_ds_full)}")

    results = {
        "fspl": {},
        "inception": {"sample_constraints": [], "time_constraints": []},
        "elm": {"sample_constraints": []},
        "real_time": []
    }

    # --- Stage 1: FSPL Baseline ---
    print("\n--- Stage 1: FSPL Baseline ---")
    fspl_model = FSPLModel().to(device)
    t0 = time.time()
    fspl_metrics = evaluate_fspl(fspl_model, test_loader, device)
    fspl_inf_time = (time.time() - t0) / len(test_ds_full)
    results["fspl"] = {"metrics": fspl_metrics, "inf_time": fspl_inf_time}
    print(f"FSPL Test RMSE: {fspl_metrics['RMSE']:.4f}")

    # --- Stage 2: InceptionTime experiments (Sample Scarcity) ---
    print("\n--- Stage 2: InceptionTime Sample Constraints (Early Stopping) ---")
    
    sample_sizes = [100, 500, 1000, 2000, len(train_ds_full)]
    sample_sizes = [s for s in sample_sizes if s <= len(train_ds_full)]
    
    for n in sample_sizes:
        print(f"\nTraining Inception with {n} samples (Target RMSE: {TARGET_RMSE})...")
        # To be consistent with the plan, we subset the training data but use standard val_split for early stopping.
        # We'll use a local UAVDataset with max_samples for simplicity in the train() call.
        subset_dataset = UAVDataset(config=config, target=TARGET, max_samples=n + len(val_ds_full) + len(test_ds_full), seed=seed)
        
        t_train_start = time.time()
        model, epochs_run = train_inception(
            subset_dataset, 
            val_split=0.15,
            epochs=25, # New max epochs per request
            seed=seed,
            device=device_str,
            checkpoint_dir=None,
            target_rmse=TARGET_RMSE
        )
        train_time = time.time() - t_train_start
        
        t0 = time.time()
        metrics = evaluate_inception(model, test_loader, device)
        inf_time = (time.time() - t0) / len(test_ds_full)
        
        results["inception"]["sample_constraints"].append({
            "n": n,
            "metrics": metrics,
            "train_time": train_time,
            "epochs": epochs_run,
            "inf_time": inf_time
        })

    # --- Stage 3: Extreme Learning Machine (ELM) Experiments ---
    print("\n--- Stage 3: Extreme Learning Machine (ELM) Sample Constraints ---")
    for n in sample_sizes:
        print(f"Fitting ELM with {n} samples...")
        train_subset = torch.utils.data.Subset(train_ds_full, range(n))
        train_loader_subset = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        
        elm_model = ExtremeLearningMachine(in_channels=8, r_max=500, hidden_size=1024, seed=seed).to(device)
        
        t_train_start = time.time()
        elm_model.fit(train_loader_subset, device)
        train_time = time.time() - t_train_start
        
        t0 = time.time()
        metrics = evaluate_elm(elm_model, test_loader, device)
        inf_time = (time.time() - t0) / len(test_ds_full)
        
        results["elm"]["sample_constraints"].append({
            "n": n,
            "metrics": metrics,
            "train_time": train_time,
            "inf_time": inf_time
        })

    # --- Stage 4: Real-Time In-Flight Scenario ---
    print(f"\n--- Stage 4: Real-Time In-Flight Scenario ({args.flight}) ---")
    flight_dataset = UAVDataset(config=config, target=TARGET, dataset_filenames=[args.flight], seed=seed)
    
    fractions = [0.2, 0.4, 0.6, 0.8]
    for frac in fractions:
        print(f"\nProcessing {int(frac*100)}% flight data...")
        train_ds_rt, _, test_ds_rt = get_sequential_splits(flight_dataset, train_frac=frac, val_frac=0.0)
        
        # Inception training (Full 25 epochs, no early stopping to see full potential)
        # We need a small val set for the train() function even if not strictly needed for analytical comparison
        # Let's take a tiny slice of train for val to satisfy the train() API
        n_train_rt = len(train_ds_rt)
        n_val_rt = max(1, int(n_train_rt * 0.1))
        n_train_real = n_train_rt - n_val_rt
        train_ds_real = torch.utils.data.Subset(train_ds_rt, range(n_train_real))
        val_ds_real = torch.utils.data.Subset(train_ds_rt, range(n_train_real, n_train_rt))
        
        # To use train_inception, we need a Dataset object or modify train to accept loaders.
        # Since train() creates loaders from dataset, we'll wrap our Subset in a dummy object if needed,
        # but train() expects UAVDataset attributes like r_max. 
        # Actually, let's just manually run the training loop here or refactor.
        # For brevity, we'll simulate the train() call by passing the full flight_dataset and 
        # using Subset indices manually in a local loop.
        
        # Training Inception
        print(f"Training InceptionTime on {len(train_ds_rt)} samples...")
        model_rt = InceptionTime(in_channels=8, r_max=500).to(device)
        optimizer, scheduler = torch.optim.AdamW(model_rt.parameters(), lr=1e-3), None # Simplified for RT
        train_loader_rt = DataLoader(train_ds_rt, batch_size=BATCH_SIZE, shuffle=True)
        
        t_inc_start = time.time()
        model_rt.train()
        for epoch in range(25):
            for x, y_seq, mask in train_loader_rt:
                x, y_seq, mask = x.to(device), y_seq.to(device), mask.to(device)
                optimizer.zero_grad()
                pred = model_rt(x)
                loss = torch.nn.functional.smooth_l1_loss(pred[mask], y_seq[mask].float())
                loss.backward()
                optimizer.step()
        inc_train_time = time.time() - t_inc_start
        
        # Training ELM
        print(f"Fitting ELM on {len(train_ds_rt)} samples...")
        elm_rt = ExtremeLearningMachine(in_channels=8, r_max=500, hidden_size=1024, seed=seed).to(device)
        t_elm_start = time.time()
        elm_rt.fit(train_loader_rt, device)
        elm_train_time = time.time() - t_elm_start
        
        # Evaluation on remaining unseen data
        test_loader_rt = DataLoader(test_ds_rt, batch_size=BATCH_SIZE, shuffle=False)
        
        # Inception Metrics
        t0 = time.time()
        inc_metrics = evaluate_inception(model_rt, test_loader_rt, device)
        inc_inf_time = (time.time() - t0) / len(test_ds_rt)
        
        # ELM Metrics
        t0 = time.time()
        elm_metrics = evaluate_elm(elm_rt, test_loader_rt, device)
        elm_inf_time = (time.time() - t0) / len(test_ds_rt)
        
        results["real_time"].append({
            "frac": frac,
            "train_samples": len(train_ds_rt),
            "test_samples": len(test_ds_rt),
            "inc_rmse": inc_metrics["RMSE"],
            "elm_rmse": elm_metrics["RMSE"],
            "inc_train_time": inc_train_time,
            "elm_train_time": elm_train_time,
            "inc_inf_time": inc_inf_time,
            "elm_inf_time": elm_inf_time
        })

    # --- Plotting ---
    print("\nGenerating Plots...")
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)
    
    # 1. RMSE vs Sample Size
    plt.figure(figsize=(10, 6))
    plt.axhline(y=results["fspl"]["metrics"]["RMSE"], color='r', linestyle='--', label='FSPL Baseline')
    inc_samples = [r["n"] for r in results["inception"]["sample_constraints"]]
    inc_rmse = [r["metrics"]["RMSE"] for r in results["inception"]["sample_constraints"]]
    plt.plot(inc_samples, inc_rmse, marker='o', label='InceptionTime')
    elm_samples = [r["n"] for r in results["elm"]["sample_constraints"]]
    elm_rmse = [r["metrics"]["RMSE"] for r in results["elm"]["sample_constraints"]]
    plt.plot(elm_samples, elm_rmse, marker='s', label='ELM')
    plt.xlabel("Number of Training Samples")
    plt.ylabel("Test RMSE (dB)")
    plt.title("Model Accuracy vs. Data Availability")
    plt.legend(); plt.grid(True)
    plt.savefig(output_dir / "rmse_vs_samples.png")
    
    # 2. Convergence Speed (Inception Epochs vs Samples)
    plt.figure(figsize=(10, 6))
    plt.plot(inc_samples, [r["epochs"] for r in results["inception"]["sample_constraints"]], marker='o', color='blue')
    plt.xlabel("Number of Training Samples")
    plt.ylabel("Epochs to reach 5.0 RMSE")
    plt.title("InceptionTime Convergence Speed")
    plt.grid(True)
    plt.savefig(output_dir / "inception_convergence.png")

    # 3. Real-Time Accuracy progression
    plt.figure(figsize=(10, 6))
    rt_fracs = [r["frac"]*100 for r in results["real_time"]]
    plt.plot(rt_fracs, [r["inc_rmse"] for r in results["real_time"]], marker='o', label='InceptionTime')
    plt.plot(rt_fracs, [r["elm_rmse"] for r in results["real_time"]], marker='s', label='ELM')
    plt.xlabel("% Flight Data used for Training")
    plt.ylabel("RMSE on Remaining Flight (dB)")
    plt.title(f"Real-Time In-Flight Adaptation ({args.flight})")
    plt.legend(); plt.grid(True)
    plt.savefig(output_dir / "real_time_accuracy.png")

    # --- Report Generation ---
    print("\nGenerating learning_report.md...")
    report_content = f"""# Learning Report: UAV RSRP Prediction Evolution

This report summarizes the transition from physics-based path loss models to deep learning and lightweight edge models for UAV-based 5G RSRP prediction.

## Experiment Configuration
- **Master Seed**: {seed}
- **Device**: {device_str}
- **Total Samples (Global)**: {len(full_dataset)}
- **Test Samples (Global, Unseen)**: {len(test_ds_full)}

## 1. Physics Baseline: Line of Sight (FSPL)
- **Test RMSE**: {results["fspl"]["metrics"]["RMSE"]:.4f} dB
- **Inference Time**: {results["fspl"]["inf_time"]*1000:.4f} ms/sample

## 2. Sample Scarcity & Convergence Speed
This section shows how much data is required for InceptionTime to reach a target accuracy of **{TARGET_RMSE} dB RMSE**.

| Samples | Test RMSE | Epochs to Target | Train Time (s) |
|---------|-----------|------------------|----------------|
"""
    for r in results["inception"]["sample_constraints"]:
        report_content += f"| {r['n']} | {r['metrics']['RMSE']:.4f} | {r['epochs']} | {r['train_time']:.2f} |\n"

    report_content += """
## 3. Lightweight Edge Model: ELM
ELM results on the same global sample constraints.

| Samples | Test RMSE | Fit Time (s) |
|---------|-----------|--------------|
"""
    for r in results["elm"]["sample_constraints"]:
        report_content += f"| {r['n']} | {r['metrics']['RMSE']:.4f} | {r['train_time']:.4f} |\n"

    report_content += f"""
## 4. Real-Time In-Flight Scenario
**Target Flight**: `{args.flight}`
This experiment simulates a UAV collecting data and adapting its model mid-flight. At each stage, the model is trained on the first X% of the flight and tested on the remaining (unseen) path.

| % Train | Samples | Inc RMSE | ELM RMSE | Inc Train (s) | ELM Fit (s) | Inc Inf (ms) | ELM Inf (ms) |
|---------|---------|----------|----------|---------------|-------------|--------------|--------------|
"""
    for r in results["real_time"]:
        report_content += (f"| {int(r['frac']*100)}% | {r['train_samples']} | {r['inc_rmse']:.4f} | {r['elm_rmse']:.4f} | "
                           f"{r['inc_train_time']:.2f} | {r['elm_train_time']:.4f} | "
                           f"{r['inc_inf_time']*1000:.4f} | {r['elm_inf_time']*1000:.4f} |\n")

    report_content += f"""
## 5. Visual Summary
![Real-Time Accuracy](plots/real_time_accuracy.png)
*Figure: Accuracy improvement as the UAV collects more data from the specific flight path.*

## Reproducibility
To recreate these exact results, run the following command:
```bash
python src/aerpaw_processing/paper/learning_experiment.py --seed {seed} --flight {args.flight}
```
"""
    with open("learning_report.md", "w") as f:
        f.write(report_content)
    
    print("\nRefactor and Experiment Complete. See learning_report.md and plots/ directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-stage ML pipeline experiment for UAV RSRP.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Master seed for reproducibility.")
    parser.add_argument("--epochs", type=int, default=50, help="Max epochs for Inception training.")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cpu, cuda, mps).")
    parser.add_argument("--flight", type=str, default=DEFAULT_FLIGHT, help="Specific flight for real-time scenario.")
    args = parser.parse_args()
    run_experiment(args)
