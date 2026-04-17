"""
Showcase Experiment: ELM vs Inception for On-board UAV Learning

This script implements an experiment that mimics a real-world UAV flight path,
highlighting the architectural differences between an iterative backpropagation 
model (Inception) and a single-step analytical model (ELM). 

Evaluation Criteria:
1. Memory Footprint
2. Pre-flight Training Latency
3. In-flight Inference Latency
4. On-board Learning Scenario (Mid-flight adaptation)
"""

import time
import argparse
import torch
from torch.utils.data import DataLoader, Subset

from aerpaw_processing.paper.inception_dataloader import InceptionDataset
from aerpaw_processing.paper.preprocess_utils import DatasetConfig
from aerpaw_processing.paper.inception_model import InceptionTime, train_one_epoch, build_optimizer_and_scheduler, evaluate
from aerpaw_processing.paper.elm_model import ExtremeLearningMachine


def train_elm_analytical(model: ExtremeLearningMachine, dataset: InceptionDataset, indices: list[int]):
    """
    Train ELM using Moore-Penrose pseudoinverse (Analytical Solution).
    This computes the exact output weights in a single mathematical step, 
    bypassing the need for iterative backpropagation.
    """
    model.eval() # Ensure we don't track gradients
    device = next(model.parameters()).device
    
    H_list = []
    T_list = []
    
    loader = DataLoader(Subset(dataset, indices), batch_size=64, shuffle=False)
    with torch.no_grad():
        for x, y_seq, mask in loader:
            x = x.to(device)
            y_seq = y_seq.to(device)
            mask = mask.to(device)
            
            # 1. Map input to hidden space (Random projection)
            h = model.hidden_layer(x)
            h = model.activation(h) # (B, hidden_size, R_max)
            
            # 2. Extract features at masked positions
            h = h.permute(0, 2, 1) # (B, R_max, hidden_size)
            h_masked = h[mask]     # (N, hidden_size)
            t_masked = y_seq[mask] # (N,)
            
            H_list.append(h_masked)
            T_list.append(t_masked)
            
    if not H_list:
        return
        
    H_mat = torch.cat(H_list, dim=0)
    T_mat = torch.cat(T_list, dim=0).unsqueeze(1)
    
    # 3. Add bias column
    ones = torch.ones(H_mat.size(0), 1, device=device)
    H_mat_bias = torch.cat([H_mat, ones], dim=1)
    
    # 4. Compute output weights using generalized inverse: beta = H_dagger * T
    W_b = torch.linalg.pinv(H_mat_bias) @ T_mat
    
    # 5. Extract weights and bias and assign to model
    W = W_b[:-1, :].t() # (1, hidden_size)
    b = W_b[-1, :]      # (1,)
    
    model.output_layer.weight.data = W.unsqueeze(-1) # (1, hidden_size, 1)
    model.output_layer.bias.data = b


def get_model_size_mb(model: torch.nn.Module) -> float:
    """Calculate the memory footprint of the model weights."""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)


def main():
    parser = argparse.ArgumentParser(description="UAV Edge Computing Showcase")
    parser.add_argument("--num-pre-epochs", type=int, default=1, help="Number of pre-flight training epochs")
    parser.add_argument("--num-in-epochs", type=int, default=1, help="Number of in-flight retraining epochs")
    args = parser.parse_args()

    print("="*60)
    print(" UAV Edge Computing Showcase: ELM vs InceptionTime ")
    print("="*60)
    
    # Ensure dataset is available
    print("\n[Initializing] Loading RSRP Dataset...")
    config = DatasetConfig()
    try:
        dataset = InceptionDataset(config=config)
    except Exception as e:
        print(f"Dataset loading failed: {e}. Please ensure dataset exists or config is valid.")
        return
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device selected: {device}")
    
    # Instantiate models
    elm = ExtremeLearningMachine(in_channels=8, r_max=dataset.r_max, hidden_size=128).to(device)
    inception = InceptionTime(in_channels=8, r_max=dataset.r_max).to(device)
    
    # 1. Memory Footprint
    print("\n--- 1. Memory Footprint ---")
    elm_size = get_model_size_mb(elm)
    inc_size = get_model_size_mb(inception)
    print(f"  Inception Model : {inc_size:.4f} MB ({sum(p.numel() for p in inception.parameters()):,} params)")
    print(f"  ELM Model       : {elm_size:.4f} MB ({sum(p.numel() for p in elm.parameters()):,} params)")
    
    # Splits
    total_samples = len(dataset)
    train_size = int(total_samples * 0.8)
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, total_samples))
    
    print(f"\nTotal Dataset Samples: {total_samples}")
    print(f"Pre-flight Training Samples: {len(train_indices)}")
    
    # 2. Pre-flight Training
    print("\n--- 2. Pre-flight Training ---")
    
    # ELM analytical training
    print("  Training ELM (Analytical Matrix Inversion)...")
    start_time = time.time()
    train_elm_analytical(elm, dataset, train_indices)
    elm_preflight_time = time.time() - start_time
    print(f"  -> ELM Time: {elm_preflight_time:.4f} seconds (Converged in 1 step)")
    
    # Inception training
    print(f"  Training Inception (Iterative Backpropagation - {args.num_pre_epochs} Epoch Simulation)...")
    inc_loader = DataLoader(Subset(dataset, train_indices), batch_size=64, shuffle=True)
    optimizer, scheduler = build_optimizer_and_scheduler(inception, total_steps=len(inc_loader) * args.num_pre_epochs)
    
    start_time = time.time()
    for _ in range(args.num_pre_epochs):
        train_one_epoch(inception, inc_loader, optimizer, scheduler, device)
    inc_preflight_time = time.time() - start_time
    print(f"  -> Inception Time: {inc_preflight_time:.4f} seconds (For {args.num_pre_epochs} epochs)")
    
    # 3. In-flight Inference Latency
    print("\n--- 3. In-flight Inference Latency ---")
    print("  Simulating predictions for incoming telemetry coordinates...")
    single_loader = DataLoader(Subset(dataset, [val_indices[0]]), batch_size=1)
    x, _, _ = next(iter(single_loader))
    x = x.to(device)
    
    # Warmup
    for _ in range(10):
        elm(x)
        inception(x)
        
    runs = 100
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(runs):
            elm(x)
    elm_inf_time = ((time.time() - start_time) / runs) * 1000 # ms
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(runs):
            inception(x)
    inc_inf_time = ((time.time() - start_time) / runs) * 1000 # ms
    
    print(f"  Inception Inference Latency: {inc_inf_time:.2f} ms / sample")
    print(f"  ELM Inference Latency:       {elm_inf_time:.2f} ms / sample")
    
    # 4. On-board Learning Scenario
    print("\n--- 4. The 'On-board Learning' Scenario ---")
    print("  UAV encountered sudden environmental changes.")
    print("  Triggering mid-flight retraining on the last 500 spatial coordinates...")
    
    recent_indices = val_indices[-min(500, len(val_indices)):]
    
    # ELM
    start_time = time.time()
    train_elm_analytical(elm, dataset, recent_indices)
    elm_retrain_time = time.time() - start_time
    print(f"  -> ELM Retraining Time:       {elm_retrain_time:.4f} seconds")
    
    # Inception
    recent_loader = DataLoader(Subset(dataset, recent_indices), batch_size=64, shuffle=True)
    start_time = time.time()
    for _ in range(args.num_in_epochs):
        train_one_epoch(inception, recent_loader, optimizer, scheduler, device)
    inc_retrain_time = time.time() - start_time
    print(f"  -> Inception Retraining Time: {inc_retrain_time:.4f} seconds ({args.num_in_epochs} epochs)")
    
    # 5. Model Accuracy
    print("\n--- 5. Model Accuracy ---")
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=64, shuffle=False)
    
    elm_metrics = evaluate(elm, val_loader, device)
    inc_metrics = evaluate(inception, val_loader, device)
    
    print(f"  Inception RMSE: {inc_metrics['RMSE']:.4f} dBm | MAE: {inc_metrics['MAE']:.4f}")
    print(f"  ELM RMSE:       {elm_metrics['RMSE']:.4f} dBm | MAE: {elm_metrics['MAE']:.4f}")
    
    print("\n" + "="*60)
    print(" Experiment Complete ")
    print("="*60)

if __name__ == '__main__':
    main()
