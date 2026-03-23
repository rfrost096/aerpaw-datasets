import logging
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import torch

from aerpaw_processing.paper.dataloader import SignalDataset
from aerpaw_processing.resources.config.config_init import load_config

load_config()


logger = logging.getLogger(__name__)

# LoS PL Model Constants (Section II & IV-A)
C = 299792458  # Speed of light (m/s)
FC = 3.4e9     # Center frequency (Hz) for n77 band
PT_DBM = 10.0  # Total transmit power (dBm)
N_PRB = 273    # Number of physical resource blocks for 100MHz bandwidth
N_SC = 12      # Number of subcarriers per resource block
BS_HEIGHT = 10.0 # Base Station height (m)

def calculate_deterministic_rsrp(x, y, z):
    """
    Calculate deterministic RSRP based on the LoS Path Loss model (Eq. 4 & 5).
    
    Args:
        x, y, z: Coordinates relative to the launch point (m).
                 BS is assumed to be at (0, 0, BS_HEIGHT).
    """
    # 1. Distances and Angles (Eq. 1)
    dh = np.sqrt(x**2 + y**2)
    dv = np.abs(z - BS_HEIGHT)
    d3d = np.sqrt(dh**2 + dv**2)
    
    # Avoid division by zero
    d3d = np.clip(d3d, 1e-6, None)
    
    # Elevation and Azimuth (for antenna gain patterns)
    theta_l = np.arctan2(dh, z - BS_HEIGHT) # Angle from vertical
    phi_l = np.arctan2(y, x)                # Azimuth angle
    
    # 2. Transmit Power per SS resource element (Eq. 5)
    ptx_ss_dbm = PT_DBM - 10 * np.log10(N_PRB * N_SC)
    
    # 3. LoS Path Loss (Eq. 4)
    # PL_LoS = 20*log10(c/4*pi) - 20*log10(fc) - 20*log10(d3d) + G_bs + G_uav
    # Note: We assume G_uav = 0 dBi and a simple directional G_bs for this implementation
    
    # Simplified directional BS antenna gain (120-degree beamwidth)
    phi_3db = 120 * (np.pi / 180)
    g_max_bs = 14.0 # Typical gain in dBi
    g_bs = g_max_bs - np.minimum(12 * (phi_l / phi_3db)**2, 20.0) # 20dB front-to-back ratio
    
    # Constant part of Friis Equation
    pl_const = 20 * np.log10(C / (4 * np.pi)) - 20 * np.log10(FC)
    
    pl_los_db = pl_const - 20 * np.log10(d3d) + g_bs
    
    # 4. SS-RSRP (Eq. 5)
    rsrp_deterministic = ptx_ss_dbm + pl_los_db
    
    return rsrp_deterministic

def test_training_loop():
    dataset = SignalDataset(
        dataset_num=18, flight_name="Yaw 45 Flight", label_col="RSRP_NR_5G"
    )

    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    num_features = len(dataset.feature_cols)
    logger.info(
        f"Dataset loaded. Found {num_features} features and {len(dataset)} samples."
    )
    
    # Find indices for x, y, z in feature columns
    try:
        x_idx = dataset.feature_cols.index('x')
        y_idx = dataset.feature_cols.index('y')
        z_idx = dataset.feature_cols.index('z')
    except ValueError:
        logger.error(f"Required features (x, y, z) not found in {dataset.feature_cols}")
        return

    model = nn.Sequential(
        nn.Linear(num_features, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    logger.info("Starting mock training loop with LoS PL comparison...")
    epochs = 2
    for epoch in range(epochs):
        running_loss = 0.0
        running_los_err = 0.0
        for batch_idx, (features, labels) in enumerate(dataloader):
            optimizer.zero_grad()

            outputs = model(features).squeeze()
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            # Calculate LoS PL model RSRP for comparison
            batch_x = features[:, x_idx].numpy()
            batch_y = features[:, y_idx].numpy()
            batch_z = features[:, z_idx].numpy()
            
            rsrp_los = calculate_deterministic_rsrp(batch_x, batch_y, batch_z)
            rsrp_los_tensor = torch.tensor(rsrp_los, dtype=torch.float32)
            
            los_error = torch.mean((rsrp_los_tensor - labels)**2).item()
            running_los_err += los_error

            if batch_idx == 0:
                logger.info(
                    f"Epoch [{epoch+1}/{epochs}] - ML Loss: {loss.item():.4f}, LoS Model MSE: {los_error:.4f}"
                )

    logger.info("Training loop test completed successfully!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_training_loop()
