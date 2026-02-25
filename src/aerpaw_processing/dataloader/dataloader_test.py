import logging
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from aerpaw_processing.dataloader.dataloader import SignalDataset
from aerpaw_processing.resources.config.config_init import load_config

load_config()


logger = logging.getLogger(__name__)


def test_training_loop():
    dataset = SignalDataset(
        dataset_num=18, flight_name="Yaw 45 Flight", label_col="rsrp"
    )

    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    num_features = len(dataset.feature_cols)
    logger.info(
        f"Dataset loaded. Found {num_features} features and {len(dataset)} samples."
    )

    model = nn.Sequential(
        nn.Linear(num_features, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    logger.info("Starting mock training loop...")
    epochs = 2
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (features, labels) in enumerate(dataloader):
            optimizer.zero_grad()

            outputs = model(features).squeeze()

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx == 0:
                logger.info(
                    f"Epoch [{epoch+1}/{epochs}] - First batch loss: {loss.item():.4f}"
                )

    logger.info("Training loop test completed successfully!")


if __name__ == "__main__":
    test_training_loop()
