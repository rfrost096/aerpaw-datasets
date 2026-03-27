import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from aerpaw_processing.paper.dataloader import UAVFlightDataset
from aerpaw_processing.paper.preprocess_utils import DatasetConfig


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, inception_dim=1):
        super().__init__()
        
        if inception_dim == 1:
            Conv = nn.Conv1d
            MaxPool = nn.MaxPool1d
        elif inception_dim == 2:
            Conv = nn.Conv2d
            MaxPool = nn.MaxPool2d
        elif inception_dim == 3:
            Conv = nn.Conv3d
            MaxPool = nn.MaxPool3d
        else:
            raise ValueError("inception_dim must be 1, 2, or 3")

        # 1x1 conv branch
        self.branch1 = Conv(in_channels, out_channels, kernel_size=1)
        
        # 3x3 conv branch
        self.branch2 = nn.Sequential(
            Conv(in_channels, out_channels, kernel_size=1),
            nn.ReLU(),
            Conv(out_channels, out_channels, kernel_size=3, padding=1)
        )
        
        # 5x5 conv branch
        self.branch3 = nn.Sequential(
            Conv(in_channels, out_channels, kernel_size=1),
            nn.ReLU(),
            Conv(out_channels, out_channels, kernel_size=5, padding=2)
        )
        
        # max pool branch
        self.branch4 = nn.Sequential(
            MaxPool(kernel_size=3, stride=1, padding=1),
            Conv(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        return torch.relu(torch.cat([out1, out2, out3, out4], dim=1))


class InceptionModel(nn.Module):
    def __init__(self, in_channels, inception_dim=1):
        super().__init__()
        self.inception1 = InceptionBlock(in_channels, 8, inception_dim)
        self.inception2 = InceptionBlock(8 * 4, 16, inception_dim)
        
        self.flatten = nn.Flatten()
        # 16 * 4 channels = 64
        self.fc = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.flatten(x)
        return self.fc(x)


class LineOfSightPathLossModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Simple linear layer to model RSRP = A - B * log10(d)
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


class CombinedFlightModel(nn.Module):
    def __init__(self, in_channels, inception_dim=1):
        super().__init__()
        self.inception = InceptionModel(in_channels, inception_dim)
        self.los_model = LineOfSightPathLossModel()
        
        # Learn a weighted combination of the two models' predictions
        self.combiner = nn.Linear(2, 1)

    def forward(self, x_inception, x_los):
        inc_out = self.inception(x_inception)
        los_out = self.los_model(x_los)
        
        # Combine the outputs
        combined = torch.cat([inc_out, los_out], dim=1)
        return self.combiner(combined)


def train_model():
    # Configure dataset
    config = DatasetConfig()
    
    # We want to use datasets in CleanDatasets/data/sigcols_mad_RSRP/
    # If the default ID doesn't match this, we need to enforce it.
    # The default behavior inside DatasetConfig will target this based on preprocess_utils flags.
    
    target_col = "RSRP_NR_5G"
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 10
    inception_dim = 1 # Change to 2 or 3 to test 2D/3D

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading dataset...")
    full_dataset = UAVFlightDataset(
        config=config, 
        target=target_col, 
        inception_dim=inception_dim,
        flag_all_features=True
    )

    sample_x_inception, sample_x_los, _ = full_dataset[0]
    in_channels = sample_x_inception.shape[0]

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(
        f"Dataset split: {train_size} training samples, {val_size} validation samples."
    )
    print(f"Model Inception input channels: {in_channels}, Inception Dim: {inception_dim}\n")

    model = CombinedFlightModel(in_channels=in_channels, inception_dim=inception_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0

        for batch_x_inc, batch_x_los, batch_y in train_loader:
            batch_x_inc = batch_x_inc.to(device)
            batch_x_los = batch_x_los.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            predictions = model(batch_x_inc, batch_x_los)

            if len(batch_y.shape) == 1:
                batch_y = batch_y.unsqueeze(1)

            loss = criterion(predictions, batch_y.float())

            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * batch_x_inc.size(0)

        avg_train_loss = running_train_loss / train_size

        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for batch_x_inc, batch_x_los, batch_y in val_loader:
                batch_x_inc = batch_x_inc.to(device)
                batch_x_los = batch_x_los.to(device)
                batch_y = batch_y.to(device)
                
                predictions = model(batch_x_inc, batch_x_los)

                if len(batch_y.shape) == 1:
                    batch_y = batch_y.unsqueeze(1)

                loss = criterion(predictions, batch_y.float())
                running_val_loss += loss.item() * batch_x_inc.size(0)

        avg_val_loss = running_val_loss / val_size

        print(
            f"Epoch [{epoch + 1:02d}/{num_epochs}] | Train Loss (MSE): {avg_train_loss:.4f} | Val Loss (MSE): {avg_val_loss:.4f}"
        )

    print("\nTraining complete!")


if __name__ == "__main__":
    train_model()
