from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class WaterDataset(Dataset):
    def __init__(self, csv_path):
        super().__init__()  # inherit from Dataset
        df = pd.read_csv(csv_path)
        self.data = df.to_numpy()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # return the data and label
        features = self.data[idx:, :-1]  # all columns except the last one
        label = self.data[idx:, -1]  # last column
        return features, label


from torch.utils.data import DataLoader
import torch.nn as nn

# Create the dataset
dataset_train = WaterDataset("water_train.csv")

# Create the DataLoader
dataloader_train = DataLoader(
    dataset_train,
    batch_size=2,
    shuffle=True,
)

# Example: get a batch
features, labels = next(iter(dataloader_train))
print(f"Features: {features},\nLabels: {labels}")

# Sequential model definition
net = nn.Sequential(
    nn.Linear(9, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid(),
)


# Class-based model definition
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(9, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.sigmoid(self.fc3(x))
        return x


net_class = Net()

# Print the model
print(net_class)