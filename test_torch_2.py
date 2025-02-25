import torch
import torch.nn as nn
import torch.optim as optim

class PreferenceModel(nn.Module):
    def __init__(self, input_size):
        super(PreferenceModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))  # Probabilité de "like"