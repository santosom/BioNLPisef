import torch.nn as nn
import torch.nn.functional as F


class classify(nn.Module):
    def __init__(self, in_size, hidden_size):
        super().__init__()
        self.layer1 = nn.Linear(in_size, hidden_size)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.sigmoid(self.output(x))
        return x
