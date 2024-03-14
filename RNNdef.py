import argparse
import math
import os

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from Scripts import build_vocab

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.rnn = nn.RNN(input_size, hidden_size, num_layers=2, dropout=0.3, nonlinearity='relu')
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.rnn(x)[0]
        x = self.linear(x)
        x = self.dropout(x)
        return torch.sigmoid(x)