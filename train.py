import numpy as np
import pandas as pd
import torch
import torch.nn as nn
#pretty sure that these are just for nice looking graphs. fun though
import matplotlib.pyplot as plt
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw, MolFromSmiles, MolToSmiles
import pickle
import os

from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import math
import argparse


#also need to load in *trained* model. with state_dict() and such
def train(trfm, learning_rate, opt, epochs, batch_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    total_loss = 0
    labels_ids = {'RB': 0, 'NRB': 1}
    n_labels = len(labels_ids)
    trfm.train()


#def eval():
