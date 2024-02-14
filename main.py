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

from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
import math
import argparse

#our first step is going to be to load in and pretrain the model we'll hopefully be using
gpu = torch.cuda.is_available()
print("HAS CUDA: ", gpu)
device = 'cpu'
if gpu:
    device = 'cuda'


model = torch.load('Data/smilesPretrained.pkl', map_location=device)
print("hello world!")

df = pd.read_table('Data/chembl_24_1_chemreps.txt.gz')
smiles  = df['canonical_smiles'].values
to_drop = []
for i,sm in enumerate(smiles):
    if len(sm)>100:
        to_drop.append(i)
    if df['chembl_id'][i]=='CHEMBL1201364':
        to_drop.append(i)

df_dropped = df.drop(to_drop)
df_dropped = df_dropped.drop(['standard_inchi', 'standard_inchi_key'], axis=1)
L = len(df_dropped)
df_dropped.head()

df_dropped.to_csv('Data/chembl_24.csv', index=False)

from pretrain_trfm import TrfmSeq2seq
from build_vocab import WordVocab
from utils import split

pad_index = 0
unk_index = 1
eos_index = 2
sos_index = 3
mask_index = 4

vocab = WordVocab.load_vocab('Data/vocab.pkl')

trfm = TrfmSeq2seq(len(vocab), 256, len(vocab), 4)
trfm.load_state_dict(torch.load('Data/smilesPretrained.pkl', map_location=torch.device('cpu')), strict=False)
trfm.eval()
print(trfm)
#print("input size = ", str(len(vocab)), ", 256, ", str(len(vocab)), ", 4")
print("see you later world")

#decrease this learning rate if the model performs poorly
learning_rate = .01
optimizer = optim.Adam(trfm.parameters(), lr = learning_rate)
