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

from sklearn.metrics import roc_auc_score
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import math
import argparse
from build_vocab import WordVocab
from utils import split
from pretrain_trfm import TrfmSeq2seq
from sklearn.neural_network import MLPClassifier

pad_index = 0
unk_index = 1
eos_index = 2
sos_index = 3
mask_index = 4

vocab = WordVocab.load_vocab('data/vocab.pkl')

print("hello world")
trfm = TrfmSeq2seq(len(vocab), 256, len(vocab), 4)
trfm.load_state_dict(torch.load('Data/smilesPretrained.pkl', map_location=torch.device('cpu')), strict=False)

#decrease this learning rate if the model performs poorly
learning_rate = .01
optimizer = optim.Adam(trfm.parameters(), lr = learning_rate)
df_train = pd.read_csv('Data/RB_train.csv')
df_val = pd.read_csv('Data/RB_val.csv')

#this feels so random...?
rates = 2**np.arange(7)/80
def get_inputs(sm):
    seq_len = 220
    sm = sm.split()
    if len(sm)>218:
        print('SMILES is too long ({:d})'.format(len(sm)))
        sm = sm[:109]+sm[-109:]
    ids = [vocab.stoi.get(token, unk_index) for token in sm]
    ids = [sos_index] + ids + [eos_index]
    seg = [1]*len(ids)
    padding = [pad_index]*(seq_len - len(ids))
    ids.extend(padding), seg.extend(padding)
    return ids, seg

def get_array(smiles):
    x_id, x_seg = [], []
    for sm in smiles:
        a,b = get_inputs(sm)
        x_id.append(a)
        x_seg.append(b)
    return torch.tensor(x_id), torch.tensor(x_seg)

def testBiodegrade(X_train, X_val, Y_train, Y_val, rate, n_repeats):
    auc = np.empty(n_repeats)
    for i in range(n_repeats):
        clf = MLPClassifier(max_iter=1000)
        clf.fit(X_train, Y_train)
        y_score = clf.predict_proba(X_val)
        auc[i] = roc_auc_score(Y_val, y_score[:, 1])
    ret = {}
    ret['auc mean'] = np.mean(auc)
    ret['auc std'] = np.std(auc)
    return ret

#also need to load in *trained* model. with state_dict() and such
#trfm, learning_rate, opt, epochs, batch_size
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    total_loss = 0
    #labels_ids = {'RB': 0, 'NRB': 1}
    #n_labels = len(labels_ids)
    trfm.train()

    #encode (?)
    smiles_split = [split(sm) for sm in df_train['processed_smiles'].values]
    smilesID, _ = get_array(smiles_split)
    smiles_train = trfm.encode(torch.t(smilesID))
    print(smiles_train.shape)
    #x is the smiles
    smiles_split = [split(sm) for sm in df_val['processed_smiles'].values]
    smilesID, _ = get_array(smiles_split)
    smiles_val = trfm.encode(torch.t(smilesID))
    print(smiles_val.shape)
    labels_train, labels_val = df_train['Class'].values, df_val['Class'].values

    print(testBiodegrade(smiles_train, smiles_val, labels_train, labels_val, rates, 20))



    #put into a dataset

    #define the training loop

train()
#def eval():
