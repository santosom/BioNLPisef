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
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit, KFold

pad_index = 0
unk_index = 1
eos_index = 2
sos_index = 3
mask_index = 4

vocab = WordVocab.load_vocab('data/vocab.pkl')
len_vocab = 45
print("vocab length: ", len(vocab))

print("hello world")
trfm = TrfmSeq2seq(len_vocab, 256, len_vocab , 4)
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

def testBiodegrade(smilesTrain, labelsTrain, smilesTest, labelsTest, n_repeats): #go through and comment this
    print('biodegrade method has been called')
    auc = np.empty(n_repeats)
    for i in range(n_repeats):
        clf = MLPClassifier(max_iter=1000)
        clf.fit(smilesTrain, labelsTrain)
        y_score = clf.predict_proba(smilesTest)
        auc[i] = roc_auc_score(labelsTest, y_score[:, 1])
    ret = {}
    ret['auc mean'] = np.mean(auc)
    ret['auc std'] = np.std(auc)
    return ret

#trfm, learning_rate, opt, epochs, batch_size
def _train():
    dataset = pd.read_csv('Data/all_RB.csv')
    smiles_split = [split(sm) for sm in dataset['processed_smiles'].values]
    smilesID, _ = get_array(smiles_split)
    smiles_train = trfm.encode(torch.t(smilesID))
    print(f'{smiles_train.shape}, the datatype is {type(smiles_train)}')
    labels_train = dataset['Class'].values

    kfold = StratifiedKFold(n_splits=3, shuffle=True)
    #one of the issues with k fold here is that we don't really have a singular dataset at the end of the encoding, since we need to use the encoded smiles taken above. at the same time, we need to base the train/test split off of the distribution of classes
    #if we were to implement it, here is where we would want to add a for-loop iterating through all of the sets and using them with testBiodegradable()
    for train_index, test_index in kfold.split(smiles_train, labels_train):
        x_train, x_val, y_train, y_val = smiles_train[train_index], smiles_train[test_index], labels_train[train_index], labels_train[test_index]
        print(testBiodegrade(x_train, y_train, x_val, y_val, 50))





"""def iterateTest():

    kfold = StratifiedKFold(n_splits=5, shuffle=True)
    sets = kfold.split(dataset,dataset['Class'])

    overallscore = []
    for train_index, test_index in sets:

        print('hiiiiiiiiiiiiiiiiiiiii')
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]

        currentAve = _train(X_train, X_val)
        overallscore.append(currentAve)
        print(f'current ave is {currentAve}')

    overallAverage = np.mean(overallscore)
    print("OVERALL AVERAGE: ", overallAverage)"""

_train()







