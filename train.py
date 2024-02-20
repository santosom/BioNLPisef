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
from classifier import classify
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit, KFold
from dataset import biodegradeDataset

pad_index = 0
unk_index = 1
eos_index = 2
sos_index = 3
mask_index = 4
batch_size = 64

vocab = WordVocab.load_vocab('data/vocab.pkl')
len_vocab = 45
print("vocab length: ", len(vocab))

print("hello world")
trfm = TrfmSeq2seq(len_vocab, 256, len_vocab , 4)
trfm.load_state_dict(torch.load('Data/smilesPretrained.pkl', map_location=torch.device('cpu')), strict=False)

#decrease this learning rate if the model performs poorly
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

def testBiodegrade(smilesTrain, labelsTrain, smilesTest, labelsTest, n_repeats): #replace MLP classifier with custom classifier. I hate this guy
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

def customBiodegrade(model, device, train_loader, optimizer, epoch):
    customModel = model
    learning_rate = .01
    optimizer = optim.Adam(trfm.parameters(), lr=learning_rate)
    loss_fn = torch.nn.BCELoss()
    n_epochs = 40
    #define training loop
    for epic in range(epoch):
        for idx, (inputs, labels) in enumerate(train_loader):
            customModel.train()
            print(f'In the loop, input type is {type(inputs)} and label type is {type(labels)}')
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

#trfm, learning_rate, opt, epochs, batch_size
def _train():
    dataset = pd.read_csv('Data/all_RB.csv')
    smiles_split = [split(sm) for sm in dataset['processed_smiles'].values]
    smilesID, _ = get_array(smiles_split)
    smiles_train = trfm.encode(torch.t(smilesID))
    print(f'{smiles_train.shape}, the datatype is {type(smiles_train)}')
    labels_train = dataset['Class'].values

    #you need to shuffle before you split because otherwise you'll get training data with only biodegradable chemicals and validation data with only non biodegradable data, and stuff like that
    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    overallAverageList = []
    for train_index, test_index in kfold.split(smiles_train, labels_train):
        x_train, x_val, y_train, y_val = smiles_train[train_index], smiles_train[test_index], labels_train[train_index], labels_train[test_index]
        currentAverageDict = testBiodegrade(x_train, y_train, x_val, y_val, 20)
        print(currentAverageDict)
        overallAverageList.append(currentAverageDict['auc mean'])

    overallAverage = np.mean(overallAverageList)
    print(f"Process is done! The overall mean score is {overallAverage}")

def _train2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #need to go back and replace with Data/all_RB.csv
    dataset = pd.read_csv('Data/RB_val.csv')
    smiles_split = [split(sm) for sm in dataset['processed_smiles'].values]
    smilesID, _ = get_array(smiles_split)
    smiles_train = trfm.encode(torch.t(smilesID))
    labels_train = dataset['Class'].values

    kfold = StratifiedKFold(n_splits=10, shuffle=True)

    for train_index, test_index in kfold.split(smiles_train, labels_train):
        trainingData = biodegradeDataset(smiles_train[train_index], labels_train[train_index])
        testingData = biodegradeDataset(smiles_train[test_index], labels_train[test_index])


        # Define the data loaders for the current fold
        train_loader = DataLoader(
            dataset=trainingData,
            batch_size=batch_size
        )
        test_loader = DataLoader(
            dataset=testingData,
            batch_size=batch_size
        )

        # Initialize the model and optimizer
        model = classify(1024, 64).to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
        print('intialized model and optimizer')

        # Train the model on the current fold
        for epoch in range(1, 11):
            customBiodegrade(model, device, train_loader, optimizer, epoch)

        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += nn.functional.nll_loss(output, target, reduction="sum").item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100.0 * correct / len(test_loader.dataset)

_train2()