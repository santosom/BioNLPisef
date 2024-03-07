import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw, MolFromSmiles, MolToSmiles

from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from torch import optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from Scripts.build_vocab import WordVocab
from utils import split
from pretrain_trfm import TrfmSeq2seq, evaluate
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit, KFold
from dataset import biodegradeDataset
from torch.optim import adam
from torch.nn.functional import normalize

print('hello this is evaluation')

class LSTM(nn.Module):
    def __init__(self, hide_dim, n_layers):
        self.hide_dim = hide_dim
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=1024, hidden_size=hide_dim, num_layers=n_layers, dropout=.35, batch_first=True)
        self.embedding = nn.Embedding(LEN_VOCAB, 300)
        self.linear = nn.Linear(hide_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.4)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x)
        #linear is the dense layer here
        x = self.linear(x)
        x = self.dropout2(x)
        return torch.sigmoid(x)

pad_index = 0
unk_index = 1
eos_index = 2
sos_index = 3
mask_index = 4

rates = 2**np.arange(7)/80
VOCAB = WordVocab.load_vocab('data/vocab.pkl')
LEN_VOCAB = 45

trfm = TrfmSeq2seq(LEN_VOCAB, 256, LEN_VOCAB, 4)
trfm.load_state_dict(torch.load('Data/smilesPretrained.pkl', map_location=torch.device('cpu')), strict=False)
model = LSTM(124, 3)
#model.load_state_dict(torch.load('Models/trainedRB2'))
def get_inputs(sm):
    seq_len = 220
    sm = sm.split()
    if len(sm)>218:
        print('SMILES is too long ({:d})'.format(len(sm)))
        sm = sm[:109]+sm[-109:]
    ids = [VOCAB.stoi.get(token, unk_index) for token in sm]
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

data = pd.read_csv('Data/RB_Final.csv')
babyData = pd.read_csv('Data/baby_dataset.csv')

def itsevaluation(dataset):
    print('len dataset currently is ', dataset)
    smiles_split = [split(sm) for sm in dataset['processed_smiles'].values]
    smilesID, _ = get_array(smiles_split)
    smiles = trfm.encode(torch.t(smilesID))
    labels = dataset['Class'].values

    testing_data = biodegradeDataset(smiles, labels)
    #batch size defaults to 1
    test_loader = DataLoader(
        dataset=testing_data,
        batch_size=1
    )

    model.eval()
    loss_fn = torch.nn.BCELoss()

    all_labels = []
    all_predictions = []
    runningLoss = []
    test_loss = 0
    correct = 0
    total_validation_records = 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            i += 1
            total_validation_records += labels.size(0)
            outputs = model(inputs)
            print(outputs)
            outputs = outputs.to(torch.float64)
            labels = labels.to(torch.float64)
            labels = labels.unsqueeze(1)
            loss = loss_fn(outputs, labels)
            runningLoss.append(loss.item())

            predicted = torch.round(outputs)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.view(-1).tolist())
            all_predictions.extend(predicted.view(-1).tolist())

            if i % 10 == 0:
                print(i, " loss is ", np.mean(runningLoss))
                print(i, " accuracy is ", correct / i)
                print(" ")
    test_loss = np.mean(runningLoss)
    accuracy = correct / len(testing_data)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='binary')

    print(f'FINAL EVALUATION: LOSS: {test_loss}, ACCURACY: {accuracy}, PRECISION: {precision}, RECALL: {recall}, F1: {f1}')

itsevaluation(data)




