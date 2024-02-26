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
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from Scripts.build_vocab import WordVocab
from utils import split
from pretrain_trfm import TrfmSeq2seq, evaluate
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit, KFold
from dataset import biodegradeDataset
from torch.nn.functional import normalize

print('hello hello?')

pad_index = 0
unk_index = 1
eos_index = 2
sos_index = 3
mask_index = 4
batch_size = 100

rates = 2**np.arange(7)/80
VOCAB = WordVocab.load_vocab('data/vocab.pkl')
LEN_VOCAB = 45

trfm = TrfmSeq2seq(LEN_VOCAB, 256, LEN_VOCAB, 4)
trfm.load_state_dict(torch.load('Data/smilesPretrained.pkl', map_location=torch.device('cpu')), strict=False)
#change back to all_RB.csv
dataset = pd.read_csv('Data/all_RB.csv')

"""        #embeds = self.word_embeddings(smiles)
        lstm_out, _ = self.lstm(vocab_size, hidden_dim)
        tag_space = self.hidden2tag(lstm_out.view(len(smiles), -1))
        #tag_scores = F.log_softmax(tag_space, dim=1)
        outputs = self.sigmoid(tag_space)"""

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


class LSTM(nn.Module):
    def __init__(self, hide_dim, n_layers):
        self.hide_dim = hide_dim
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=1024, hidden_size=hide_dim, num_layers=n_layers, batch_first=True)
        self.embedding = nn.Embedding(LEN_VOCAB, 300)
        self.linear = nn.Linear(hide_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.linear(x)
        return torch.sigmoid(x)

"""def accuracy(predicted, actual):
    roundedPredict = torch.round(predicted)

    correct = (roundedPredict == actual).float()
    acc = correct.sum() / len(correct)
    return acc"""

def evalTensor(t):
    torch.set_printoptions(threshold=10000)
    print(t)
    for val in t:
        if (val > 1) or (val < 0):
            print('VALUE IN TENSOR OUT OF BOUNDS. VALUE IS ', val)


def calculateAccuracy(outputs, labels):
    # _, predicted = torch.max(outputs, dim=1)
    predicted = outputs
    # use sigmoid to convert to binary
    predicted = torch.round(predicted)
    correct = (predicted == labels).sum().item()
    # debug the comparison
    # print(f'Predicted: {predicted}')
    # print(f'Labels: {labels}')
    return 100 * (correct / len(labels))

def trainLoop(model, epochs, training_data, testing_data, optimizer, criterion, fold):
    epoch_loss = 0.0
    epoch_acc = 0.0
    e_lossListTrain = []
    e_lossListVal = []
    e_accListTrain = []
    e_accListVal = []

    for e in range(epochs):
        epoch_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        all_labels = []
        all_predictions = []
        epoch_records = 0
        for batch, (inputs, labels) in enumerate(training_data):
            #update model based on loss
            epoch_records += labels.size(0)
            model.train()
            optimizer.zero_grad()

            inputs = normalize(inputs, p=1.0, dim=0)
            outputs = model(inputs)
            outputs = outputs.to(torch.float32)
            labels = labels.to(torch.float32)
            labels = labels.unsqueeze(1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            #training acc and loss
            model.eval()
            outputs = model(inputs)
            outputs = outputs.to(torch.float32)
            test_loss = criterion(outputs, labels)
            accuracy = calculateAccuracy(outputs, labels)
            print(f'    Batch: {batch} Loss: {test_loss.item():.4f} Accuracy: {accuracy:.2f}%')

            epoch_loss += loss.item()
            e_lossListTrain.append(loss.item())

            # Convert outputs to binary predictions
            preds = outputs.round()  # Assuming sigmoid activation at the output; adjust if necessary

            # Update total and correct predictions for accuracy calculation
            total_predictions += labels.size(0)
            correct_predictions += (preds == labels).sum().item()

            # Store labels and predictions for F1 score calculation
            all_labels.extend(labels.view(-1).tolist())
            all_predictions.extend(preds.view(-1).tolist())

        # Calculate epoch accuracy on training data
        epoch_acc = correct_predictions / total_predictions
        print("EPOCH ACC ON TRAINING: ", epoch_acc)
        e_accListTrain.append(epoch_acc)

        #Calculate epoch accuracy on val data
        model.eval()
        test_loss = 0
        correct = 0
        total_validation_records = 0

        with torch.no_grad():
            for inputs, labels in testing_data:
                total_validation_records += labels.size(0)
                inputs = normalize(inputs, p=1.0, dim=0)
                outputs = model(inputs)
                outputs = outputs.to(torch.float32)
                labels = labels.to(torch.float32)
                labels = labels.unsqueeze(1)
                test_loss += criterion(outputs, labels)
                predicted = torch.round(outputs)
                correct += (predicted == labels).sum().item()
        test_loss /= len(testing_data.dataset)
        accuracy = 100.0 * correct / len(testing_data.dataset)
        e_accListVal.append(accuracy)
        e_lossListVal.append(test_loss)

        # Calculate precision, recall, and F1 score
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='binary')
        print(f'  Epoch {e} Training - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f} Records: {epoch_records}')
        print(f'  Epoch {e} Validation - Loss: {test_loss}, Accuracy: {accuracy}')

    plt.figure(1)
    print('epoch num is ', epochs)
    labelName = str(fold) + 'train_acc'
    plt.plot(e_accListTrain, label=labelName)
    plt.legend()
    graphName = 'Graphs/' + str(fold) + 'TrainAcc'
    plt.savefig(graphName)
    plt.clf()

    labelName = str(fold) + 'val_acc'
    plt.plot(e_accListVal, label=labelName)
    plt.legend()
    graphName = 'Graphs/' + str(fold) + 'ValAcc'
    plt.savefig(graphName)
    plt.clf()

    plt.figure(2)
    plt.plot(e_lossListTrain, label='train_loss')
    plt.legend()
    graphName = 'Graphs/' + str(fold) + 'TrainLoss'
    plt.savefig(graphName)
    plt.clf()

    plt.figure(2)
    plt.plot(e_lossListVal, label='val_loss')
    plt.legend()
    graphName = 'Graphs/' + str(fold) + 'ValLoss'
    plt.savefig(graphName)
    plt.clf()

    test_loss /= len(testing_data.dataset)
    accuracy = 100.0 * correct / len(testing_data.dataset)
    print('Validation records: ', total_validation_records)
    print(f'Validation: Fold {fold} Average loss: {test_loss:.4f} Accuracy: {correct}/{len(testing_data.dataset)} ({accuracy:.2f}%)')
    # print the size of the testing data

    print(" ")

"""    model.eval()
    test_loss = 0
    correct = 0
    total_validation_records = 0
    with torch.no_grad():
        for inputs, labels in testing_data:
            total_validation_records += labels.size(0)
            inputs = normalize(inputs, p=1.0, dim=0)
            outputs = model(inputs)
            outputs = outputs.to(torch.float32)
            labels = labels.to(torch.float32)
            labels = labels.unsqueeze(1)
            test_loss += criterion(outputs, labels)
            predicted = torch.round(outputs)
            correct += (predicted == labels).sum().item()"""

def formatAndFold():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smiles_split = [split(sm) for sm in dataset['processed_smiles'].values]
    smilesID, _ = get_array(smiles_split)
    smiles_train = trfm.encode(torch.t(smilesID))
    labels_train = dataset['Class'].values

    # critical hyperparameters
    epoch = 4
    ksplits = 3
    learning_rate = 0.004

    # Initialize the model and optimizer
    model = LSTM(1024, 2)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.BCELoss()

    fold = 0
    kfold = StratifiedKFold(n_splits=ksplits, shuffle=True)
    for train_index, test_index in kfold.split(smiles_train, labels_train):
        fold = fold + 1
        training_data = biodegradeDataset(smiles_train[train_index], labels_train[train_index])
        testing_data = biodegradeDataset(smiles_train[test_index], labels_train[test_index])

        # Define the data loaders for the current fold
        train_loader = DataLoader(
            dataset=training_data,
            batch_size=batch_size
        )
        test_loader = DataLoader(
            dataset=testing_data,
            batch_size=batch_size
        )

        # change the number of epoch back to 20
        trainLoop(model, epoch, train_loader, test_loader, optimizer, loss_fn, fold)

formatAndFold()
