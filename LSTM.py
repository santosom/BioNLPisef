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
from torch.nn.functional import normalize

print('hello hello?')

pad_index = 0
unk_index = 1
eos_index = 2
sos_index = 3
mask_index = 4
batch_size = 64

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
        self.lstm = nn.LSTM(input_size=1024, hidden_size=hide_dim, num_layers=n_layers, dropout=.4, batch_first=True)
        self.embedding = nn.Embedding(LEN_VOCAB, 300)
        self.linear = nn.Linear(hide_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.4)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x)
        #linear is the dense layer here
        x = self.linear(x)
        x = self.dropout2(x)
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

def trainLoop(model, epochs, training_data, testing_data, optimizer, criterion, fold, scheduler):
    epoch_loss = 0.0
    epoch_acc = 0.0
    e_lossListTrain = []
    e_lossListVal = []
    e_accListTrain = []
    e_accListVal = []

    e_precisionListTrain = []
    e_precisionListVal = []
    e_recallListTrain = []
    e_recallListVal = []
    e_F1ListTrain = []
    e_F1ListVal = []

    for e in range(epochs):
        epoch_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        all_labels = []
        all_predictions = []
        epoch_records = 0

        lossListForEpoch = []
        for batch, (inputs, labels) in enumerate(training_data):
            #update model based on loss
            epoch_records += labels.size(0)
            model.train()
            optimizer.zero_grad()

            inputs = normalize(inputs, p=2.0, dim=0)
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
            #print(f'    Batch: {batch} Loss: {test_loss.item():.4f} Accuracy: {accuracy:.2f}%')

            epoch_loss += loss.item()
            lossListForEpoch.append(loss.item())

            # Convert outputs to binary predictions
            preds = outputs.round()  # Assuming sigmoid activation at the output; adjust if necessary

            # Update total and correct predictions for accuracy calculation
            total_predictions += labels.size(0)
            correct_predictions += (preds == labels).sum().item()

            # Store labels and predictions for F1 score calculation
            all_labels.extend(labels.view(-1).tolist())
            all_predictions.extend(preds.view(-1).tolist())

        before_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        after_lr = optimizer.param_groups[0]["lr"]
        print("  Epoch %d: SGD lr %.8f -> %.8f" % (e, before_lr, after_lr))

        # Calculate epoch accuracy on training data
        epoch_acc = correct_predictions / total_predictions
        #print("EPOCH ACC ON TRAINING: ", epoch_acc)
        e_accListTrain.append(epoch_acc)

        aveLossForEpoch = np.mean(lossListForEpoch)
        e_lossListTrain.append(aveLossForEpoch)

        #Calculate epoch accuracy on val data
        model.eval()
        test_loss = 0
        correct = 0
        total_validation_records = 0

        valLabels = []
        valPredictions = []
        valRunningLoss = []

        with torch.no_grad():
            for inputs, labels in testing_data:
                total_validation_records += labels.size(0)
                inputs = normalize(inputs, p=2.0, dim=0)
                outputs = model(inputs)
                outputs = outputs.to(torch.float32)
                labels = labels.to(torch.float32)
                labels = labels.unsqueeze(1)
                l = criterion(outputs, labels)
                valRunningLoss.append(l.item())


                predicted = torch.round(outputs)
                correct += (predicted == labels).sum().item()

                # Store labels and predictions for F1 score calculation
                valLabels.extend(labels.view(-1).tolist())
                valPredictions.extend(predicted.view(-1).tolist())
        test_loss = np.mean(valRunningLoss)
        accuracy = correct / len(testing_data.dataset)
        e_accListVal.append(accuracy)
        e_lossListVal.append(test_loss)
        #print('labels is ', len(valLabels), ' while predictions is ', len(valPredictions),'. they are types ', type(valLabels), type(valPredictions))

        # Calculate precision, recall, and F1 score
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='binary')
        precisionV, recallV, f1V, _ = precision_recall_fscore_support(valLabels, valPredictions, average='binary')
        #bc i'm getting nauesous
        if (e%10==0):
            print(
                f'  Epoch {e} Training - Loss: {aveLossForEpoch:.4f}, Accuracy: {epoch_acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f} Records: {epoch_records}')
            print(
                f'             Epoch {e} Validation - Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precisionV:.4f}, Recall: {recallV:.4f}, F1 Score: {f1V:.4f}')

        if (e > (epochs-10)):
            e_precisionListTrain.append(precision)
            e_precisionListVal.append(precisionV)
            e_recallListTrain.append(recall)
            e_recallListVal.append(recallV)
            e_F1ListTrain.append(f1)
            e_F1ListVal.append(f1V)

    test_loss /= len(testing_data.dataset)
    accuracy = correct / len(testing_data.dataset)
    print('Validation records: ', total_validation_records)
    print(f'Validation: Fold {fold} Average loss: {test_loss:.4f} Accuracy: {correct}/{len(testing_data.dataset)} ({accuracy:.2f}%)')
    # print the size of the testing data

    plt.figure(1)
    #print('epoch num is ', epochs)
    plt.plot(e_accListTrain, label='train_acc')
    plt.plot(e_accListVal, label='val_acc')
    plt.legend()
    plt.ylim(0, 1)
    graphName = 'Graphs/' + str(fold) + 'TrainAcc'
    plt.savefig(graphName)
    plt.clf()

    plt.figure(2)
    plt.plot(e_lossListTrain, label='train_loss')
    plt.plot(e_lossListVal, label='val_loss')
    plt.legend()
    graphName = 'Graphs/' + str(fold) + 'Loss'
    plt.savefig(graphName)
    plt.clf()

    plt.figure(3)
    plt.plot(e_recallListTrain, label='Train Recall')
    plt.plot(e_recallListVal, label='Validation Recall')
    plt.legend()
    plt.ylim(0, 1)
    graphName = 'Graphs/' + str(fold) + 'Recall'
    plt.savefig(graphName)
    plt.clf()

    plt.figure(4)
    plt.plot(e_precisionListTrain, label='Train Precision')
    plt.plot(e_precisionListVal, label='Validation Precision')
    plt.legend()
    plt.ylim(0, 1)
    graphName = 'Graphs/' + str(fold) + 'Precision'
    plt.savefig(graphName)
    plt.clf()

    plt.figure(5)
    plt.plot(e_F1ListTrain, label='Train F1')
    plt.plot(e_F1ListVal, label='Validation F1')
    plt.legend()
    plt.ylim(0, 1)
    graphName = 'Graphs/' + str(fold) + 'F1'
    plt.savefig(graphName)
    plt.clf()

    return test_loss, (correct/len(testing_data.dataset))

def formatAndFold():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smiles_split = [split(sm) for sm in dataset['processed_smiles'].values]
    smilesID, _ = get_array(smiles_split)
    smiles_train = trfm.encode(torch.t(smilesID))
    labels_train = dataset['Class'].values

    # critical hyperparameters
    epoch = 300
    ksplits = 3
    #learning_rate = 0.000001
    #learning_rate = 0.0001
    learning_rate = 0.0001
    learning_rate = 0.00008
    allAveLoss = []
    allAveAcc = []

    # Initialize the model and optimizer

    fold = 0
    #shuffle is currently false, was previously true
    kfold = StratifiedKFold(n_splits=ksplits, shuffle=True)
    for train_index, test_index in kfold.split(smiles_train, labels_train):
        #changed from 64
        model = LSTM(32, 3)
        #change to Adam
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=.8, total_iters=300)
        loss_fn = torch.nn.BCELoss()

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
        loopLoss, loopAcc = trainLoop(model, epoch, train_loader, test_loader, optimizer, loss_fn, fold, scheduler)
        allAveLoss.append(loopLoss)
        allAveAcc.append(loopAcc)

    meanLoss = np.mean(allAveLoss)
    meanAcc = np.mean(allAveAcc)
    SDLoss = np.std(allAveLoss)
    SDAcc = np.std(allAveAcc)

    print(" ")
    print(f'MEAN LOSS FOR ALL FOLDS IS {meanLoss}, SD {SDLoss}. MEAN ACCURACY IS {meanAcc}, SD {SDAcc}.')
formatAndFold()
