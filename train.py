import numpy as np
import pandas as pd
import torch
import torch.nn as nn
# pretty sure that these are just for nice looking graphs. fun though
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

import Scripts.build_vocab
from Scripts import build_vocab
from extend_trfm import ExtendedTrfmSeq2seqForBinaryClassification
from utils import split
from pretrain_trfm import TrfmSeq2seq
from classifier import classify
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit, KFold
from dataset import biodegradeDataset
from Scripts.build_vocab import WordVocab

pad_index = 0
unk_index = 1
eos_index = 2
sos_index = 3
mask_index = 4
batch_size = 64

vocab = WordVocab.load_vocab('Data/vocab.pkl')
len_vocab = 45
print("vocab length: ", len(vocab))

print("hello world")
trfm = TrfmSeq2seq(len_vocab, 256, len_vocab, 4)
trfm.load_state_dict(torch.load('Data/smilesPretrained.pkl', map_location=torch.device('cpu')), strict=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# decrease this learning rate if the model performs poorly
df_train = pd.read_csv('Data/RB_train.csv')
df_val = pd.read_csv('Data/RB_val.csv')

# this feels so random...?
rates = 2 ** np.arange(7) / 80


def get_inputs(sm):
    seq_len = 220
    sm = sm.split()
    if len(sm) > 218:
        print('SMILES is too long ({:d})'.format(len(sm)))
        sm = sm[:109] + sm[-109:]
    ids = [vocab.stoi.get(token, unk_index) for token in sm]
    ids = [sos_index] + ids + [eos_index]
    seg = [1] * len(ids)
    padding = [pad_index] * (seq_len - len(ids))
    ids.extend(padding), seg.extend(padding)
    return ids, seg


def get_array(smiles):
    x_id, x_seg = [], []
    for sm in smiles:
        a, b = get_inputs(sm)
        x_id.append(a)
        x_seg.append(b)
    return torch.tensor(x_id), torch.tensor(x_seg)


def testBiodegrade(smilesTrain, labelsTrain, smilesTest, labelsTest,
                   n_repeats):  # replace MLP classifier with custom classifier. I hate this guy
    print('biodegrade method has been called')
    auc = np.empty(n_repeats)
    for i in range(n_repeats):
        # clf = MLPClassifier(max_iter=1000)
        clf = trfm
        # clf.fit(smilesTrain, labelsTrain)
        # y_score = clf.predict_proba(smilesTest)

        ## write code that will get the y_score from the model
        ## error is TypeError: embedding(): argument 'indices' (position 2) must be Tensor, not numpy.ndarray
        smilesTest = torch.tensor(smilesTest).to(device).long()
        # maybe; b_input_ids = torch.tensor(b_input_ids).to(device).long()

        y_score = clf(smilesTest)

        auc[i] = roc_auc_score(labelsTest, y_score[:, 1])
        print('AUC {:d}: {:.4f}'.format(i, auc[i]))
    ret = {}
    ret['auc mean'] = np.mean(auc)
    ret['auc std'] = np.std(auc)
    return ret


def customBiodegrade(customModel, train_loader, val_loader, optimizer,
                     epoch):  # this is an entire training loop, NOT just one epoch
    global b_input_ids
    loss_fn = torch.nn.BCELoss()
    n_epochs = 40
    # define training loop
    accuracies = []
    for e in range(epoch):
        epoch_loss = 0
        running_loss = 0
        for idx, (inputs, labels) in enumerate(train_loader):
            print(f'Epoch {e + 1}, Batch {idx + 1}')
            print(f'  Inputs: {inputs}, Labels: {labels}')
            # print the length of inputs and labels
            print(f'  Inputs len: {len(inputs)}, Labels len: {len(labels)}')

            customModel.train()  # Ensure the model is in training mode

            optimizer.zero_grad()  # Clear gradients for the next train step

            # inputs = inputs.to(torch.float32)  # Ensure inputs are float32
            # labels = labels.to(torch.float32)  # Ensure labels are float32 for consistency in loss calculation


            print(f'  Batch {idx + 1} Inputs: {inputs.shape}, Labels: {labels.shape}')
            b_input_ids = torch.tensor(inputs).to(device).long()
            outputs = customModel(b_input_ids)
            # outputs = outputs.squeeze()  # Squeeze outputs if necessary to match the dimensions of 'labels'

            loss = loss_fn(outputs, labels)  # Compute the loss between the outputs and the labels
            loss.backward()  # Backward pass: compute gradient of the loss with respect to model parameters
            optimizer.step()  # Perform a single optimization step (parameter update)

            epoch_loss += outputs.shape[0] * loss.item()  # Update total loss for the epoch

            # Print loss every 2 batches
            if idx % 2 == 0:
                print(f'  Batch {idx + 1} Loss: {loss.item()}')

                # Reset running_loss if needed or perform additional operations
                # It seems there was an intent to track 'running_loss' but it's not updated within the loop.
                # Ensure you initialize 'running_loss' before the loop and update it appropriately if you plan to use it.

        # implement method of evaluating the loss and validation every epoch here
        customModel.eval()
        val_loss = 0.0
        # should be the same process as above, just with the validation loader
        total_correct = 0
        total_samples = 0
        print(f"there are {len(val_loader)} in val loader")
        for i, (inputs, labels) in enumerate(val_loader):
            # check the size of val_loader here

            customModel.eval()

            outputs = customModel(inputs)
            outputs = outputs.to(torch.float32)
            labels = labels.to(torch.float32)
            _, predicted = torch.max(outputs, 1)

            loss = loss_fn(outputs.squeeze(-1), labels)
            val_loss += loss.item()
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
        accuracy = 100 * total_correct / total_samples
        print(f'Epoch {e + 1}: Accuracy = {accuracy:.2f}%')
        accuracies.append(accuracy)

    fig, ax = plt.subplots()
    ax.plot(range(epoch), accuracies)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy per Epoch')
    plt.show()
    print("MODEL ACCURACY ON FOLD:", np.mean(accuracies))
    print(" ")
    # implement method of evaluating the loss and validation every train cycle here


# trfm, learning_rate, opt, epochs, batch_size
def _train():
    dataset = pd.read_csv('Data/all_RB.csv')
    smiles_split = [split(sm) for sm in dataset['processed_smiles'].values]
    smilesID, _ = get_array(smiles_split)
    smiles_train = trfm.encode(torch.t(smilesID))
    print(f'{smiles_train.shape}, the datatype is {type(smiles_train)}')
    labels_train = dataset['Class'].values

    # you need to shuffle before you split because otherwise you'll get training data with only biodegradable chemicals and validation data with only non biodegradable data, and stuff like that
    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    overallAverageList = []
    for train_index, test_index in kfold.split(smiles_train, labels_train):
        x_train, x_val, y_train, y_val = smiles_train[train_index], smiles_train[test_index], labels_train[train_index], \
        labels_train[test_index]
        currentAverageDict = testBiodegrade(x_train, y_train, x_val, y_val, 1) ## HACK make this 20
        print(currentAverageDict)
        overallAverageList.append(currentAverageDict['auc mean'])

    overallAverage = np.mean(overallAverageList)
    print(f"Process is done! The overall mean score is {overallAverage}")


def _train2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # need to go back and replace with Data/all_RB.csv
    dataset = pd.read_csv('Data/all_RB.csv')
    smiles_split = [split(sm) for sm in dataset['processed_smiles'].values]
    print(smiles_split)
    smilesID, _ = get_array(smiles_split)
    print(smilesID)
    smiles_train = trfm.encode(torch.t(smilesID))
    ## conert smiles_train to a tensor with dtype float32
    smiles_train = torch.tensor(smiles_train, dtype=torch.float32)
    labels_train = dataset['Class'].values
    epoch = 10

#         sequence = [self.char_to_index[char] for char in self.smiles[idx]]
#         sequence_tensor = torch.tensor(sequence, dtype=torch.long)  # Ensure it's a LongTensor
#         label_tensor = torch.tensor(self.labels[idx], dtype=torch.float)  # Assuming binary classification


    kfold = StratifiedKFold(n_splits=4, shuffle=True)

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
        learning_rate = .01
        # model = classify(1024, 64).to(device)
        # trfm.classifier = classify(1024, 64).to(device)
        # replace 45 with the length of the vocab
        # foo = ExtendedTrfmSeq2seqForBinaryClassification(45, 256, 45, 4).to(device)
        optimizer = optim.Adam(trfm.parameters(), lr=learning_rate)

        # lock down all the existing layers other than our new layers so we only train the new layers

        print('intialized model and optimizer')

        # Train the model on the current fold
        customBiodegrade(trfm, train_loader, test_loader, optimizer, epoch)


_train() #this one is fully operational
# _train2()  # this one isn't (still need to create method of evaluating accuracy)
