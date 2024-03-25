import numpy as np
import pandas as pd
import torch
#pretty sure that these are just for nice looking graphs. fun though
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from torch import optim
from torch.utils.data import DataLoader
from Scripts.build_vocab import WordVocab
from utils import split
from pretrain_trfm import TrfmSeq2seq
from classifier import classify
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from dataset import biodegradeDataset

pad_index = 0
unk_index = 1
eos_index = 2
sos_index = 3
mask_index = 4
batch_size = 64

vocab = WordVocab.load_vocab('../Data/vocab.pkl')
len_vocab = 45
print("vocab length: ", len(vocab))

print("hello world")
trfm = TrfmSeq2seq(len_vocab, 256, len_vocab , 4)
trfm.load_state_dict(torch.load('../Data/smilesPretrained.pkl', map_location=torch.device('cpu')), strict=False)

#decrease this learning rate if the model performs poorly
df_train = pd.read_csv('../Data/RB_train.csv')
df_val = pd.read_csv('../Data/RB_val.csv')

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

"""def validate(customModel, val_loader, optimizer, loss_fn):
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
    accuracies.append(accuracy)"""


def customBiodegrade(model, train_loader, val_loader, optimizer, epoch): #this is an entire training loop, NOT just one epoch
    customModel = model
    loss_fn = torch.nn.BCELoss()
    n_epochs = 40
    #define training loop
    accuracies = []
    for e in range(epoch):
        epoch_loss = 0
        running_loss = 0
        for idx, (inputs, labels) in enumerate(train_loader):
            print("LABELS- ", labels, " and the type is ", type(labels))
            customModel.train()
            #clear gradient

            outputs = customModel(inputs)
            outputs = outputs.to(torch.float32)
            labels = labels.to(torch.float32)

            print(f'output size is {outputs.size()} while label size is {labels.size()}')

            loss = loss_fn(outputs.squeeze(), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss)
        #implement method of evaluating the loss and validation every epoch here
        customModel.eval()
        val_loss = 0.0
        #should be the same process as above, just with the validation loader
        total_correct = 0
        total_samples = 0
        print(f"there are {len(val_loader)} in val loader")
    fig, ax = plt.subplots()
    ax.plot(range(epoch), accuracies)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy per Epoch')
    plt.show()
    print("MODEL ACCURACY ON FOLD:", np.mean(accuracies))
    print(" ")
    #implement method of evaluating the loss and validation every train cycle here

#trfm, learning_rate, opt, epochs, batch_size
def _train():
    dataset = pd.read_csv('../Data/all_RB.csv')
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
    dataset = pd.read_csv('../Data/all_RB.csv')
    smiles_split = [split(sm) for sm in dataset['processed_smiles'].values]
    smilesID, _ = get_array(smiles_split)
    smiles_train = trfm.encode(torch.t(smilesID))
    labels_train = dataset['Class'].values
    epoch = 10

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
        model = classify(1024, 64).to(device)
        optimizer = optim.Adam(trfm.parameters(), lr=learning_rate)
        print('intialized model and optimizer')

        # Train the model on the current fold
        customBiodegrade(model, train_loader, test_loader, optimizer, epoch)
#_train() #this one is fully operational
_train2() #this one isn't (still need to create method of evaluating accuracy)