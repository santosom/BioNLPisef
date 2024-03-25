# what it says on the tin

from torch.utils.data import Dataset
import pandas as pd


# Prepare Training + Validation dataset
def prepareNewData():
    # Load in dataset excel files
    biodegradable_smiles = pd.read_excel('Data/bioDataset2.xlsx', sheet_name="Train+Test")
    test_bsms = pd.read_excel('Data/bioDataset2.xlsx', sheet_name="External validation")

    # Iterate through SMILES in dataset, drop any SMILES that are too big
    smiles = biodegradable_smiles['Smiles'].values
    to_drop = []
    for i, sm in enumerate(smiles):
        if len(sm) > 100:
            to_drop.append(i)

    # Make a new dataset comprised of remaining datapoints
    dropped_dataset = biodegradable_smiles.drop(to_drop)
    # Get rid of all data columns beyond SMILES, class (biodegradable classification), and status (validation or
    # training)
    dropped_dataset = dropped_dataset.drop(dropped_dataset.columns[4:45], axis=1)
    dropped_dataset = dropped_dataset.drop('CAS-RN', axis=1)

    # Process smiles to get it to be more like the pretrained dataset
    smiles = dropped_dataset['Smiles'].values
    pro_sms = []
    for sm in smiles:
        sm = ' '.join(list(sm))
        before = ['C l -', 'C l', 'O -', 'N +', 'n +', 'B r -', 'B r', 'N a +', 'N a', 'I -', 'S i']
        after = ['Cl-', 'Cl', 'O-', 'N+', 'n+', 'Br-', 'Br', 'Na+', 'Na', 'I-', 'Si']
        for b, a in zip(before, after):
            sm = sm.replace(b, a)
        pro_sms.append(sm)

    _class = dropped_dataset['Class']
    numberedLabels = []
    for label in _class:
        # Replace RB with 1, NRB with 0. Classes now have numbers instead of strings
        if (label == 'RB'):
            numberedLabels.append(1)
        else:
            numberedLabels.append(0)
    dropped_dataset = dropped_dataset.drop('Class', axis=1)
    dropped_dataset.insert(1, 'Class', numberedLabels, True)
    dropped_dataset.insert(1, 'processed_smiles', pro_sms, True)

    # Don't split into training/validation just make a big csv
    dropped_dataset.drop('Status', axis=1)
    dropped_dataset.to_csv('Data/all_RB.csv')
    baby_dataset = dropped_dataset.drop(dropped_dataset.index[3:])
    baby_dataset.to_csv('Data/baby_dataset.csv')

    # Alternatively, split dataset into training/validation csv's based on individual datapoints' "Status" labels in
    # the dataset
    training_index = dropped_dataset.index[dropped_dataset['Status'] == 'Train'].tolist()
    validation_index = dropped_dataset.index[dropped_dataset['Status'] == 'Test'].tolist()

    validation_dataset = dropped_dataset.drop(training_index)
    training_dataset = dropped_dataset.drop(validation_index)

    # Prune the "Status" labels from dataset
    training_dataset = training_dataset.drop('Status', axis=1)
    validation_dataset = validation_dataset.drop('Status', axis=1)
    validation_dataset.to_csv('Data/RB_val.csv')
    training_dataset.to_csv('Data/RB_train.csv')

    # Double check datasets by printing out first few values
    print('TRAINING')
    print(training_dataset.head())
    print(training_dataset['Class'].value_counts(normalize=True))
    print('VALIDATION')
    print(validation_dataset.head())
    print(validation_dataset['Class'].value_counts(normalize=True))
    print('training size is ', len(training_dataset), ' and val size is ', len(validation_dataset))

def prepareFinalData():
    print("preparing final data")
    test_bsms = pd.read_excel('Data/bioDataset2.xlsx', sheet_name="External validation")

    smiles = test_bsms['Smiles'].values
    to_drop = []
    for i, sm in enumerate(smiles):
        if len(sm) > 100:
            to_drop.append(i)

    dropped_dataset = test_bsms.drop(to_drop)
    dropped_dataset = dropped_dataset.drop(dropped_dataset.columns[4:45], axis=1)
    dropped_dataset = dropped_dataset.drop('CAS-RN', axis=1)

    # Process smiles to get it to be more like the pretrained dataset
    smiles = dropped_dataset['Smiles'].values
    pro_sms = []
    for sm in smiles:
        sm = ' '.join(list(sm))
        before = ['C l -', 'C l', 'O -', 'N +', 'n +', 'B r -', 'B r', 'N a +', 'N a', 'I -', 'S i']
        after = ['Cl-', 'Cl', 'O-', 'N+', 'n+', 'Br-', 'Br', 'Na+', 'Na', 'I-', 'Si']
        for b, a in zip(before, after):
            sm = sm.replace(b, a)
        pro_sms.append(sm)

    # Replace RB with 1, NRB with 0. Classes now have numbers instead of strings
    _class = dropped_dataset['class']
    numberedLabels = []
    for label in _class:
        if (label == 'RB'):
            numberedLabels.append(1)
        else:
            numberedLabels.append(0)
    dropped_dataset = dropped_dataset.drop('class', axis=1)
    dropped_dataset.insert(1, 'Class', numberedLabels, True)

    dropped_dataset.insert(1, 'processed_smiles', pro_sms, True)
    dropped_dataset.to_csv('Data/RB_Final.csv')
    print(dropped_dataset.head())
    print(dropped_dataset['Class'].value_counts(normalize=True))

# please just let me upload to github
# This is only the training+validation dataset
#prepareFinalData()
val = pd.read_csv('Data/RB_val.csv')
print('Class distribution for val:')
print(val['Class'].value_counts(normalize=True))
"""if __name__ == '__main__':
    prepareNewData()"""

