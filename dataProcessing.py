# what it says on the tin

from torch.utils.data import Dataset
import pandas as pd


def prepareNewData():
    biodegradable_smiles = pd.read_excel('Data/bioDataset.xlsx', sheet_name="Train+Test")
    test_bsms = pd.read_excel('Data/bioDataset.xlsx', sheet_name="External validation")

    dataset_len = len(biodegradable_smiles)
    smiles = biodegradable_smiles['Smiles'].values
    to_drop = []
    for i, sm in enumerate(smiles):
        if len(sm) > 100:
            to_drop.append(i)

    dropped_dataset = biodegradable_smiles.drop(to_drop)
    redunantcol = []
    for col in biodegradable_smiles.columns:
        if not ((col == 'Smiles') or (col == 'Class') or (col == 'Status')):
            redunantcol.append(i)
    print('dropping ', len(redunantcol))
    dropped_dataset = dropped_dataset.drop(dropped_dataset.columns[4:45], axis=1)
    dropped_dataset = dropped_dataset.drop('CAS-RN', axis=1)

    # process smiles to get it to be more like the pretrained dataset
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
        # replace RB with 1, NRB with 0. Classes now have numbers
        if (label == 'RB'):
            numberedLabels.append(1)
        else:
            numberedLabels.append(0)
    dropped_dataset = dropped_dataset.drop('Class', axis=1)
    dropped_dataset.insert(1, 'Class', numberedLabels, True)

    # NOTE: might have to return to this and create an empty/random array for chembl_id if training goes poorly the
    # first time, because that's something the original dataset has that we don't
    print('processed smiles len: ', len(pro_sms))
    dropped_dataset.insert(1, 'processed_smiles', pro_sms, True)

    # don't split into training/validation just make a big csv
    dropped_dataset.drop('Status', axis=1)
    dropped_dataset.to_csv('Data/all_RB.csv')
    baby_dataset = dropped_dataset.drop(dropped_dataset.index[3:])
    baby_dataset.to_csv('Data/baby_dataset.csv')


"""    training_index = dropped_dataset.index[dropped_dataset['Status'] == 'Train'].tolist()
    validation_index = dropped_dataset.index[dropped_dataset['Status'] == 'Test'].tolist()

    validation_dataset = dropped_dataset.drop(training_index)
    training_dataset = dropped_dataset.drop(validation_index)

    training_dataset = training_dataset.drop('Status', axis=1)
    validation_dataset = validation_dataset.drop('Status', axis=1)
    validation_dataset.to_csv('Data/RB_train.csv')
    training_dataset.to_csv('Data/RB_val.csv')

    print(training_dataset.head())
    print(training_dataset['Class'].value_counts(normalize=True))
    print(validation_dataset.head())
    print(validation_dataset['Class'].value_counts(normalize=True))"""


class customSmilesDataset(Dataset):
    def __init__(self, rootFile, isTrain):
        print("need to actually... define stuff here. pytorch uses the dataset object to load in material to train")
    # def __len__(self):
    #   return(len(self.))


# please just let me upload to github
# This is only the training+validation dataset


if __name__ == '__main__':
    prepareNewData()

# need to put this in a tensor. before that, need to attach the correct labels for biodegradable and not
# will also need to convert smiles into something useable....
