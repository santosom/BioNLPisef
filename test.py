import numpy as np
import torch

from Scripts.build_vocab import WordVocab
from dataset import Seq2seqDataset
from utils import split
import array

# make an array of strings to test the split function
samples = []
samples.append('CCCCCCCCCCC')
samples.append('CCC1CO1')
samples.append('CCOCCOCCOCCO')
samples.append('OC=O')
samples.append('CCCCOCCOCCOC(C)=O')
samples.append('CCOCCOCCOC(C)=O')

# # use the following but without spaces
# samples[0] = 'Fc1ccc(cc1)N2C(=O)CC(C2=O)c3noc4cccc34'
# samples[1] = 'Clc1ccc(cc1)N2C(=O)CC(C2=O)c3noc4cccc34'
# samples[2] = 'COC(=O)CSC[C@@H]([C@@H](CC(C)C)C(=O)N[C@@H](Cc1ccccc1)C(=O)N)C(=O)NO'
# samples[3] = 'CN1[C@@H](CF)[C@H]1c2ccnc2'
# samples[4] = '[Br-].CCCCCCCCCCCCCCCCC[N+](C)(C)CCO'
# samples[5] = 'N[C@@H](Cc1ccc(O)cc1)C(=O)N2CCC[C@H]2C(=O)N[C@@H](Cc3ccccc3)C(=O)Nc4ccccc4'

pad_index = 0
unk_index = 1
eos_index = 2
sos_index = 3
mask_index = 4
batch_size = 64

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

numSamples = len(samples)

splitSamples = [None] * numSamples
for i in range(len(samples)):
    splitSamples[i] = split(samples[i])

VOCAB = WordVocab.load_vocab('data/vocab.pkl')

# print the vocabularies
print('vocabularies:')
# print(VOCAB.stoi)
print(VOCAB)

s2sdataset = Seq2seqDataset(samples, VOCAB)

homegrown = [None] * numSamples
for i in range(len(samples)):
    homegrown[i] = torch.from_numpy(np.array(get_inputs(splitSamples[i])))[0]

# loop over the samples and print each vector representation
for i in range(len(s2sdataset)):
    print(f'smile: {samples[i]} split: {splitSamples[i]}\nvector1: {s2sdataset.__getitem__(i)}\nvector2: {homegrown[i]}\n')
