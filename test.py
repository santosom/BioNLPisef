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

numSamples = len(samples)

splitSamples = [None] * numSamples
for i in range(len(samples)):
    splitSamples[i] = split(samples[i])

VOCAB = WordVocab.load_vocab('data/vocab.pkl')
s2sdataset = Seq2seqDataset(splitSamples, VOCAB)

# loop over the samples and print each vector representation
for i in range(len(s2sdataset)):
    print(f'smile: {samples[i]} split: {splitSamples[i]}\nvector: {s2sdataset.__getitem__(i)}\n\n')