import argparse
import pickle
from collections import Counter
#NOTE: I don't think this is actually a Vocab object, like what's commonly used in Pytorch NLP problems. This is a custom object made seperately. We make vocab.pkl with this
#Look at the lack of any import torch here
# Why is this the only file in the entire project with comments.
#The comments were originally in Japanese, just run through Google Translate. I used the transformer to figure out the transformer!

class TorchVocab(object):
    """
    :property freqs: collections.Counter, An object that holds the frequency of occurrence of words in the corpus
    :property stoi: collections.defaultdict, string → Dictionary showing ID correspondence
    :property itos: collections.defaultdict, id → Dictionary of string mappings
    """
    def __init__(self, counter, max_size=None, min_freq=1, specials=['<pad>', '<oov>'],
                 vectors=None, unk_init=None, vectors_cache=None):
        """
        :param counter: collections.Counter, counter to measure the frequency of words included in the data
        :param max_size: int, Maximum size of vocabulary. If None, there is no maximum value. Default is None
        :param min_freq: int, Minimum frequency of occurrence of a word in the vocabulary. Words with occurrences less than this number will not be added to the vocabulary.
        :param specials: list of str, token to be registered in the vocabulary in advance
        :param vectors: list of vectors, Pretrained vectors. ex.) Vocab.load_vectors
        """
        self.freqs = counter
        counter = counter.copy()
        min_freq = max(min_freq, 1)

        self.itos = list(specials)
        # special tokens, Frequency of occurrence is not counted when creating vocabulary
        for token in specials:
            del counter[token]

        max_size = None if max_size is None else max_size + len(self.itos)

        # Sort first by frequency, then by letter order
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)
        
        # Items whose appearance frequency is less than min_freq are not added to vocab.
        # there's a frequency threshold as defined by min_freq
        for word, freq in words_and_frequencies:
            if freq < min_freq or len(self.itos) == max_size:
                break
            self.itos.append(word)

        # Create stoi by replacing k and v in dict
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

        self.vectors = None
        if vectors is not None:
            self.load_vectors(vectors, unk_init=unk_init, cache=vectors_cache)
        else:
            assert unk_init is None and vectors_cache is None

    def __eq__(self, other):
        if self.freqs != other.freqs:
            return False
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        if self.vectors != other.vectors:
            return False
        return True

    def __len__(self):
        return len(self.itos)

    def vocab_rerank(self):
        self.stoi = {word: i for i, word in enumerate(self.itos)}

    def extend(self, v, sort=False):
        words = sorted(v.itos) if sort else v.itos
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1


class Vocab(TorchVocab): #the WordVocab class below should inherit all of these functions? and this class should inherit functions from the above class
    def __init__(self, counter, max_size=None, min_freq=1):
        self.pad_index = 0
        self.unk_index = 1
        self.eos_index = 2
        self.sos_index = 3
        self.mask_index = 4
        super().__init__(counter, specials=["<pad>", "<unk>", "<eos>", "<sos>", "<mask>"], max_size=max_size, min_freq=min_freq)

    # override用
    #i believe this creates a list of the passed values?
    # or this doesn't return anything. the comment above indicates it's a required def with no real purpose.
    def to_seq(self, sentece, seq_len, with_eos=False, with_sos=False) -> list:
        pass

    # override用
    #this shouldn't return anything. the comment above indicates it's a required def with no real purpose.
    def from_seq(self, seq, join=False, with_pad=False):
        pass

    @staticmethod
    def load_vocab(vocab_path: str) -> 'Vocab':
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    def save_vocab(self, vocab_path):
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)


# create vocab from text file
class WordVocab(Vocab):
    def __init__(self, texts, max_size=None, min_freq=1):
        print("Building Vocab")
        counter = Counter()
        for line in texts:
            if isinstance(line, list):
                words = line
            else:
                words = line.replace("\n", "").replace("\t", "").split()

            for word in words:
                counter[word] += 1
        super().__init__(counter, max_size=max_size, min_freq=min_freq)

    def to_seq(self, sentence, seq_len=None, with_eos=False, with_sos=False, with_len=False):
        if isinstance(sentence, str):
            sentence = sentence.split()

        seq = [self.stoi.get(word, self.unk_index) for word in sentence]

        if with_eos:
            seq += [self.eos_index]  # this would be index 1
        if with_sos:
            seq = [self.sos_index] + seq

        origin_seq_len = len(seq)

        if seq_len is None:
            pass
        elif len(seq) <= seq_len:
            seq += [self.pad_index for _ in range(seq_len - len(seq))]
        else:
            seq = seq[:seq_len]

        return (seq, origin_seq_len) if with_len else seq

    def from_seq(self, seq, join=False, with_pad=False):
        words = [self.itos[idx]
                 if idx < len(self.itos)
                 else "<%d>" % idx
                 for idx in seq
                 if not with_pad or idx != self.pad_index]

        return " ".join(words) if join else words

    @staticmethod
    def load_vocab(vocab_path: str) -> 'WordVocab':
        with open(vocab_path, "rb") as f:
            #i believe this returns a Vocab object
            return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(description='Build a vocabulary pickle')
    parser.add_argument('--corpus_path', '-c', type=str, default='data/chembl24_corpus.txt', help='path to th ecorpus')
    parser.add_argument('--out_path', '-o', type=str, default='data/vocab.pkl', help='output file')
    parser.add_argument('--min_freq', '-m', type=int, default=500, help='minimum frequency for vocabulary')
    parser.add_argument('--vocab_size', '-v', type=int, default=None, help='max vocabulary size')
    parser.add_argument('--encoding', '-e', type=str, default='utf-8', help='encoding of corpus')
    args = parser.parse_args()

    with open(args.corpus_path, "r", encoding=args.encoding) as f:
        vocab = WordVocab(f, max_size=args.vocab_size, min_freq=args.min_freq)

    print("VOCAB SIZE:", len(vocab))
    vocab.save_vocab(args.out_path)

if __name__=='__main__':
    main()