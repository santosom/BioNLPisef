import torch.nn as nn
import torch.nn.functional as F
from sklearn.neural_network import MLPClassifier

from pretrain_trfm import TrfmSeq2seq


class ExtendedTrfmSeq2seqForBinaryClassification(TrfmSeq2seq):
    def __init__(self, in_size, hidden_size, out_size, n_layers, dropout=0.1):
        # def __init__(self, in_size, hidden_size, out_size, n_layers, dropout=0.1):
        super(ExtendedTrfmSeq2seqForBinaryClassification, self).__init__(in_size, hidden_size, out_size, n_layers, dropout)
        # Define the classifier for binary classification
        # Adjust the size of the input features to match the output of your Transformer model
        # self.classifier = MLPClassifier(max_iter=1000)
        # self.classifier = nn.Sequential(
        #     nn.Linear(hidden_size, 128),  # Example intermediate layer
        #     nn.ReLU(),
        #     nn.Linear(128, 1),  # Output layer for binary classification
        #     nn.Sigmoid()  # Sigmoid activation to output probabilities
        # )

    def forward(self, src):
        # Use the original Transformer model to get the encoding
        print("forward method has been called, src: ", src)
        trfm_output = super(ExtendedTrfmSeq2seqForBinaryClassification, self).forward(src)
        print("trfm_output: ", trfm_output)
        # You might need to adapt this part based on the actual output of your Transformer
        # If trfm_output is a sequence, decide how to aggregate it (e.g., taking the last vector, mean, etc.)
        # Example: output = self.classifier(trfm_output[:, -1, :]) if you want to use the last vector for classification
        output = self.classifier(trfm_output)
        print("output: ", output)
        return output
