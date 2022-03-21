from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import nn


class MLP(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes multi-layer perceptron object.    
        Args:
            n_inputs: number of inputs (i.e., dimension of an input vector).
            n_hidden: list of integers, where each integer is the number of units in each linear layer
            n_classes: number of classes of the classification problem (i.e., output dimension of the network)
        """
        super(MLP, self).__init__()

        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_classes = n_classes

        self.layers = []
        prev_dim = n_inputs
        for num_units in n_hidden:
            self.layers.append(nn.Linear(prev_dim, num_units))
            self.layers.append(nn.ReLU())
            prev_dim = num_units
        self.layers.append(nn.Linear(prev_dim, n_classes))
        self.layers.append(nn.Softmax(dim=1))

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        """
        Predict network output from input by passing it through several layers.
        Args:
            x: input to the network
        Returns:
            out: output of the network
        """
        out = self.layers(x)
        return out
