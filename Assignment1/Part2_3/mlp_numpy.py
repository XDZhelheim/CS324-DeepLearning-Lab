from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import *


class MLP(object):
    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes multi-layer perceptron object.    
        Args:
            n_inputs: number of inputs (i.e., dimension of an input vector).
            n_hidden: list of integers, where each integer is the number of units in each linear layer
            n_classes: number of classes of the classification problem (i.e., output dimension of the network)
        """
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_classes = n_classes

        self.layers = []
        prev_dim = n_inputs
        for num_units in n_hidden:
            self.layers.append(Linear(prev_dim, num_units))
            self.layers.append(ReLU())
            prev_dim = num_units
        self.layers.append(Linear(prev_dim, n_classes))
        self.layers.append(SoftMax())

    def forward(self, x):
        """
        Predict network output from input by passing it through several layers.
        Args:
            x: input to the network
        Returns:
            out: output of the network
        """
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, dout):
        """
        Performs backward propagation pass given the loss gradients. 
        Args:
            dout: gradients of the loss
        """
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

if __name__ == "__main__":
    mlp = MLP(2, [2, 2], 2)
    mlp.forward(np.array([[1, 2], [1, 2], [1, 2]]))
    
    y_pred=np.array([[0.8, 0.2], [0.3, 0.7], [0.3, 0.7]])
    y_true=np.array([[1, 0], [0, 1], [0, 1]])
    
    ce=CrossEntropy()
    loss_grad=ce.backward(y_pred, y_true)
    mlp.backward(loss_grad)
    
    print("---")
    print(mlp.layers[0].grads["weight"])
    print(mlp.layers[2].grads["weight"])
    print(mlp.layers[4].grads["weight"])
    