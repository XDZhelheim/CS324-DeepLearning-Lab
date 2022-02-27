import numpy as np


class Linear(object):
    def __init__(self, in_features, out_features):
        """
        Module initialisation.
        Args:
            in_features: input dimension
            out_features: output dimension
        TODO:
        1) Initialize weights self.params['weight'] using normal distribution with mean = 0 and 
        std = 0.0001.
        2) Initialize biases self.params['bias'] with 0. 
        3) Initialize gradients with zeros.
        """
        self.params = {}
        self.params["weight"] = np.random.normal(loc=0,
                                                 scale=0.0001,
                                                 size=(in_features,
                                                       out_features))
        self.params["bias"] = np.zeros(out_features)

        self.grads = {}

    def forward(self, x):
        """
        Forward pass (i.e., compute output from input).
        Args:
            x: input to the module
        Returns:
            out: output of the module
        Hint: Similarly to pytorch, you can store the computed values inside the object
        and use them in the backward pass computation. This is true for *all* forward methods of *all* modules in this class
        """
        self.x = x
        self.out = x @ self.params["weight"] + self.params["bias"]
        
        print("out shape", self.out.shape)
        return self.out

    def backward(self, dout):
        """
        Backward pass (i.e., compute gradient).
        Args:
            dout: gradients of the previous module
        Returns:
            dx: gradients with respect to the input of the module
        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to 
        layer parameters in self.grads['weight'] and self.grads['bias']. 
        """
        self.grads["weight"] = self.x.T @ dout
        self.grads["bias"] = dout
        dx = dout @ self.params["weight"].T
        
        print("dx shape", dx.shape)
        return dx


class ReLU(object):
    def forward(self, x):
        """
        Forward pass.
        Args:
            x: input to the module
        Returns:
            out: output of the module
        """
        self.out = np.maximum(0, x)

        print("out shape", self.out.shape)
        return self.out

    def backward(self, dout):
        """
        Backward pass.
        Args:
            dout: gradients of the previous module
        Returns:
            dx: gradients with respect to the input of the module
        """
        dx = np.where(self.out > 0, dout, 0)
        
        print("dx shape", dx.shape)
        return dx


class SoftMax(object):
    def forward(self, x):
        """
        Forward pass.
        Args:
            x: input to the module
        Returns:
            out: output of the module
    
        TODO:
        Implement forward pass of the module. 
        To stabilize computation you should use the so-called Max Trick
        https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
        
        """
        y = np.exp(x - np.max(x))
        self.out = y / np.sum(y, axis=1).reshape(-1, 1)
        
        print("out shape", self.out.shape)
        return self.out

    def backward(self, dout):
        """
        Backward pass. 
        Args:
            dout: gradients of the previous module
        Returns:
            dx: gradients with respect to the input of the module
        """
        # dx = np.diagflat(self.out) - np.outer(self.out, self.out)
        # dx = dout @ dx  #!
        dx = (dout - np.reshape(np.sum(dout * self.out, 1), [-1, 1])) * self.out
        
        print("dx shape", dx.shape)
        return dx


class CrossEntropy(object):
    def forward(self, x, y):
        """
        Forward pass.
        Args:
            x: input to the module
            y: labels of the input
        Returns:
            out: cross entropy loss
        """
        out = -np.sum(y * np.log(x + 1e-15))  # avoid zero
        
        print("out shape", out.shape)
        return out

    def backward(self, x, y):
        """
        Backward pass.
        Args:
            x: input to the module
            y: labels of the input
        Returns:
            dx: gradient of the loss with respect to the input x.
        """
        dx = -y / (x + 1e-15)  # avoid zero
        
        print("dx shape", dx.shape)
        return dx
