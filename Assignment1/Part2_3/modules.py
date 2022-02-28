import numpy as np


class Linear(object):
    def __init__(self, in_features, out_features):
        """
        Module initialisation.
        Args:
            in_features: input dimension
            out_features: output dimension
            
        - vectorization (batch)
        - multiple hidden units
        
        These will take 2 more dims into consideration.
            
        1) Initialize weights self.params['weight'] using normal distribution with mean = 0 and 
        std = 0.0001.
        2) Initialize biases self.params['bias'] with 0. 
        3) Initialize gradients with zeros.
        """
        self.params = {}
        self.params["weight"] = np.random.normal(loc=1, # !!!! dont set loc=0 层数多的时候 w太接近0导致严重的梯度消失问题 !!!!
                                                 scale=0.0001,
                                                 size=(in_features,
                                                       out_features))
        self.params["bias"] = np.zeros(out_features)

        self.grads = {}

    def forward(self, x):
        """
        Forward pass (i.e., compute output from input).
        Args:
            x: input to the module (m, n) m=batch_size, n=in_features=dim_of_x[i]
        Returns:
            out: output of the module (m, out_features) out_features=num_of_hidden_units
            
        out = x @ w + b: (m * n) @ (n * units) + (units * 1) = (m * units)
        
        eg. m=batch_size=2, units=out_features=3
        - a=[[1, 2, 3], [4, 5, 6]]
        - b=[0.1, 0.2, 0.3]
        - then a+b=[[1.1, 2.2, 3.3], [4.1, 4.2, 4.3]]
        - every unit have a unique b[i]
        - when units=out_features=1, it becomes the simple case: w is a vector, b is a number
        - when m=1 and units=1, it is part1, no vectorization, single hidden unit
            
        Hint: Similarly to pytorch, you can store the computed values inside the object
        and use them in the backward pass computation. This is true for *all* forward methods of *all* modules in this class
        """
        self.x = x
        self.out = x @ self.params["weight"] + self.params["bias"]  # (m * n) @ (n * units) + (units * 1) = (m * units)

        # print("linear out shape", self.out.shape, sep="\t")

        return self.out

    def backward(self, dout):
        """
        Backward pass (i.e., compute gradient).
        Args:
            dout: gradients of the previous module
        Returns:
            dx: gradients with respect to the input of the module
            
        Implement backward pass of the module. Store gradient of the loss with respect to 
        layer parameters in self.grads['weight'] and self.grads['bias']. 
        """
        self.grads["weight"] = (self.x.T @ dout) / self.x.shape[0]
        self.grads["bias"] = np.mean(dout, axis=0)
        dx = dout @ self.params["weight"].T

        # print("linear dx shape  ", dx.shape, sep="\t")

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
        self.x=x

        # print("relu out shape  ", self.out.shape, sep="\t")

        return self.out

    def backward(self, dout):
        """
        Backward pass.
        Args:
            dout: gradients of the previous module
        Returns:
            dx: gradients with respect to the input of the module
        """
        dx = np.where(self.x > 0, dout, 0)

        # print("relu dx shape    ", dx.shape, sep="\t")
        
        return dx


class SoftMax(object):
    def forward(self, x):
        """
        Forward pass.
        Args:
            x: input to the module (m, n_classes)
        Returns:
            out: output of the module (m, n_classes)
            
        Calculate probabilities for each x[i]. Shape of x[i] is (1, n_classes).
        
        - out[i] = exp(x[i]) / sum(exp(x[i]))
        - out != exp(x) / sum(exp(x)) because sum will flat the array
        - out != exp(x) / sum(exp(x), axis=1) because it is c[i]=a[i]/b, and we need c[i]=a[i]/b[i]
        - out = exp(x) / sum(exp(x), axis=1).reshape(-1, 1)
        
        ---
        
        eg. two inputs a=[[1, 3], [3, 3]], then b=sum(a, axis=1)=[4, 6]
        
        a/b=[[0.25, 0.5], [0.75, 0.5]] which is [[1/4, 3/6], [3/4, 3/6]]
        
        So add a second dim to b: b.reshape(-1, 1)=[[4], [6]]
        
        Then a/b=[[1/4, 3/4], [3/6, 3/6]]=[[0.25, 0.75], [0.5, 0.5]]
        
        ---
        
        To stabilize computation you should use the so-called Max Trick
        https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
        
        """
        y = np.exp(x - np.max(x))
        self.out = y / np.sum(y, axis=1).reshape(-1, 1)  # (m * n_classes)

        # print("softmax out shape", self.out.shape, sep="\t")

        return self.out

    def backward(self, dout):
        """
        Backward pass. 
        Args:
            dout: gradients of the previous module
        Returns:
            dx: gradients with respect to the input of the module
        """
        dx = np.array([
            np.diagflat(out_i) - np.outer(out_i, out_i) for out_i in self.out
        ])
        dx = np.array([dout[i] @ dx[i] for i in range(len(dout))])

        # dx = (dout - np.reshape(np.sum(dout * self.out, 1), [-1, 1])) * self.out

        # print("softmax dx shape", dx.shape, sep="\t")
        
        return dx


class CrossEntropy(object):
    def forward(self, x, y):
        """
        Forward pass.
        
        J = 1/m * Σ(i=1 to m) CELoss(y_pred[i], y_true[i])
        CELoss[i] = -Σ(j=1 to n_classes) y_pred[i][j]*log(y_true[i][j])
        
        Args:
            x: input to the module: y_pred (m * n_classes)
            y: labels of the input: y_true (m * n_classes)
        Returns:
            out: cross entropy loss
        
        ---
        
        eg.
        
        x=y_pred=[[0.8, 0.2], [0.5, 0.5], [0.3, 0.7]]
        
        y=y_true=[[1, 0], [0, 1], [0, 1]]
        
        1. y * log(x) = [[0.22, 0], [0, 0.69], [0, 0.35]] (m * n_classes) 对应元素乘
        2. CELoss = -sum(y*np.log(x), axis=1)=[0.22, 0.69, 0.35] (1 * m)
        3. mean(CELoss) = 0.42
        
        ---
        """
        loss_list = -np.sum(y * np.log(x + 1e-15), axis=1)  # (1 * m) vector
        out = np.mean(loss_list)

        return out

    def backward(self, x, y):
        """
        Backward pass.
        Args:
            x: input to the module (y_pred)
            y: labels of the input (y_true)
        Returns:
            dx: gradient of the loss with respect to the input x.
            
        ∂L/∂y_pred[i] = -y_true[i]/y_pred[i]
            
        Vectorization:
        suppose y_pred = [[0.8, 0.2], [0.5, 0.5], [0.3, 0.7]], y_true = [[1, 0], [0, 1], [0, 1]]
        
        Then ∂L/∂y_pred = -y_true/y_pred = -[[1/0.8, 0/0.2], [0/0.5, 1/0.5], [0/0.3, 1/0.7]] 对应元素除
        """
        dx = -y / (x + 1e-15)  # (m * n_classes)

        # print("CE dx shape     ", dx.shape, sep="\t")

        return dx
    