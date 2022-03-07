import numpy as np
import datetime

SEED = 24


class Perceptron():
    def __init__(self, dim, max_epochs=1e2, learning_rate=1e-2):
        """
        Initializes perceptron object.
        Args:
            dim: dimension of x.
            max_epochs: maximum number of training cycles.
            learning_rate: magnitude of weight changes at each training cycle
        """
        self.dim = dim
        self.max_epochs = max_epochs
        self.lr = learning_rate

        self.w = np.zeros(dim)
        self.b = 0

    def forward(self, input):
        """
        Predict label from input 
        Args:
            input: array of dimension equal to n_inputs.
        """
        print("Test set shape", input.shape)

        return np.sign(self.w @ input.T + self.b)

    def train(self, training_inputs, labels, verbose=10):
        """
        Train the perceptron
        Args:
            training_inputs: list of numpy arrays of training points.
            labels: arrays of expected output value for the corresponding point in training_inputs.
        """
        print("Train set shape", training_inputs.shape)
        data = list(zip(training_inputs, labels))
        np.random.seed(SEED)

        epoch = 0
        flag = True
        while flag and epoch < self.max_epochs:
            flag = False
            np.random.shuffle(data)
            for x, y in data:
                if y * (self.w @ x + self.b) <= 0:
                    self.w = self.w + self.lr * x * y
                    self.b = self.b + self.lr * y
                    flag = True

            loss = 0
            for x, y in data:
                if y * (self.w @ x + self.b) <= 0:
                    loss += -y * (self.w @ x + self.b)

            epoch += 1
            if epoch % verbose == 0:
                print(datetime.datetime.now(), "Epoch", epoch,
                      "Train Loss = %.3f" % loss)

    def gen_dataset(self, num, mu1, sigma1, mu2, sigma2):
        np.random.seed(SEED)
        x1 = np.random.normal(loc=mu1, scale=sigma1, size=(num, self.dim))
        x2 = np.random.normal(loc=mu2, scale=sigma2, size=(num, self.dim))
        y1 = np.ones(num)
        y2 = -np.ones(num)

        return np.concatenate((x1, x2), axis=0), np.concatenate((y1, y2),
                                                                axis=0)
