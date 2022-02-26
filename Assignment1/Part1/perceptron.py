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

    def train(self, training_inputs, labels, verbose=10, save_best=False):
        """
        Train the perceptron
        Args:
            training_inputs: list of numpy arrays of training points.
            labels: arrays of expected output value for the corresponding point in training_inputs.
        """
        print("Train set shape", training_inputs.shape)
        data = list(zip(training_inputs, labels))
        np.random.seed(SEED)

        min_loss, min_loss_epoch, best_arg = np.finfo(np.float32).max, -1, (self.w, self.b)
        epoch = 0
        flag = True
        while flag and epoch < self.max_epochs:
            flag = False
            loss = 0
            np.random.shuffle(data)
            for x, y in data:
                if y * (self.w @ x.T + self.b) <= 0:
                    loss += -y * (self.w @ x.T + self.b)
                    self.w = self.w + self.lr * x * y
                    self.b = self.b + self.lr * y
                    flag = True
                    
            epoch += 1
            if epoch % verbose == 0:
                print(datetime.datetime.now(), "Epoch", epoch,
                      "Train Loss = %.3f" % loss)
                
            if loss < min_loss:
                min_loss=loss
                min_loss_epoch=epoch
                best_arg=(self.w, self.b)
        
        if save_best:
            self.w, self.b = best_arg
            print("Using model at epoch {}. Loss = {:.3f}".format(min_loss_epoch, min_loss))

    def gen_dataset(self, num, mu1, sigma1, mu2, sigma2):
        np.random.seed(SEED)
        x1 = np.random.normal(loc=mu1, scale=sigma1, size=(num, self.dim))
        x2 = np.random.normal(loc=mu2, scale=sigma2, size=(num, self.dim))
        y1 = np.ones(num)
        y2 = -np.ones(num)

        return np.concatenate((x1, x2), axis=0), np.concatenate((y1, y2),
                                                                axis=0)
