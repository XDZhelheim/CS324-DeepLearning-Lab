from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from pytorch_mlp import MLP
import torch
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import datetime
import matplotlib.pyplot as plt

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 1500
EVAL_FREQ_DEFAULT = 10
MAX_EPOCHS_DEFAULT = 200
EVAL_FREQ_DEFAULT = 10
BATCH_SIZE_DEFAULT = 4

SEED = 220321

# def onehot_encode(label_vec, n_classes):
#     return np.eye(n_classes)[label_vec]


def onehot_decode(label):
    return torch.argmax(label, dim=1)


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e., the average of correct predictions
    of the network.
    Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        labels: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding of ground-truth labels
    Returns:
        accuracy: scalar float, the accuracy of predictions.
    """

    pred_decode = onehot_decode(predictions)
    true_decode = targets

    assert (len(pred_decode) == len(true_decode))

    acc = torch.mean((pred_decode == true_decode).float())

    return acc


def eval_model(model, criterion, x, y):
    out = model.forward(x)
    loss = criterion.forward(out, y)
    acc = accuracy(out, y)

    return float(loss), float(acc)


def train(model,
          x_train,
          y_train,
          x_eval,
          y_eval,
          criterion,
          batch_size=8,
          max_epochs=200,
          learning_rate=0.01,
          verbose=10,
          save_fig=False,
          visual_model=False,
          quiet=False,
          gpu=False):
    """
    Performs training and evaluation of MLP model.
    NOTE: You should evaluate the model on the whole test set each eval_freq iterations.
    """
    if save_fig or visual_model:
        if save_fig:
            if not os.path.exists("./images"):
                os.mkdir("./images")
        xline = np.linspace(-1.5, 2.5, 300)
        yline = np.linspace(-0.75, 1.25, 300)
        xx, yy = np.meshgrid(xline, yline)
        xy = np.c_[xx.ravel(), yy.ravel()]
        xy = torch.FloatTensor(xy)
        if gpu and torch.cuda.is_available():
            xy = xy.cuda()

        def draw():
            plt.figure(figsize=(5, 5))
            plt.title(f"Classification - epoch: {str(epoch+1).zfill(3)}",
                      fontsize=15)
            plt.xlabel("X", fontsize=15)
            plt.ylabel("Y", fontsize=15)

            pred = onehot_decode(model.forward(xy).cpu())
            zz = pred.reshape(xx.shape)
            plt.contourf(xx, yy, zz, alpha=0.7, cmap=plt.cm.Spectral)
            plt.scatter(x_train[:, 0].cpu(),
                        x_train[:, 1].cpu(),
                        c=y_train.cpu(),
                        s=50,
                        cmap=plt.cm.Spectral)

        if visual_model:
            train_loss_list = []
            train_acc_list = []
            eval_loss_list = []
            eval_acc_list = []

    print("Train set shape", x_train.shape)
    num_batches = len(x_train) // batch_size
    print("Num of batches =", num_batches)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    np.random.seed(SEED)
    for epoch in range(max_epochs):
        random_index = np.random.choice(len(x_train),
                                        size=len(x_train),
                                        replace=False)
        for i in range(num_batches):
            batch_index = random_index[i * batch_size:(i + 1) * batch_size]
            x_batch = x_train[batch_index]
            y_batch = y_train[batch_index]

            out_batch = model.forward(x_batch)
            loss = criterion.forward(out_batch, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % verbose == 0:
            if not quiet or visual_model:
                train_loss, train_acc = eval_model(model, criterion, x_train,
                                                   y_train)
                eval_loss, eval_acc = eval_model(model, criterion, x_eval,
                                                 y_eval)

            if not quiet:
                print(datetime.datetime.now(), "Epoch", epoch + 1,
                      "\tTrain Loss = %.5f" % train_loss,
                      "Train acc = %.3f " % train_acc,
                      "Eval Loss = %.5f" % eval_loss,
                      "Eval acc = %.3f " % eval_acc)

            if visual_model:
                train_loss_list.append(train_loss)
                train_acc_list.append(train_acc)
                eval_loss_list.append(eval_loss)
                eval_acc_list.append(eval_acc)

        if save_fig:
            draw()
            plt.savefig(f"./images/{epoch+1}.png")
            plt.close()

    if visual_model:
        draw()
        plt.show()

        plt.plot(np.arange(verbose, epoch + 2, step=verbose),
                 train_loss_list,
                 "-",
                 label="train_loss")
        plt.plot(np.arange(verbose, epoch + 2, step=verbose),
                 eval_loss_list,
                 "-",
                 label="eval_loss")
        plt.title("Epoch-Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

        plt.plot(np.arange(verbose, epoch + 2, step=verbose),
                 train_acc_list,
                 "-",
                 label="train_acc")
        plt.plot(np.arange(verbose, epoch + 2, step=verbose),
                 eval_acc_list,
                 "-",
                 label="eval_acc")
        plt.title("Epoch-Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()


def main(args):
    """
    Main function
    """
    units = np.array(args.dnn_hidden_units.split(","), dtype=np.int8)
    points = 2000

    x, y = make_moons(points, random_state=SEED)
    x = torch.FloatTensor(x)
    y = torch.LongTensor(y)
    if args.gpu and torch.cuda.is_available():
        print("---Using CUDA---")
        x = x.cuda()
        y = y.cuda()
    else:
        if args.gpu and not torch.cuda.is_available():
            print("Warning: CUDA not available.")
        print("---Using CPU---")
    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        test_size=0.3,
                                                        random_state=SEED)

    mlp = MLP(x.shape[1], units, 2)
    if args.gpu and torch.cuda.is_available():
        mlp.cuda()
    print(mlp)

    criterion = torch.nn.CrossEntropyLoss() # !!! torch does not accept one-hot labels !!!

    if args.batch_size == -1:
        batch_size = len(x_train)  # BGD
    else:
        batch_size = args.batch_size

    print("Using test set as evaluation.")

    train(mlp,
          x_train,
          y_train,
          x_eval=x_test,
          y_eval=y_test,
          criterion=criterion,
          batch_size=batch_size,
          max_epochs=args.max_epochs,
          learning_rate=args.learning_rate,
          verbose=args.eval_freq,
          save_fig=args.save_fig,
          visual_model=args.visual_model,
          quiet=args.quiet,
          gpu=args.gpu)


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dnn_hidden_units",
        "-u",
        type=str,
        default=DNN_HIDDEN_UNITS_DEFAULT,
        help="Comma separated list of number of units in each hidden layer.")
    parser.add_argument("--learning_rate",
                        "-l",
                        type=float,
                        default=LEARNING_RATE_DEFAULT,
                        help="Learning rate.")
    parser.add_argument("--max_epochs",
                        "-e",
                        type=int,
                        default=MAX_EPOCHS_DEFAULT,
                        help="Number of epochs to run trainer.")
    parser.add_argument("--eval_freq",
                        "-f",
                        type=int,
                        default=EVAL_FREQ_DEFAULT,
                        help="Frequency of evaluation on the test set.")
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=BATCH_SIZE_DEFAULT,
        help="Batch size: 1 -- SGD, -1 -- BGD, others -- Mini-batch GD")
    parser.add_argument("--save_fig",
                        "-s",
                        action="store_true",
                        help="Save model visualization figure on each epoch.")
    parser.add_argument("--visual_model",
                        "-v",
                        action="store_true",
                        help="Visualize model after training.")
    parser.add_argument("--quiet",
                        "-q",
                        action="store_true",
                        help="No stdout when training.")
    parser.add_argument("--gpu", "-g", action="store_true", help="Use GPU.")

    args = parser.parse_args()

    main(args)