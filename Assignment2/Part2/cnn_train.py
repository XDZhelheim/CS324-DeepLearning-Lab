from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from cnn_model import CNN
import torch
import torchvision
import matplotlib.pyplot as plt
import datetime

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 64
MAX_EPOCHS_DEFAULT = 200
EVAL_FREQ_DEFAULT = 10
OPTIMIZER_DEFAULT = "ADAM"
DATA_DIR_DEFAULT = "./data"

SEED = 220322


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

    return float(acc)

@torch.no_grad()
def eval_model(model, criterion, x, y):
    out = model.forward(x)
    loss = criterion.forward(out, y)
    acc = accuracy(out, y)

    return float(loss), float(acc)


def train(model,
          trainset,
          evalset,
          criterion,
          batch_size=BATCH_SIZE_DEFAULT,
          max_epochs=MAX_EPOCHS_DEFAULT,
          learning_rate=LEARNING_RATE_DEFAULT,
          verbose=EVAL_FREQ_DEFAULT,
          num_workers=0,
          visual_model=False,
          quiet=False,
          gpu=False):
    """
    Performs training and evaluation of MLP model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    if visual_model:
        train_loss_list = []
        train_acc_list = []
        eval_loss_list = []
        eval_acc_list = []

    if gpu and torch.cuda.is_available():
        print("---Using CUDA---")
        model.cuda()
    else:
        if gpu and not torch.cuda.is_available():
            print("Warning: CUDA not available.")
        print("---Using CPU---")
    # print(model)

    trainset_loader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=num_workers,
                                                  pin_memory=True)
    evalset_loader = torch.utils.data.DataLoader(evalset,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 num_workers=num_workers,
                                                 pin_memory=True)

    print("Train set shape", trainset.data.shape)
    num_batches = len(trainset) // batch_size
    print("Num of batches =", num_batches)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    np.random.seed(SEED)
    for epoch in range(max_epochs):
        batch_loss_list = []
        batch_acc_list = []
        for x_batch, y_batch in trainset_loader:
            if gpu and torch.cuda.is_available():
                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()

            out_batch = model.forward(x_batch)
            batch_acc_list.append(accuracy(out_batch, y_batch))

            loss = criterion.forward(out_batch, y_batch)
            batch_loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % verbose == 0:
            if not quiet or visual_model:
                # train loss
                train_loss = sum(batch_loss_list) / len(batch_loss_list)
                train_acc = sum(batch_acc_list) / len(batch_acc_list)

                # eval loss and acc
                eval_loss, eval_acc = 0, 0
                count = 0
                for eval_x_batch, eval_y_batch in evalset_loader:
                    if gpu and torch.cuda.is_available():
                        eval_x_batch = eval_x_batch.cuda()
                        eval_y_batch = eval_y_batch.cuda()
                    eval_loss_batch, eval_acc_batch = eval_model(
                        model, criterion, eval_x_batch, eval_y_batch)

                    eval_loss += eval_loss_batch
                    eval_acc += eval_acc_batch
                    count += 1
                eval_loss /= count
                eval_acc /= count

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

    if visual_model:
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
    
    CIFAR10
    """

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='../data',
                                            train=True,
                                            download=True,
                                            transform=transform)
    testset = torchvision.datasets.CIFAR10(root='../data',
                                           train=False,
                                           download=True,
                                           transform=transform)

    cnn = CNN(3, 10)
    criterion = torch.nn.CrossEntropyLoss()

    train(cnn,
          trainset,
          testset,
          criterion=criterion,
          batch_size=args.batch_size,
          max_epochs=args.max_epochs,
          learning_rate=args.learning_rate,
          verbose=args.eval_freq,
          num_workers=args.num_workers,
          visual_model=args.visual_model,
          quiet=args.quiet,
          gpu=args.gpu)


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        "-d",
                        type=str,
                        default=DATA_DIR_DEFAULT,
                        help="Directory for storing input data")
    parser.add_argument("--learning_rate",
                        "-l",
                        type=float,
                        default=LEARNING_RATE_DEFAULT,
                        help="Learning rate.")
    parser.add_argument("--batch_size",
                        "-b",
                        type=int,
                        default=BATCH_SIZE_DEFAULT,
                        help="Batch size")
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
    parser.add_argument("--num_workers",
                        "-w",
                        type=int,
                        default=0,
                        help="Param num_workers of pytorch. Set to 0 when using Jupyter!!!")
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
