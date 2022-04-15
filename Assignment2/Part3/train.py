from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader
import datetime
import matplotlib.pyplot as plt

from dataset import PalindromeDataset
from vanilla_rnn import VanillaRNN

SEED = 220413


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
          optimizer,
          batch_size,
          train_steps,
          eval_steps,
          max_norm,
          verbose,
          num_workers=0,
          visual_model=False,
          quiet=False,
          gpu=False):

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
                                                  num_workers=num_workers)
    evalset_loader = torch.utils.data.DataLoader(evalset,
                                                 batch_size=batch_size,
                                                 num_workers=num_workers)

    # print("Train set shape", trainset.data.shape)
    # num_batches = len(trainset) // batch_size
    # print("Num of batches =", num_batches)

    np.random.seed(SEED)
    loss_sum = 0
    acc_sum = 0
    for step, (x_batch, y_batch) in enumerate(trainset_loader):
        if gpu and torch.cuda.is_available():
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
            
        out_batch = model.forward(x_batch)
        acc_sum += accuracy(out_batch, y_batch)

        loss = criterion.forward(out_batch, y_batch)
        loss_sum += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

        if (step + 1) % verbose == 0:
            if not quiet or visual_model:
                # train loss and acc
                train_loss = loss_sum / verbose
                train_acc = acc_sum / verbose

                # eval loss and acc
                eval_loss, eval_acc = 0, 0
                for eval_step, (eval_x_batch,
                                eval_y_batch) in enumerate(evalset_loader):
                    if gpu and torch.cuda.is_available():
                        eval_x_batch = eval_x_batch.cuda()
                        eval_y_batch = eval_y_batch.cuda()
                    eval_loss_batch, eval_acc_batch = eval_model(
                        model, criterion, eval_x_batch, eval_y_batch)

                    eval_loss += eval_loss_batch
                    eval_acc += eval_acc_batch

                    if eval_step >= eval_steps - 1:
                        break
                eval_loss /= eval_steps
                eval_acc /= eval_steps

            if not quiet:
                print(datetime.datetime.now(), "Step", step + 1,
                      "\tTrain Loss = %.5f" % train_loss,
                      "Train acc = %.3f " % train_acc,
                      "Eval Loss = %.5f" % eval_loss,
                      "Eval acc = %.3f " % eval_acc)

            if visual_model:
                train_loss_list.append(train_loss)
                train_acc_list.append(train_acc)
                eval_loss_list.append(eval_loss)
                eval_acc_list.append(eval_acc)
                
            loss_sum = 0
            acc_sum = 0

        if step >= train_steps - 1:
            break

    if visual_model:
        plt.plot(np.arange(verbose, step + 2, step=verbose),
                 train_loss_list,
                 "-",
                 label="train_loss")
        plt.plot(np.arange(verbose, step + 2, step=verbose),
                 eval_loss_list,
                 "-",
                 label="eval_loss")
        plt.title("Step-Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

        plt.plot(np.arange(verbose, step + 2, step=verbose),
                 train_acc_list,
                 "-",
                 label="train_acc")
        plt.plot(np.arange(verbose, step + 2, step=verbose),
                 eval_acc_list,
                 "-",
                 label="eval_acc")
        plt.title("Step-Accuracy")
        plt.xlabel("Step")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()


def main(args):
    model = VanillaRNN(args.input_dim, args.num_hidden, args.num_classes)

    trainset = PalindromeDataset(args.input_length + 1)
    testset = PalindromeDataset(args.input_length + 1)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate)

    train(model,
          trainset,
          testset,
          criterion=criterion,
          optimizer=optimizer,
          batch_size=args.batch_size,
          train_steps=args.train_steps,
          eval_steps=5,
          max_norm=args.max_norm,
          verbose=args.eval_freq,
          num_workers=args.num_workers,
          visual_model=args.visual_model,
          quiet=args.quiet,
          gpu=args.gpu)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_length",
                        type=int,
                        default=10,
                        help="Length of an input sequence")
    parser.add_argument("--input_dim",
                        type=int,
                        default=1,
                        help="Dimensionality of input sequence")
    parser.add_argument("--num_classes",
                        type=int,
                        default=10,
                        help="Dimensionality of output sequence")
    parser.add_argument("--num_hidden",
                        type=int,
                        default=128,
                        help="Number of hidden units in the model")
    parser.add_argument("--learning_rate",
                        "-l",
                        type=float,
                        default=1e-3,
                        help="Learning rate.")
    parser.add_argument("--batch_size",
                        "-b",
                        type=int,
                        default=128,
                        help="Batch size")
    parser.add_argument("--train_steps",
                        "-s",
                        type=int,
                        default=1000,
                        help="Number of training steps(not epoch).")
    parser.add_argument("--max_norm", type=float, default=10.0)
    parser.add_argument("--eval_freq",
                        "-f",
                        type=int,
                        default=10,
                        help="Frequency of evaluation on the test set.")
    parser.add_argument(
        "--num_workers",
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