from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import math

################################################################################


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.Wgx = Parameter(torch.empty(input_dim, hidden_dim))
        self.Wgh = Parameter(torch.empty(hidden_dim, hidden_dim))
        self.bg = Parameter(torch.empty(hidden_dim))

        self.Wix = Parameter(torch.empty(input_dim, hidden_dim))
        self.Wih = Parameter(torch.empty(hidden_dim, hidden_dim))
        self.bi = Parameter(torch.empty(hidden_dim))

        self.Wfx = Parameter(torch.empty(input_dim, hidden_dim))
        self.Wfh = Parameter(torch.empty(hidden_dim, hidden_dim))
        self.bf = Parameter(torch.empty(hidden_dim))

        self.Wox = Parameter(torch.empty(input_dim, hidden_dim))
        self.Woh = Parameter(torch.empty(hidden_dim, hidden_dim))
        self.bo = Parameter(torch.empty(hidden_dim))

        self.Wph = Parameter(torch.empty(hidden_dim, output_dim))
        self.bp = Parameter(torch.empty(output_dim))

        self.reset_parameters()

    def reset_parameters(self):
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/rnn.py
        stdv = 1.0 / math.sqrt(self.hidden_dim) if self.hidden_dim > 0 else 0
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        """
        x: (batch_size, seq_len, input_dim)
        input_dim = 1, no word embedding
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        x = x.view(batch_size, seq_len, self.input_dim)

        for timestep in range(seq_len):
            xt = x[:, timestep, :]
            if timestep == 0:
                g = torch.tanh(xt @ self.Wgx + self.bg)

                i = torch.sigmoid(xt @ self.Wix + self.bi)
                f = torch.sigmoid(xt @ self.Wfx + self.bf)
                o = torch.sigmoid(xt @ self.Wox + self.bo)

                c = g * i
                h = torch.tanh(c) * o
            else:
                g = torch.tanh(xt @ self.Wgx + h @ self.Wgh + self.bg)

                i = torch.sigmoid(xt @ self.Wix + h @ self.Wih + self.bi)
                f = torch.sigmoid(xt @ self.Wfx + h @ self.Wfh + self.bf)
                o = torch.sigmoid(xt @ self.Wox + h @ self.Woh + self.bo)

                c = g * i + c * f
                h = torch.tanh(c) * o

        out = h @ self.Wph + self.bp
        out = torch.softmax(out, dim=1)

        return out

