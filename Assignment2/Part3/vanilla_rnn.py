from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import math


class VanillaRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(VanillaRNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.Whx = nn.Parameter(torch.empty(input_dim, hidden_dim))
        self.Whh = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.bh = nn.Parameter(torch.empty(hidden_dim))

        self.Wph = nn.Parameter(torch.empty(hidden_dim, output_dim))
        self.bo = nn.Parameter(torch.empty(output_dim))

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
                h = torch.tanh(xt @ self.Whx + self.bh)
            else:
                h = torch.tanh(xt @ self.Whx + h @ self.Whh + self.bh)

        out = h @ self.Wph + self.bo
        out = nn.functional.softmax(out, dim=1)
        return out