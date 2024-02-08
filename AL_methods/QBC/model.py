import torch
from torch.nn import Linear
from torch.nn import Sigmoid, ReLU
from torch.nn import Module
import numpy as np
import random
from torch.nn import Sigmoid, ReLU, BatchNorm1d, ELU

NN_setting = [50, 30, 15] 


class MLP(Module):
    # define model elements
    def __init__(self, n_inputs):
        super().__init__()

        # input to first hidden layer
        self.hidden1 = Linear(n_inputs, NN_setting[0])
        self.act1 = ELU()

        # second hidden layer
        self.hidden2 = Linear(NN_setting[0], NN_setting[1])
        self.act2 = ELU()

        # second hidden layer
        self.hidden3 = Linear(NN_setting[1], NN_setting[2])
        self.act3 = ELU()

        self.hidden4 = Linear(NN_setting[2], 1)

    # forward propagate input
    def forward(self, X):

        X = self.hidden1(X)
        X = self.act1(X)

        X = self.hidden2(X)
        X = self.act2(X)

        X = self.hidden3(X)
        X = self.act3(X)

        X = self.hidden4(X)

        return X