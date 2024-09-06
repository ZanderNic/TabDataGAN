import copy

from torch import nn
import numpy as np


def get_nolin_akt(name : str):
    if name == "relu":
        return nn.ReLU
    elif name == "leaky_relu":
        return nn.LeakyReLU
    elif name == "sigmoid":
        return nn.Sigmoid
    elif name == "tanh":
        return nn.Tanh
    elif name == "softmax":
        return nn.Softmax
    else:
        raise ValueError(f"Unknown activation function: {name}")
    

def get_loss_function(name : str):
    if name == "cross_entropy":
        return nn.CrossEntropyLoss
    elif name == "mse":
        return nn.MSELoss
    else:
        raise ValueError(f"Unknown loss function: {name}")

