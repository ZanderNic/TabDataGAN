from torch import nn
import numpy as np
import time

def format_time(seconds):
    minutes = int((seconds % 3600) // 60)
    seconds_int = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{minutes:02d}min:{seconds_int:02d}s:{milliseconds:03d}ms"

def format_time_with_h(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds_int = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02d}h:{minutes:02d}min:{seconds_int:02d}s:{milliseconds:03d}ms"


def get_nolin_akt(name : str):
    if name == "relu":
        return nn.ReLU()
    elif name == "leaky_relu":
        return nn.LeakyReLU()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "softmax":
        return nn.Softmax()
    else:
        raise ValueError(f"Unknown activation function: {name}")
    

def get_loss_function(name : str):
    if name == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif name == "mse":
        return nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss function: {name}")

