from torch import nn

def get_nolin_akt(name):
    if name == 'relu':
        return nn.ReLU
    elif name == 'leaky_relu':
        return nn.LeakyReLU
    elif name == 'sigmoid':
        return nn.Sigmoid
    elif name == 'tanh':
        return nn.Tanh
    else:
        raise ValueError(f"Unknown activation function: {name}")