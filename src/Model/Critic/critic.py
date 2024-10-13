#std lib
from typing import Any

# 3 party
import torch
from torch import nn

# Projekt imports 
from ..Gans._gan_utils import get_nolin_act, find_w, init_weights

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'



class Discriminator(nn.Module):
    """
    Discriminator or Critic used for a WGAN that uses a MLP
    """
    def __init__(
        self, 
        discriminator_n_units_in: int,
        discriminator_n_layers_hidden: int = 3, 
        discriminator_n_units_hidden: int = 100, 
        discriminator_nonlin: str = "leaky_relu", 
        discriminator_batch_norm: bool = False, 
        discriminator_dropout: float = 0.001, 
        device: Any = DEVICE, 
    ):
        super().__init__()
        
        self.device = device
        self.net = nn.Sequential()

        discriminator_nonlin = get_nolin_act(discriminator_nonlin)

        self.net.append(nn.Linear(discriminator_n_units_in, discriminator_n_units_hidden))
        if discriminator_batch_norm:
            self.net.append(nn.BatchNorm1d(discriminator_n_units_hidden))

        self.net.append(discriminator_nonlin)
        self.net.append(nn.Dropout(discriminator_dropout))

        for _ in range(discriminator_n_layers_hidden - 1):
            self.net.append(nn.Linear(discriminator_n_units_hidden, discriminator_n_units_hidden))
            if discriminator_batch_norm:
                self.net.append(nn.BatchNorm1d(discriminator_n_units_hidden))
            self.net.append(discriminator_nonlin)
            self.net.append(nn.Dropout(discriminator_dropout))

        self.net.append(nn.Linear(discriminator_n_units_hidden, 1))  

        self.net.apply(init_weights) 

    def forward(self, x):
        x = x.to(self.device)
        return self.net(x).view(-1)


class Conv_Discriminator(nn.Module):
    """
    Discriminator used for a WGAN that uses Conv Layer (similar to Conv_Critic).
    """
    def __init__(
        self, 
        discriminator_n_units_in: int,
        discriminator_n_layers_hidden: int = 3, 
        discriminator_n_kernels_hidden: int = 64, 
        discriminator_nonlin: str = "leaky_relu", 
        discriminator_batch_norm: bool = False, 
        discriminator_dropout: float = 0.3, 
        device: Any = DEVICE, 
    ):
        super().__init__()
        
        self.device = device
        discriminator_nonlin = get_nolin_act(discriminator_nonlin)
        
        # Find the appropriate size for 2D tensors
        self.w, self.extra = find_w(discriminator_n_units_in)

        layers = []
        in_channels = 1  
        out_channels = discriminator_n_kernels_hidden
        
        # Add convolutional layers
        for _ in range(discriminator_n_layers_hidden):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            if discriminator_batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(discriminator_nonlin)
            layers.append(nn.Dropout(discriminator_dropout))
            in_channels = out_channels
        
        final_conv_output_size = out_channels * self.w * self.w
        
        layers.append(nn.Flatten())
        layers.append(nn.Linear(final_conv_output_size, 1))
        
        self.net = nn.Sequential(*layers)
        self.net.apply(init_weights)  

    def forward(self, x):
        x = x.to(self.device)

        if self.extra > 0:
            padding = torch.zeros((x.shape[0], self.extra), device=self.device)
            x = torch.cat([x, padding], dim=1)
        
        x = x.view(x.shape[0], 1, self.w, self.w)

        return self.net(x).view(-1)  
