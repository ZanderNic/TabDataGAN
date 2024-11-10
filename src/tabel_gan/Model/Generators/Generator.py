from typing import Any, Callable, List, Optional, Tuple

# third party
import torch
from torch import nn

# Projects imports
from ..Gans._gan_utils import get_nolin_act, find_w, init_weights


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' # If device is not set try to use cuda else cpu


class Conv_Generator(nn.Module):
    """
    A implementation of a Generator that uses Conv but still for Tabular data
    """
    def __init__(
        self, 
        generator_n_units_in,
        generator_n_units_conditional,
        generator_n_units_out, 
        generator_n_layers_hidden: int = 5, 
        generator_n_kernels_hidden: int = 128,  
        generator_nonlin: str = "leaky_relu", 
        generator_nonlin_out: str = "leaky_relu", 
        generator_batch_norm: bool = False, 
        generator_dropout: float = 0.001, 
        device: Any = DEVICE, 
    ):
        super().__init__()
        
        self.n_units_lat = generator_n_units_in
        self.device = device
        
        generator_nonlin = get_nolin_act(generator_nonlin)
        generator_nonlin_out = get_nolin_act(generator_nonlin_out)
        
        # Find the size of the 2D tensor (w, h) where w = h and w = 2 * n and n element N that can fit the input
        self.w, self.extra = find_w(generator_n_units_in + generator_n_units_conditional)
        
        layers = []
        in_channels = 1  
        out_channels = generator_n_kernels_hidden
        
        # Add convolutional layers
        for _ in range(generator_n_layers_hidden):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            if generator_batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(generator_nonlin)
            layers.append(nn.Dropout(generator_dropout))
            in_channels = out_channels
        
        final_conv_output_size = out_channels * self.w * self.w
        
        layers.append(nn.Flatten())
        layers.append(nn.Linear(final_conv_output_size, generator_n_units_out))
        layers.append(generator_nonlin_out)
        
        self.net = nn.Sequential(*layers)
        self.net.apply(init_weights) 


    def forward(self, x):
        if self.extra > 0:
            padding = torch.zeros((x.shape[0], self.extra), device=self.device)
            x = torch.cat([x, padding], dim=1)
        
        x = x.view(x.shape[0], 1, self.w, self.w)
        return self.net(x)  # Use self.net for all layers
    
    def generate(self, n_samples, condition): 
        self.eval()  
        with torch.no_grad():
            noise = torch.randn(n_samples, self.n_units_lat, device=self.device).to(self.device) 
            condition = condition.to(self.device)
            generator_input = torch.cat([noise, condition], dim=1) 
            gen_data = self.forward(generator_input) 
        return gen_data


class Generator(nn.Module):
    
    def __init__(
        self, 
        generator_n_units_in,
        generator_n_units_conditional,
        generator_n_units_out, 
        generator_n_layers_hidden: int = 3, 
        generator_n_units_hidden: int = 200, 
        generator_nonlin: str = "leaky_relu", 
        generator_nonlin_out: str = "leaky_relu", 
        generator_batch_norm: bool = False, 
        generator_dropout: float = 0.001, 
        device: Any = DEVICE, 
    ):
        super().__init__()
        
        self.n_units_lat = generator_n_units_in
        self.device = device
        
        layers = []
        generator_nonlin = get_nolin_act(generator_nonlin)
        generator_nonlin_out = get_nolin_act(generator_nonlin_out)
        
        layers.append(nn.Linear(generator_n_units_in + generator_n_units_conditional, generator_n_units_hidden))
        if generator_batch_norm:
            layers.append(nn.BatchNorm1d(generator_n_units_hidden))
        layers.append(generator_nonlin)
        layers.append(nn.Dropout(generator_dropout))

        for _ in range(generator_n_layers_hidden - 1):
            layers.append(nn.Linear(generator_n_units_hidden, generator_n_units_hidden))
            if generator_batch_norm:
                layers.append(nn.BatchNorm1d(generator_n_units_hidden))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(generator_dropout))

        layers.append(nn.Linear(generator_n_units_hidden, generator_n_units_out))
        layers.append(generator_nonlin_out)
        
        self.net = nn.Sequential(*layers)
        self.net.apply(init_weights) 


    def forward(self, x): 
        return self.net(x)  

    def generate(self, n_samples, condition): 
        self.eval()  
        with torch.no_grad():
            noise = torch.randn(n_samples, self.n_units_lat, device=self.device).to(self.device) 
            condition = condition.to(self.device)
            generator_input = torch.cat([noise, condition], dim=1) 
            gen_data = self.forward(generator_input) 
        return gen_data

""""""
class Generator(nn.Module):
    
    def __init__(
        self, 
        generator_n_units_in,
        generator_n_units_conditional,
        generator_n_units_out, 
        generator_n_layers_hidden: int = 3, 
        generator_n_units_hidden: int = 200, 
        generator_nonlin: str = "leaky_relu", 
        generator_nonlin_out: str = "leaky_relu", 
        generator_batch_norm: bool = False, 
        generator_dropout: float = 0.001, 
        device: Any = DEVICE, 
    ):
        super().__init__()
        
        self.n_units_lat = generator_n_units_in

        self.device = device
        self.net = nn.Sequential()

        generator_nonlin = get_nolin_act(generator_nonlin)
        generator_nonlin_out = get_nolin_act(generator_nonlin_out)
        
        self.net.append(nn.Linear(generator_n_units_in + generator_n_units_conditional, generator_n_units_hidden))
        if generator_batch_norm:
            self.net.append(nn.BatchNorm1d(generator_n_units_hidden))

        self.net.append(generator_nonlin)
        self.net.append(nn.Dropout(generator_dropout))

        for _ in range(generator_n_layers_hidden - 1):
                self.net.append(nn.Linear(generator_n_units_hidden, generator_n_units_hidden))
                if generator_batch_norm:
                    self.net.append(nn.BatchNorm1d(generator_n_units_hidden))
                self.net.append(nn.LeakyReLU())
                self.net.append(nn.Dropout(generator_dropout))

        self.net.append(nn.Linear(generator_n_units_hidden, generator_n_units_out))
        self.net.append(generator_nonlin_out)


    def forward(self, x): 
        self.net.to(self.device)
        return self.net(x)

    def generate(self, n_samples, condition): 
        self.eval()  
        with torch.no_grad():
            noise = torch.randn(n_samples, self.n_units_lat, device=self.device).to(self.device) 
            condition = condition.to(self.device)
            generator_input = torch.cat([noise, condition], dim=1) 
            gen_data = self.forward(generator_input) 
        return gen_data
""""""