from typing import Any, Callable, List, Optional, Tuple, Dict

# third party
import torch
from torch import nn

# Projects imports
from TabDataGAN.Model.Gans._gan_utils import get_nolin_act, find_w, init_weights


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' # If device is not set try to use cuda else cpu


class Generator(nn.Module):
    def __init__(
        self,
        generator_n_units_in: int,
        generator_n_units_conditional: int,
        generator_n_units_out: int,
        units_per_column: List[int],
        columns_in_order: List[str],
        categorical_columns: List[str],
        numerical_columns: List[str],
        units_per_col: List[float],
        generator_n_layers_hidden: int = 3,
        generator_n_units_hidden: int = 200,
        generator_nonlin: str = "leaky_relu",
        generator_nonlin_out_num: str = "tanh",
        generator_nonlin_out_cat: str = "softmax",
        generator_batch_norm: bool = False,
        generator_dropout: float = 0.001,
        device: Any = DEVICE,
    ):
        super().__init__()

        self.n_units_lat = generator_n_units_in
        self.device = device
        self.units_per_column = units_per_column
        self.columns_in_order = columns_in_order
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.units_per_col = units_per_col  

        # Activation functions for outputs
        self.generator_nonlin_out_num = get_nolin_act(generator_nonlin_out_num)
        self.generator_nonlin_out_cat = get_nolin_act(generator_nonlin_out_cat)

        layers = []
        generator_nonlin = get_nolin_act(generator_nonlin)

        layers.append(nn.Linear(generator_n_units_in + generator_n_units_conditional, generator_n_units_hidden))
        if generator_batch_norm:
            layers.append(nn.BatchNorm1d(generator_n_units_hidden))
        layers.append(generator_nonlin)
        layers.append(nn.Dropout(generator_dropout))

        for _ in range(generator_n_layers_hidden - 1):
            layers.append(nn.Linear(generator_n_units_hidden, generator_n_units_hidden))
            if generator_batch_norm:
                layers.append(nn.BatchNorm1d(generator_n_units_hidden))
            layers.append(generator_nonlin)
            layers.append(nn.Dropout(generator_dropout))

        layers.append(nn.Linear(generator_n_units_hidden, generator_n_units_out))
        # No final activation here will be apply per feature in forward() 

        self.net = nn.Sequential(*layers)
        self.net.apply(init_weights)

    def forward(self, x):
        x = self.net(x)
        # Now split the output and apply activation functions
        outputs = []
        start = 0
        for units, col_name in zip(self.units_per_column, self.columns_in_order):
            end = start + units
            feature_output = x[:, start:end]
            if col_name in self.categorical_columns:
                feature_output = self.generator_nonlin_out_cat(feature_output) 
              

            elif col_name in self.numerical_columns:
                # Check if the Encoding was Min_max or CTGAN by checking the units per coll
                numerical_part = feature_output[:, 0] # The fist output will always be numeric 
                numerical_part = self.generator_nonlin_out_num(numerical_part)

                if units > 1:
                    one_hot_part = feature_output[:, 1:]
                    one_hot_part = self.generator_nonlin_out_cat(one_hot_part) 
                    numerical_part = numerical_part.unsqueeze(1)
                    feature_output = torch.cat([numerical_part, one_hot_part], dim=1)
                else:
                    feature_output = numerical_part.unsqueeze(1)         
            else:
                print("something wen't wrong D:")
                return ValueError("Error in the Generator forward call")
            outputs.append(feature_output)
            start = end
        x = torch.cat(outputs, dim=1)
        return x

    def generate(self, n_samples, condition):
        self.eval()
        with torch.no_grad():
            noise = torch.randn(n_samples, self.n_units_lat, device=self.device)
            condition = condition.to(self.device)
            generator_input = torch.cat([noise, condition], dim=1)
            gen_data = self.forward(generator_input)
        return gen_data


class Conv_Generator(nn.Module):
    """
    An implementation of a Generator that uses convolutional layers for tabular data.
    """
    def __init__(
        self,
        generator_n_units_in: int,
        generator_n_units_conditional: int,
        generator_n_units_out: int,
        units_per_column: List[int],
        columns_in_order: List[str],
        categorical_columns: List[str],
        numerical_columns: List[str],
        units_per_col: List[int],
        generator_n_layers_hidden: int = 5,
        generator_n_kernels_hidden: int = 128,
        generator_nonlin: str = "leaky_relu",
        generator_nonlin_out_num: str = "tanh",
        generator_nonlin_out_cat: str = "softmax",
        generator_batch_norm: bool = False,
        generator_dropout: float = 0.001,
        device: Any = DEVICE,
    ):
        super().__init__()

        self.n_units_lat = generator_n_units_in
        self.device = device
        self.units_per_column = units_per_column
        self.columns_in_order = columns_in_order
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.units_per_col = units_per_col

        # Activation functions for outputs
        self.generator_nonlin_out_num = get_nolin_act(generator_nonlin_out_num)
        self.generator_nonlin_out_cat = get_nolin_act(generator_nonlin_out_cat)

        generator_nonlin = get_nolin_act(generator_nonlin)

        # Find the size of the 2D tensor (w, h) where w = h and w**2 can fit the input
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
        # No final activation here; will apply per feature in forward()

        self.net = nn.Sequential(*layers)
        self.net.apply(init_weights)

    def forward(self, x):
        if self.extra > 0:
            padding = torch.zeros((x.shape[0], self.extra), device=self.device)
            x = torch.cat([x, padding], dim=1)

        x = x.view(x.shape[0], 1, self.w, self.w)
        x = self.net(x)

        # Now split the output and apply activation functions
        outputs = []
        start = 0
        for units, col_name in zip(self.units_per_column, self.columns_in_order):
            end = start + units
            feature_output = x[:, start:end]
            if col_name in self.categorical_columns:
                feature_output = self.generator_nonlin_out_cat(feature_output, dim=1) 

            elif col_name in self.numerical_columns:
                # Check if the Encoding was MinMax or CTGAN by checking the units per column
                numerical_part = feature_output[:, 0]  # The first output will always be numeric
                numerical_part = self.generator_nonlin_out_num(numerical_part)

                if units > 1:
                    one_hot_part = feature_output[:, 1:]
                    one_hot_part = self.generator_nonlin_out_cat(one_hot_part, dim=1)
                    numerical_part = numerical_part.unsqueeze(1)
                    feature_output = torch.cat([numerical_part, one_hot_part], dim=1)
                else:
                    feature_output = numerical_part.unsqueeze(1)
            else:
                print("Something went wrong in Conv_Generator forward method.")
                raise ValueError("Error in the Conv_Generator forward call")
            outputs.append(feature_output)
            start = end
        x = torch.cat(outputs, dim=1)
        return x

    def generate(self, n_samples, condition):
        self.eval()
        with torch.no_grad():
            noise = torch.randn(n_samples, self.n_units_lat, device=self.device)
            condition = condition.to(self.device)
            generator_input = torch.cat([noise, condition], dim=1)
            gen_data = self.forward(generator_input)
        return gen_data