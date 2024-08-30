# stdlib
from typing import Any, Callable, List, Optional, Tuple


# third party
import torch
from torch import nn
from torch.nn.modules import MLP
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# utils imports
from .dataloader import CTGan_data_set
from .CTGan_utils import get_nolin_akt

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'



class Generator(nn.Module):
    
    def __init__(
        self, 
        generator_n_units_in,
        generator_n_units_conditional,
        generator_n_units_out, 
        generator_n_layers_hidden: int = 3, 
        generator_n_units_hidden: int = 4, 
        generator_nonlin: str = "leaky_relu", 
        generator_nonlin_out: str = "leaky_relu", 
        generator_batch_norm: bool = False, 
        generator_dropout: float = 0.001, 
        device: Any = DEVICE, 
    ):
        super().__init__()
        
        self.device = device

        self.net = nn.Sequential()

        generator_nonlin = get_nolin_akt(generator_nonlin)
        generator_nonlin_out = get_nolin_akt(generator_nonlin_out)

        # First Layer
        self.net.append(nn.Linear(generator_n_units_in + generator_n_units_conditional, generator_n_units_hidden))
        if generator_batch_norm:
            self.net.append(nn.BatchNorm1d(generator_n_units_hidden))

        self.net.append(generator_nonlin)
        self.net.append(nn.Dropout(generator_dropout))

        for n_layer in range(generator_n_layers_hidden - 1):
                self.net.append(nn.Linear(generator_n_units_hidden, generator_n_units_hidden))
                if generator_batch_norm:
                    self.net.append(nn.BatchNorm1d(generator_n_units_hidden))
                self.net.append(nn.LeakyReLU())
                self.net.append(nn.Dropout(generator_dropout))

        self.net.append(nn.Linear(generator_n_units_hidden, generator_n_units_out))
        self.net.append(generator_nonlin_out)


    def forward(self, x):
        return self.net(x)

    def generate(self, n_samples, condition): #todo das hier alles
        self.eval()
        with torch.no_grad():
            fake = self(n_samples, condition)
        
        return fake


class Discriminator(nn.Module):
    
    def __init__(
        self, 
        discriminator_n_units_in,
        discriminator_n_units_conditional,
        discriminator_n_layers_hidden: int = 3, 
        discriminator_n_units_hidden: int = 4, 
        discriminator_nonlin: str = "leaky_relu", 
        discriminator_batch_norm: bool = False, 
        discriminator_dropout: float = 0.001, 
        device: Any = DEVICE, 

    
    ):
        super().__init__()
        
        self.device = device
        self.net = nn.Sequential()

        generator_nonlin = get_nolin_akt(generator_nonlin)

        self.net.append(nn.Linear(discriminator_n_units_in + discriminator_n_units_conditional, discriminator_n_layers_hidden))
        if discriminator_batch_norm:
            self.net.append(nn.BatchNorm1d(discriminator_n_layers_hidden))

        self.net.append(generator_nonlin)
        self.net.append(nn.Dropout(discriminator_dropout))


        for n_layer in range(discriminator_n_layers_hidden - 1):
                self.net.append(nn.Linear(discriminator_n_layers_hidden, discriminator_n_layers_hidden))
                if discriminator_batch_norm:
                    self.net.append(nn.BatchNorm1d(discriminator_n_layers_hidden))
                self.net.append(nn.LeakyReLU())
                self.net.append(nn.Dropout(discriminator_dropout))

        self.net.append(nn.Linear(discriminator_n_layers_hidden, 1))
        self.net.append(nn.Sigmoid)


    def forward(self, x):
         return self.net(x).view(-1)


def label_real(size):
   return torch.ones(size)

def label_fake(size):
   return torch.zeros(size)

def get_results(y_hat_real, y_hat_fake):
    acc_real = (y_hat_real >= 0.5).float().sum().item()
    acc_fake = (y_hat_fake < 0.5).float().sum().item()
    y_hat_real = y_hat_real.sum().item()
    y_hat_fake = y_hat_fake.sum().item()
    return y_hat_real, y_hat_fake, acc_real, acc_fake


class CTGan(nn.Module):

    def __init__(
        self,

        n_units_latent: int,
        
        # Generator
        generator_n_layers_hidden: int = 2,
        generator_n_units_hidden: int = 250,
        generator_nonlin: str = "leaky_relu",
        generator_nonlin_out = None,
        generator_num_steps: int = 500,
        generator_batch_norm: bool = False,
        generator_dropout: float = 0,
        generator_lr: float = 2e-4,
        generator_weight_decay: float = 1e-3,
        generator_residual: bool = True,
        generator_opt_betas: tuple = (0.9, 0.999),
        generator_extra_penalties: list = [],  
        
        # discriminator
        discriminator_n_layers_hidden: int = 3,
        discriminator_n_units_hidden: int = 300,
        discriminator_nonlin: str = "leaky_relu",
        discriminator_num_steps: int = 1,
        discriminator_batch_norm: bool = False,
        discriminator_dropout: float = 0.1,
        discriminator_lr: float = 2e-4,
        discriminator_weight_decay: float = 1e-3,
        discriminator_opt_betas: tuple = (0.9, 0.999),
        discriminator_extra_penalties: list = [],
        
        # training
        batch_size: int = 64,
        random_state: int = None,
        clipping_value: int = 0,
        lambda_gradient_penalty: float = 10,
        lambda_identifiability_penalty: float = 0.1,
        
        n_epochs: int = 200,
        n_iter_print: int = 10,
        n_iter_checkpoint: int = None,
        early_stopping_patience: int = 20,
        early_stopping_patience_metric : Any = None,
        device: Any = DEVICE, 
    ):
        
        super().__init__()

        self.device = device

        self.n_units_latent = n_units_latent

        self.generator_extra_penalties = generator_extra_penalties # todo

        self.generator_n_layers_hidden= generator_n_layers_hidden
        self.generator_n_units_hidden= generator_n_units_hidden
        self.generator_nonlin= generator_nonlin
        self.generator_nonlin_out= generator_nonlin_out
        self.generator_batch_norm= generator_batch_norm
        self.generator_dropout= generator_dropout,

        self.generator_lr = generator_lr
        self.generator_weight_decay = generator_weight_decay
        self.generator_residual= generator_residual
        self.generator_opt_betas = generator_opt_betas

        self.discriminator_n_layers_hidden= discriminator_n_layers_hidden
        self.discriminator_n_units_hidden= discriminator_n_units_hidden
        self.discriminator_nonlin= discriminator_nonlin
        self.discriminator_batch_norm= discriminator_batch_norm
        self.discriminator_dropout= discriminator_dropout

        self.discriminator_lr = discriminator_lr
        self.discriminator_weight_decay = discriminator_weight_decay
        self.discriminator_opt_betas = discriminator_opt_betas
        self.discriminator_extra_penalties = discriminator_extra_penalties
       
        self.generator_num_steps = generator_num_steps
        self.discriminator_num_steps = discriminator_num_steps
        self.n_epochs = n_epochs
        self.n_iter_print = n_iter_print
        self.n_iter_checkpoint = n_iter_checkpoint
  
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_patience_metric = early_stopping_patience_metric

        self.batch_size = batch_size
        self.clipping_value = clipping_value
        self.lambda_gradient_penalty = lambda_gradient_penalty
        self.lambda_identifiability_penalty = lambda_identifiability_penalty
        
        self.random_state = random_state
    
    def fit(self, data_loader: CTGan_data_set):
        """

            Args: 
                X: CTGan_data_set
        
        """

        if not self.generator and self.discriminator:

            self.output_space = data_loader.output_space    
            self.n_units_conditional = 3 


            self.generator = Generator(
                generator_n_units_in= self.n_units_latent,
                generator_n_units_conditional= self.n_units_conditional,
                generator_n_units_out= self.output_space, 
                generator_n_layers_hidden= self.generator_n_layers_hidden, 
                generator_n_units_hidden= self.generator_n_units_hidden, 
                generator_nonlin= self.generator_nonlin, 
                generator_nonlin_out= self.generator_nonlin_out, 
                generator_batch_norm= self.generator_batch_norm, 
                generator_dropout= self.generator_dropout, 
                device= self.device, 
            )

            self.discriminator = Discriminator(
                discriminator_n_units_in= self.output_space,
                discriminator_n_units_conditional= self.n_units_conditional,
                discriminator_n_layers_hidden= self.discriminator_n_layers_hidden, 
                discriminator_n_units_hidden= self.discriminator_n_units_hidden, 
                discriminator_nonlin= self.discriminator_nonlin, 
                discriminator_batch_norm= self.discriminator_batch_norm, 
                discriminator_dropout= self.discriminator_dropout, 
                device= self.device, 
            )

            self.discriminator_nonlin_out = nn.BCELoss(reduction='sum')
            self.cond_cols = data_loader.cond_cols()
            
            self.cond_encoder = 1 #TODO
        
        generator = self.generator
        discriminator = self.discriminator

        data_loader = DataLoader(CTGan_data_set, batch_size=self.batch_size, shuffle=True)

        device = self.device

        optim_generator = torch.optim.Adam(generator, lr=self.generator_lr, betas = self.generator_opt_betasbetas)
        optim_discriminator = torch.optim.Adam(discriminator, lr=self.discriminator_lr, betas = self.discriminator_opt_betas)

        self.generator.train()
        self.discriminator.train()

        for epoch in range(1, self.n_epochs+1):

            for X in data_loader:

                X_real = X.to(device) 

                # TODO hier die conditions von den echten daten noch laden und schauen wie man das macht mit dennen von den generierten cond vlt einfach auch die conditions von den echten daten rein dann muss er das selbe generieren und wenn nicht hat der discriminator ein einfaches spiel 

                y_real = label_real(self.batch_size).to(device)
                y_fake = label_fake(self.batch_size).to(device)

                #*** train discriminator ******************************************
                for k in range(self.discriminator_num_steps):
                    # -- train with real data
                    y_hat_real = discriminator(X_real)
                    loss_real = self.discriminator_nonlin_out(y_hat_real, y_real)

                    # --- train with fake data
                    X_fake = generator.generate(self.batch_size) #TODO condition mit hinein 
                    y_hat_fake = discriminator(X_fake)
                    if self.clipping_value > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.discriminator.parameters(), self.clipping_value
                        )
                    loss_fake = self.discriminator_nonlin_out(y_hat_fake, y_fake)

                    # --- update 
                    optim_discriminator.zero_grad()
                    loss = 0.5*(loss_real + loss_fake)
                    loss.backward()
                    if self.clipping_value > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.discriminator.parameters(), self.clipping_value
                        )
                    optim_discriminator.step()

                    # --- compile results (keep last metrics)
                    result = [loss.item(), *get_results(y_hat_real, y_hat_fake)]

                #*** train generator ***********************************************
                for k in range(self.generator_num_steps):
                    # --- train with fake images
                    X_fake = generator.generate(self.batch_size)
                    y_hat = discriminator(X_fake)
                    loss = self.discriminator_nonlin_out(y_hat, y_real)

                    optim_generator.zero_grad()
                    loss.backward()
                    if self.clipping_value > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.generator.parameters(), self.clipping_value
                        )

                    optim_generator.step()

                    # --- result: keep last loss
                    loss_g = loss.item()

                result.insert(1, loss_g)
            
            if not epoch % self.n_iter_print:
                pass #TODO
                
            if not epoch % self.n_iter_checkpoint:
                pass #TODO

    
    def gen(self, num_samples, cond_df):
        
        if cond_df.empty is True:
            gen_data = self.generator.generate(num_samples)
            return gen_data
        
        else: 
            if cond_df.shape[1] != len(self.n_units_conditional):
                return
                #Todo 
            
            if set(cond_df.columns.values.tolist()) != set(self.cond_cols):
                return
                #TODO
            
            gen_data = self.generator.generate(count=cond_df.shape[0], cond=cond_df).dataframe()
            
            return gen_data