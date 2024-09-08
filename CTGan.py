# stdlib
from typing import Any, Callable, List, Optional, Tuple


# third party
import torch
from torch import nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, random_split



# Projects imports
from dataset import CTGan_data_set
from Classifier import Classifier
from CTGan_utils import get_nolin_akt, get_loss_function
from encoder_condition import Cond_Encoder
from encoder_data import Data_Encoder

# Default parameter
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' # If device is not set try to use cuda else cpu




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

        for _ in range(generator_n_layers_hidden - 1):
                self.net.append(nn.Linear(generator_n_units_hidden, generator_n_units_hidden))
                if generator_batch_norm:
                    self.net.append(nn.BatchNorm1d(generator_n_units_hidden))
                self.net.append(nn.LeakyReLU())
                self.net.append(nn.Dropout(generator_dropout))

        self.net.append(nn.Linear(generator_n_units_hidden, generator_n_units_out))
        self.net.append(generator_nonlin_out)


    def forward(self, x): #TODO hier machen das man halt auch einen batch druch geben kann 
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
        discriminator_n_units_hidden: int = 100, 
        discriminator_nonlin: str = "leaky_relu", 
        discriminator_batch_norm: bool = False, 
        discriminator_dropout: float = 0.001, 
        device: Any = DEVICE, 
    ):
        super().__init__()
        
        self.device = device
        self.net = nn.Sequential()

        discriminator_nonlin = get_nolin_akt(discriminator_nonlin)

        self.net.append(nn.Linear(discriminator_n_units_in + discriminator_n_units_conditional, discriminator_n_units_hidden))
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
        self.net.append(nn.Sigmoid())


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
        generator_nonlin: str = "relu",
        generator_nonlin_out = "sigmoid", # This function given here should return a number between ]0; 1] because the data is processed and scaled between ]0; 1]
        generator_num_steps: int = 500,
        generator_batch_norm: bool = False,
        generator_dropout: float = 0,
        generator_lr: float = 2e-4,
        generator_weight_decay: float = 1e-3,
        generator_residual: bool = True,
        generator_opt_betas: tuple = (0.9, 0.999),
        generator_extra_penalties: list = [],  #TODO 
        
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
        discriminator_extra_penalties: list = [], #TODO

        # Classifier
        classifier_n_layers_hidden: int = 3, 
        classifier_n_units_hidden: int = 4, 
        classifier_nonlin: str = "leaky_relu", 
        classifier_n_iter : int = 150,
        classifier_lr : float = 0.001,
        classifier_batch_norm: bool = False, 
        classifier_dropout: float = 0.001,
        classifier_opt_betas: tuple = (0.9, 0.99),
        classifier_loss: str = "cross_entropy",
        classifier_patience : int = 10,
        classifier_batch_size : int = 128,
    

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

        max_categorical_encoder= 30 # If a categorical column has more than 30 classes no One Hot encoding is used a general transform for those coloumns will be used (scale them betwean [0 and 1]) 
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
        self.generator_dropout= generator_dropout

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

        self.classifier_n_layers_hidden = classifier_n_layers_hidden
        self.classifier_n_units_hidden = classifier_n_units_hidden
        self.classifier_nonlin = classifier_nonlin
        self.classifier_batch_norm = classifier_batch_norm
        self.classifier_dropout = classifier_dropout
        self.classifier_opt_betas = classifier_opt_betas
        self.classifier_loss = classifier_loss 
        self.classifier_n_iter = classifier_n_iter
        self.classifier_lr = classifier_lr
        self.classifier_patience = classifier_patience
        self.classifier_batch_size = classifier_batch_size

        self.generator_num_steps = generator_num_steps
        self.discriminator_num_steps = discriminator_num_steps
        self.n_epochs = n_epochs
        self.n_iter_print = n_iter_print
        self.n_iter_checkpoint = n_iter_checkpoint
  
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_patience_metric = early_stopping_patience_metric

        self.batch_size = batch_size
        self.clipping_value = clipping_value
        self.lambda_gradient_penalty = lambda_gradient_penalty #TODO
        self.lambda_identifiability_penalty = lambda_identifiability_penalty #TODO
        
        self.random_state = random_state
        self.max_categorical_encoder = max_categorical_encoder

        self.generator = None
        self.discriminator = None


    def valid_generator_extra_penalties(generator_extra_penalties):
        """
            takes the list generator_extra_penalties and returns all the implemented ones
        """
        #TODO
    
    def preprocess_and_load_data(self, ctgan_data_set, batch_size)-> DataLoader:
        data = ctgan_data_set.dataframe()
        conditiond_encoded = self.cond_encoder(data)
        data_encoded = self.data_encoder(data)

        dataset = torch.utils.data.TensorDataset(
            torch.tensor(data_encoded.values, dtype=torch.float32), 
            conditiond_encoded
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def preprocess_and_load_train_test(self, ctgan_data_set, batch_size, train_size = 0.8): #-> Tuple(torch.dataloader()): #TODO
        data = ctgan_data_set.dataframe()
        conditiond_encoded = self.cond_encoder(data)
        data_encoded = self.data_encoder(data)
        
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(data_encoded.values, dtype=torch.float32), 
            conditiond_encoded
        )
        train_dataset, test_dataset = random_split(dataset, [train_size, 1-train_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader =  DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        return train_loader, test_loader


    def train_classifier(self, dataset : CTGan_data_set, train_size:float = 0.8):

        self.classifier_batch_size = 1 #Todo 

        train_loader, test_loader = self.preprocess_and_load_train_test(ctgan_data_set=dataset, batch_size=self.classifier_batch_size, train_size=train_size)
        self.classifier.fit(train_loader=train_loader, test_loader=test_loader, lr=self.classifier_lr, opt_betas=self.classifier_opt_betas, epochs = self.classifier_n_iter, patience=self.classifier_patience) # loss=self.classifier_loss

        



    def fit(self, ctgan_data_set: CTGan_data_set):
        """

            Args: 
                X: CTGan_data_set that was created given a condition if you want to use one for Data generation
        
        """

        if not (self.generator and self.discriminator and self.classifier):

            self.cond_encoder = Cond_Encoder(
                ctgan_data_set.dataframe(),
                categorical_columns=ctgan_data_set.cat_cols(),
                numeric_columns=ctgan_data_set.num_cols(),
                ordinal_columns=ctgan_data_set.ord_cols()
            )
            self.data_encoder = Data_Encoder(
                ctgan_data_set.dataframe(),
                categorical_columns=ctgan_data_set.cat_cols(),
                numeric_columns=ctgan_data_set.num_cols(),
                ordinal_columns=ctgan_data_set.ord_cols()
            )

            self.output_space =  self.data_encoder.encodet_dim() 
            self.n_units_conditional =  self.cond_encoder.condition_dim() 

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

            self.classifier = Classifier(
                classifier_n_units_in = self.output_space,
                classifier_n_units_out_per_category = self.cond_encoder.get_units_per_column(),
                classifier_n_layers_hidden =  self.classifier_n_layers_hidden, 
                classifier_n_units_hidden = self.classifier_n_units_hidden, 
                classifier_nonlin= self.classifier_nonlin, 
                classifier_batch_norm= self.classifier_batch_norm, 
                classifier_dropout= self.classifier_dropout,
                device= self.device,
                loss= self.classifier_loss,
            )

            # self.discriminator_nonlin_out = nn.BCELoss(reduction='sum') #TODO das ist nicht das discriminator nolin out
            self.cond_cols = ctgan_data_set.cond_cols()     # conditional columns 
            self.cat_cols = ctgan_data_set.cat_cols()       # categorical columns
            self.num_cols = ctgan_data_set.num_cols()       # numeric columns 
            self.ord_col = ctgan_data_set.ord_cols()        # ordinal columns that should be given to the data_set

        
        # for better readability
        generator = self.generator
        discriminator = self.discriminator
        classifier = self.classifier
        device = self.device


        #TODO Del this lin
        self.classifier_n_iter = 1

        data_loader = self.preprocess_and_load_data(ctgan_data_set, self.batch_size)

        self.train_classifier(ctgan_data_set)

        optim_generator = torch.optim.Adam(generator.net, lr=self.generator_lr, betas = self.generator_opt_betas)
        optim_discriminator = torch.optim.Adam(discriminator.net, lr=self.discriminator_lr, betas = self.discriminator_opt_betas)

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
                    

                    X_fake = generator.forward(self.batch_size) #TODO hier halt einen noise vektor reingeben der Batch_size mal latent space hat so  
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

    
    def gen(self, num_samples = None, cond_df = pd.DataFrame):

        if num_samples is None and cond_df.empty:
            raise ValueError("Please provide either num_samples or the cond_df to generate data")
        
        if cond_df.empty is True:
            gen_data_raw = self.generator.generate(num_samples)
            gen_data =self.data_encoder.inverse_prep(gen_data_raw)
            return gen_data
        
        else: 
            if cond_df.shape[1] != len(self.n_units_conditional):
                return #Todo
                 
            
            if set(cond_df.columns.values.tolist()) != set(self.cond_cols):
                return #TODO
                
            
            gen_data_raw = self.generator.generate(count=cond_df.shape[0], cond=cond_df).dataframe()
            gen_data =self.data_encoder.inverse_prep(gen_data_raw)

            
            return gen_data
    


    def save(self, save_path):
        pass #TODO https://pytorch.org/tutorials/beginner/saving_loading_models.html save multible state dicts append condition coloumns, self.cond_encoder somehow seld.data_ecoder also somehow

    def load(self, path):
        pass #TODO model.load_state_dict(torch.load(PATH, weights_only=True, map_location="cuda:0"))  # it is not so easy to load a model trained on cuda on your cpu 