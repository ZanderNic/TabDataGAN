# stdlib
from typing import Any, Callable, List, Optional, Tuple
import time

# third party
import torch
from torch import nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, random_split


# Projects imports
from .Base_Gan import Base_CTGan

from ..Generators.Generator import Generator
from ..Critic.critic import Discriminator

from ...Data.dataset import CTGan_data_set
from ..Extra_Pen.Classifier import Classifier
from ._gan_utils import get_nolin_act, get_loss_function, format_time, format_time_with_h, gradient_penalty, wasserstein_loss
from ...Data.encoder_condition import Cond_Encoder
from ...Data.encoder_data import Data_Encoder




# Default parameter
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' # If device is not set try to use cuda else cpu



class WCTGan(Base_CTGan):
    """
        WCTGan with extra classifier loss


        Classifier:  
            the classifier is used to predict the condition, it is trained on the real data and later used for classifiing the conditon for the generated data
            to get a nother loss
    """

    def __init__(
        self,

        n_units_latent: int = 300, 
        
        # Generator
        generator_class: nn.Module = Generator,
        generator_n_layers_hidden: int = 3,
        generator_nonlin: str = "relu",
        generator_nonlin_out: str = "sigmoid", # This function given here should return a number between ]0; 1] because the data is processed and scaled between ]0; 1]
        generator_batch_norm: bool = False,
        generator_dropout: float = 0,
        generator_lr: float = 0.0001,
        generator_weight_decay: float = 0.0001,
        generator_residual: bool = True,
        generator_opt_betas: tuple = (0.5, 0.99),
        generator_extra_penalties: list = [],  #TODO 
        
        # discriminator
        discriminator_class: nn.Module = Discriminator,
        discriminator_n_layers_hidden: int = 3,
        discriminator_nonlin: str = "leaky_relu",
        discriminator_batch_norm: bool = False,
        discriminator_dropout: float = 0,
        discriminator_lr: float =  0.0001,
        discriminator_weight_decay: float = 0.0001,
        discriminator_opt_betas: tuple = (0.5, 0.99),
        discriminator_extra_penalties: list = [], #TODO

        # Classifier
        classifier_n_layers_hidden: int = 3, 
        classifier_n_units_hidden: int = 150, 
        classifier_nonlin: str = "leaky_relu", 
        classifier_n_iter : int = 1000,
        classifier_lr : float = 0.001,
        classifier_batch_norm: bool = False, 
        classifier_dropout: float = 0.1,
        classifier_opt_betas: tuple = (0.9, 0.99),
        classifier_loss: str = "cross_entropy",
        classifier_patience : int = 25,
        classifier_batch_size : int = 128,

        # training
        batch_size: int = 128,
        random_state: int = None,
        device: Any = DEVICE,
        lambda_condition_loss_weight = 1,

        max_categorical_encoder= 30, # If a categorical column has more than 30 classes no One Hot encoding is used a general transform for those coloumns will be used (scale them betwean [0 and 1]) 
    
        **kwargs
    ):
        
        super().__init__()

        self.device = device

        self.n_units_latent = n_units_latent


        self.generator_class = generator_class
        self.generator_extra_penalties = generator_extra_penalties # todo
        

        self.generator_n_layers_hidden= generator_n_layers_hidden
        self.generator_nonlin= generator_nonlin
        self.generator_nonlin_out= generator_nonlin_out
        self.generator_batch_norm= generator_batch_norm
        self.generator_dropout= generator_dropout
        self.generator_lr = generator_lr
        self.generator_weight_decay = generator_weight_decay
        self.generator_residual= generator_residual
        self.generator_opt_betas = generator_opt_betas


        self.discriminator_class= discriminator_class
        
        self.discriminator_n_layers_hidden= discriminator_n_layers_hidden
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

        self.lambda_condition_loss_weight = lambda_condition_loss_weight  # weighting of the conditional loss that is determit by the classifier

        self.batch_size = batch_size
                
        self.random_state = random_state
        self.max_categorical_encoder = max_categorical_encoder

        self.kwargs = kwargs

        self.generator = None
        self.discriminator = None


    def valid_generator_extra_penalties(generator_extra_penalties):
        """
            takes the list generator_extra_penalties and returns all the implemented ones
        """
        valid_extra_pen = ["cond_classifier", "cond_match"]
        
        generator_extra_penalties = []

        for extra_peneltie in generator_extra_penalties:
            if extra_peneltie in valid_extra_pen:
                pass

    

    def preprocess_and_load_data(self, ctgan_data_set, batch_size)-> DataLoader:
        data = ctgan_data_set.dataframe()
        conditiond_encoded = self.cond_encoder.get_cond_from_data(data)
        data_encoded = self.data_encoder(data)

        dataset = torch.utils.data.TensorDataset(
            data_encoded, 
            conditiond_encoded
        )

        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def preprocess_and_load_data_train_test(self, ctgan_data_set, batch_size, train_size = 0.8):
        """
        
        """
        data = ctgan_data_set.dataframe()
        conditiond_encoded = self.cond_encoder.get_cond_from_data(data)
        data_encoded = self.data_encoder(data)

        dataset = torch.utils.data.TensorDataset(
            data_encoded, 
            conditiond_encoded
        )

        train_size = int(train_size * len(dataset))  
        val_size = len(dataset) - train_size  

        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader =  DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        return train_loader, val_loader


    def preprocess_and_load_train_test_classifier(self, ctgan_data_set, batch_size, train_size = 0.8): #-> Tuple(torch.dataloader()): #TODO
        data = ctgan_data_set.dataframe()
        conditiond_encoded = self.cond_encoder.get_cond_from_data(data)
        data_encoded = self.data_encoder.transform_without_condition(data) 

        dataset = torch.utils.data.TensorDataset(
            data_encoded, 
            conditiond_encoded
        )
        train_size = int(train_size * len(dataset))  
        test_size = len(dataset) - train_size  

        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader =  DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        return train_loader, test_loader


    def train_classifier(self, dataset : CTGan_data_set, train_size:float = 0.8): 

        train_loader, test_loader = self.preprocess_and_load_train_test_classifier(ctgan_data_set=dataset, batch_size=self.classifier_batch_size, train_size=train_size)
        self.classifier.fit(train_loader=train_loader, test_loader=test_loader, lr=self.classifier_lr, opt_betas=self.classifier_opt_betas, epochs = self.classifier_n_iter, patience=self.classifier_patience) 

        
    def fit(self, 
            ctgan_data_set: CTGan_data_set,
            n_epochs: int = 300,
            n_iter_print: int = 10,
            n_iter_checkpoint: int = None,
            discriminator_num_steps = 5,
            generator_num_steps = 1,
            lambda_gradient_penalty = 10, 
        
        ):
        """

            Args: 
                X: CTGan_data_set that was created given a condition if you want to use one for Data generation
        
        """

        if not (self.generator and self.discriminator and self.classifier):

            self.cond_cols = ctgan_data_set.cond_cols()
            self.cond_encoder = Cond_Encoder(
                ctgan_data_set.dataframe(),
                cond_cols=self.cond_cols,
                categorical_columns=ctgan_data_set.cat_cols(),
                numeric_columns=ctgan_data_set.num_cols(),
                ordinal_columns=ctgan_data_set.ord_cols(),
            )
            self.data_encoder = Data_Encoder(
                ctgan_data_set.dataframe(),
                cond_cols=self.cond_cols,
                categorical_columns=ctgan_data_set.cat_cols(),
                numeric_columns=ctgan_data_set.num_cols(),
                ordinal_columns=ctgan_data_set.ord_cols(),
            )

            self.output_space =  self.data_encoder.encode_dim() 
            self.n_units_conditional =  self.cond_encoder.condition_dim() 

            self.generator = self.generator_class(
                generator_n_units_in= self.n_units_latent,
                generator_n_units_conditional= self.n_units_conditional,
                generator_n_units_out= self.output_space, 
                generator_n_layers_hidden= self.generator_n_layers_hidden,  
                generator_nonlin= self.generator_nonlin, 
                generator_nonlin_out= self.generator_nonlin_out, 
                generator_batch_norm= self.generator_batch_norm, 
                generator_dropout= self.generator_dropout, 
                device= self.device, 
                **self.kwargs
            )

            self.discriminator = self.discriminator_class(
                discriminator_n_units_in= self.output_space,
                discriminator_n_layers_hidden= self.discriminator_n_layers_hidden, 
                discriminator_nonlin= self.discriminator_nonlin, 
                discriminator_batch_norm= self.discriminator_batch_norm, 
                discriminator_dropout= self.discriminator_dropout, 
                device= self.device, 
                **self.kwargs
            )
           
            self.classifier = Classifier(
                classifier_n_units_in = self.output_space - self.n_units_conditional,
                classifier_n_units_out_per_category = self.cond_encoder.get_units_per_column(),
                classifier_n_layers_hidden =  self.classifier_n_layers_hidden, 
                classifier_n_units_hidden = self.classifier_n_units_hidden, 
                classifier_nonlin= self.classifier_nonlin, 
                classifier_batch_norm= self.classifier_batch_norm, 
                classifier_dropout= self.classifier_dropout,
                device= self.device,
                loss= self.classifier_loss,
            )

            self.cond_cols = ctgan_data_set.cond_cols()     # conditional columns 
            self.cat_cols = ctgan_data_set.cat_cols()       # categorical columns
            self.num_cols = ctgan_data_set.num_cols()       # numeric columns 
            self.ord_col = ctgan_data_set.ord_cols()        # ordinal columns 
        
        # for better readability
        generator = self.generator
        discriminator = self.discriminator
        device = self.device

        self.train_classifier(ctgan_data_set)
        
        train_loader = self.preprocess_and_load_data(ctgan_data_set, self.batch_size)

        optim_generator = torch.optim.Adam(generator.net.parameters(), lr=self.generator_lr, betas=self.generator_opt_betas)
        optim_discriminator = torch.optim.Adam(discriminator.net.parameters(), lr=self.discriminator_lr, betas=self.discriminator_opt_betas)

        self.generator.train()
        self.discriminator.train()

        loss_critic_list_train, loss_gen_list_train = list(), list()

        total_time = 0  # total time

        print("Starting Training of the Gan")

        for epoch in range(1, n_epochs + 1):
            start_time = time.time() 

            epoch_loss_critic_train = 0.0
            epoch_loss_gen_train = 0.0
            cond_loss_gen_train = 0.0

            for X, cond in train_loader:
                X_real = X.to(device)
                cond = cond.to(device)
                batch_size = X_real.size(0) # better readability

                # *** Train discriminator ***
                for k in range(discriminator_num_steps):
                    z = torch.randn(batch_size, self.n_units_latent, device=device) 
                    noise_input = torch.cat([z, cond], dim=1)
                    X_fake = generator.forward(noise_input)

                    y_hat_real = discriminator(torch.cat([X_real], dim=1))
                    y_hat_fake = discriminator(torch.cat([X_fake], dim=1)) # TODO vlt sollte der disc nicht nochmal auch die cond bekommen bei beiden beim einen ist sie ja schon dabei beim andern sollte er sich ja ned drum kümmern
                    loss_critic = wasserstein_loss(y_hat_real, y_hat_fake)

                    gp = gradient_penalty(discriminator, torch.cat([X_real], dim=1), torch.cat([X_fake], dim=1), device) # Add gradient penalty 
                    loss_critic += lambda_gradient_penalty * gp
                    optim_discriminator.zero_grad()
                    loss_critic.backward()
                    optim_discriminator.step()

                epoch_loss_critic_train += loss_critic.item()

                # *** Train generator ***
                for k in range(generator_num_steps):
                    z = torch.randn(batch_size, self.n_units_latent, device=device)  # noise vec
                    noise_input = torch.cat([z, cond], dim=1)
                    X_fake = generator.forward(noise_input)

                    cond_detached = cond.detach()
                    y_hat = discriminator(torch.cat([X_fake], dim=1))
                    loss_g = -torch.mean(y_hat)

                    loss_cond = self.compute_condition_loss(X_fake.detach().cpu().numpy(), cond, X_real)   # Calculate condition loss and add to total loss

                    #cond_loss_weight = self.calculate_cond_loss_weight(epoch)
                    
                    total_loss = loss_g + self.lambda_condition_loss_weight * loss_cond    # total_loss = loss_g + self.calculate_cond_loss_weight(epoch) * loss_cond

                    optim_generator.zero_grad()
                    total_loss.backward()
                    optim_generator.step()

                cond_loss_gen_train += loss_cond.item() 
                epoch_loss_gen_train += total_loss.item()

            epoch_loss_critic_train /= len(train_loader)
            epoch_loss_gen_train /= len(train_loader)
            cond_loss_gen_train /= len(train_loader)

            loss_critic_list_train.append(epoch_loss_critic_train)
            loss_gen_list_train.append(epoch_loss_gen_train)
            
            epoch_time = time.time() - start_time
            total_time += epoch_time

            if epoch % n_iter_print == 0:   
                avg_time_per_epoch = total_time / epoch
                remaining_time = avg_time_per_epoch * (n_epochs - epoch)

                print(f"Epoch {epoch:4d} || Loss Critic: {epoch_loss_critic_train:7.4f} || Loss Gen: {epoch_loss_gen_train:7.4f} || Loss Cond: {cond_loss_gen_train:7.4f}|| Avg Time/Epoch: {format_time(avg_time_per_epoch)} || Remaining Time: {format_time_with_h(remaining_time)}")

            if n_iter_checkpoint and epoch % n_iter_checkpoint == 0:
                # TODO: Add code for saving model checkpoints
                pass

        return loss_critic_list_train, loss_gen_list_train


    def evaluate_discriminator(self, data_loader: DataLoader):
        discriminator = self.discriminator
        generator = self.generator
        device = self.device
        
        discriminator.eval()
        generator.eval()

        total_loss_critic = 0.0

        with torch.no_grad():  
            for X, cond in data_loader:
                X_real = X.to(device)
                cond = cond.to(device)
                batch_size = X_real.size(0)

                z = torch.randn(batch_size, self.n_units_latent, device=device) # Noise input for the generator
                noise_input = torch.cat([z, cond], dim=1) # noise with cond 
                X_fake = generator.forward(noise_input)

                y_hat_real = discriminator(torch.cat([X_real, cond], dim=1))
                y_hat_fake = discriminator(torch.cat([X_fake, cond], dim=1))
                loss_critic = wasserstein_loss(y_hat_real, y_hat_fake)

                total_loss_critic += loss_critic.item()

        avg_loss_critic = total_loss_critic / len(data_loader)

        return avg_loss_critic


    def evaluate_generator(self, test_loader: DataLoader):
        generator = self.generator
        discriminator = self.discriminator
        device = self.device
        
        generator.eval()
        discriminator.eval()

        total_loss_gen = 0.0
        with torch.no_grad():  
            for X, cond in test_loader:
                cond = cond.to(device)
                batch_size = X.size(0)

                z = torch.randn(batch_size, self.n_units_latent, device=device) 
                noise_input = torch.cat([z, cond], dim=1)
                X_fake = generator.forward(noise_input)

                y_hat = discriminator(torch.cat([X_fake, cond], dim=1))
                loss_g = -torch.mean(y_hat)

                loss_cond = self.compute_condition_loss(X_fake.detach().cpu().numpy(), cond)    # Condition loss
                total_loss = loss_g + self.lambda_condition_loss_weight * loss_cond

                total_loss_gen += total_loss.item()

        avg_loss_gen = total_loss_gen / len(test_loader)

        return avg_loss_gen

    


    


    def save(self, save_path):
        pass #TODO https://pytorch.org/tutorials/beginner/saving_loading_models.html save multible state dicts append condition coloumns, self.cond_encoder somehow seld.data_ecoder also somehow

    def load(self, path):
        pass #TODO model.load_state_dict(torch.load(PATH, weights_only=True, map_location="cuda:0"))  # it is not so easy to load a model trained on cuda on your cpu 

