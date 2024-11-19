# stdlib
from typing import Any, Callable, List, Optional, Tuple
import time
import os

# third party
import torch
from torch import nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, random_split


# Projects imports
from table_gan.Model.Gans.Base_Gan import Base_CTGan

from table_gan.Model.Generators.Generator import Generator
from table_gan.Model.Critic.critic import Discriminator

from table_gan.Data.dataset import CTGan_data_set
from table_gan.Model.Gans._gan_utils import format_time, format_time_with_h, gradient_penalty, wasserstein_loss
from table_gan.Data.data_encoder import DataEncoder

# Default parameter
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' # If device is not set try to use cuda else cpu

# This project is inspired by CTGAN and is an independent implementation of similar ideas.


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
        generator_n_layers_hidden: int = 2,
        generator_n_units_hidden: int = 500,
        generator_nonlin: str = "relu",
        generator_nonlin_out: str = "sigmoid", # This function given here should return a number between ]0; 1] because the data is processed and scaled between ]0; 1]
        generator_batch_norm: bool = False,
        generator_dropout: float = 0.1,
        generator_lr: float = 0.0001,
        generator_weight_decay: float = 0.0001,
        generator_opt_betas: tuple = (0.5, 0.999),
   
        # discriminator
        discriminator_class: nn.Module = Discriminator,
        discriminator_n_layers_hidden: int = 2,
        discriminator_n_units_hidden: int = 500,
        discriminator_nonlin: str = "leaky_relu",
        discriminator_batch_norm: bool = False,
        discriminator_dropout: float = 0.1,
        discriminator_lr: float =  0.0001,
        discriminator_weight_decay: float = 0.0001,
        discriminator_opt_betas: tuple = (0.5, 0.999),

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

        # Lambda extra Generator losses
        lambda_cond_loss_weight: float = 1,
        lambda_cond_classifier_loss_weight: float = 0,
        lambda_mean_loss_weight: float = 0, # values like 1 or 0 make sence (turn it of or on dont weight it)
        lambda_correlation_loss_weight: float = 0, # values like 1 or 0 make sence (turn it of or on dont weight it)
        lambda_cov_weight : float = 0, 

        # training
        batch_size: int = 200,
        random_state: int = None,
        device: Any = DEVICE,
       
        # data transfomration
        cont_transform_methode : str = "min_max", # how to transfomr continuous numeric coloumns eathter "min_max" or "mode" 
        max_continuous_modes: int = 10, # if cont_transform_methode is "mode" this parameter gives the number of modes that are estimatet 
    
        **kwargs
    ):
        super().__init__()

        self.n_units_latent = n_units_latent

        # generator
        self.generator_class = generator_class
        self.generator_n_layers_hidden= generator_n_layers_hidden
        self.generator_n_units_hidden= generator_n_units_hidden
        self.generator_nonlin= generator_nonlin
        self.generator_nonlin_out= generator_nonlin_out
        self.generator_batch_norm= generator_batch_norm
        self.generator_dropout= generator_dropout
        self.generator_lr = generator_lr
        self.generator_weight_decay = generator_weight_decay
        self.generator_opt_betas = generator_opt_betas


        # discriminator
        self.discriminator_class= discriminator_class
        self.discriminator_n_layers_hidden= discriminator_n_layers_hidden
        self.discriminator_n_units_hidden= discriminator_n_units_hidden
        self.discriminator_nonlin= discriminator_nonlin
        self.discriminator_batch_norm= discriminator_batch_norm
        self.discriminator_dropout= discriminator_dropout
        self.discriminator_lr = discriminator_lr
        self.discriminator_weight_decay = discriminator_weight_decay
        self.discriminator_opt_betas = discriminator_opt_betas


        # classifier
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


        # extra losses weights for generator 
        self.lambda_cond_classifier_loss_weight = lambda_cond_classifier_loss_weight
        self.lambda_cond_loss_weight = lambda_cond_loss_weight  # weighting of the conditional loss that is determit by the classifier
        self.lambda_mean_loss_weight = lambda_mean_loss_weight # weighting of mean loss weight 
        self.lambda_correlation_loss_weight = lambda_correlation_loss_weight
        self.lambda_cov_weight = lambda_cov_weight

        # data transfomration
        self.cont_transform_methode = cont_transform_methode 
        self.max_continuous_modes = max_continuous_modes


        # Some things 
        self.batch_size = batch_size    
        self.random_state = random_state
        self.device = device
        
        self.kwargs = kwargs
        
        self.discriminator, self.generator,  self.data_encoder = None, None, None # set this to None so they can be init in fit 

    

    def preprocess_and_load_data(self, ctgan_data_set, batch_size)-> DataLoader:
        data = ctgan_data_set.dataframe()
        conditiond_encoded = self.data_encoder.transform_cond(data)
        data_encoded = self.data_encoder.transform_data(data)

        dataset = torch.utils.data.TensorDataset(
            data_encoded, 
            conditiond_encoded
        )

        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def preprocess_and_load_data_train_test(self, ctgan_data_set, batch_size, train_size = 0.8):
        """
        
        """
        data = ctgan_data_set.dataframe()
        conditiond_encoded = self.data_encoder.transform_cond(data)
        data_encoded = self.data_encoder.transform_data(data)

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
        if self.lambda_cond_classifier_loss_weight != 0:
            data = ctgan_data_set.dataframe()
            conditiond_encoded = self.data_encoder.transform_cond(data)
            data_encoded = self.data_encoder.transform_data_df_without_condition(data) 

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
        
    def fit(self, 
            ctgan_data_set: CTGan_data_set,
            n_epochs: int = 600,
            n_iter_print: int = 10,
            n_iter_checkpoint: int = None,
            discriminator_num_steps = 5,
            generator_num_steps = 1,
            lambda_gradient_penalty = 10,
            checkpoint_dir = "checkpoints" 
        ):
        """

            Args: 
                X: CTGan_data_set that was created given a condition if you want to use one for Data generation
        
        """

        if not (self.generator and self.discriminator and self.data_encoder):

            self.cond_cols = ctgan_data_set.cond_cols()     # conditional columns 
            self.cat_cols = ctgan_data_set.cat_cols()       # categorical columns
            self.num_cols = ctgan_data_set.num_cols()       # numeric columns 
            self.ord_cols = ctgan_data_set.ord_cols()       # ordinal columns 

            # init data encoder 
            self.data_encoder = DataEncoder(
                ctgan_data_set.dataframe(),
                cond_cols= self.cond_cols,
                categorical_columns= self.cat_cols,
                numeric_columns= self.num_cols,
                ordinal_columns= self.ord_cols,
                cont_transform_method = self.cont_transform_methode,
                gmm_n_components= self.max_continuous_modes
            )

            # Get output space of generator and units conditional for both generator and discriminator
            self.output_space =  self.data_encoder.encode_dim() 
            self.n_units_conditional =  self.data_encoder.encode_cond_dim() 

            # init generator
            self.generator = self.generator_class(
                generator_n_units_in= self.n_units_latent,
                generator_n_units_conditional= self.n_units_conditional,
                generator_n_units_out= self.output_space, 
                generator_n_layers_hidden= self.generator_n_layers_hidden,
                generator_n_units_hidden  = self.generator_n_units_hidden,
                generator_nonlin= self.generator_nonlin, 
                generator_nonlin_out= self.generator_nonlin_out, 
                generator_batch_norm= self.generator_batch_norm, 
                generator_dropout= self.generator_dropout, 
                device= self.device, 
                **self.kwargs
            )

            # init discriminator
            self.discriminator = self.discriminator_class(
                discriminator_n_units_in= self.output_space,
                discriminator_units_conditional= self.n_units_conditional,
                discriminator_n_units_hidden= self.discriminator_n_units_hidden,
                discriminator_n_layers_hidden= self.discriminator_n_layers_hidden, 
                discriminator_nonlin= self.discriminator_nonlin, 
                discriminator_batch_norm= self.discriminator_batch_norm, 
                discriminator_dropout= self.discriminator_dropout, 
                device= self.device, 
                **self.kwargs
            )

            # init generator extra penalties when the weights are set to != 0 and if something needs to be init
            self.init_generator_extra_penalties(ctgan_data_set)

           
        # for better readability
        generator = self.generator
        discriminator = self.discriminator
        device = self.device

        train_loader = self.preprocess_and_load_data(ctgan_data_set, self.batch_size)

        optim_generator = torch.optim.Adam(generator.net.parameters(), lr=self.generator_lr, betas=self.generator_opt_betas)
        optim_discriminator = torch.optim.Adam(discriminator.net.parameters(), lr=self.discriminator_lr, betas=self.discriminator_opt_betas)

        self.generator.train()
        self.discriminator.train()

        loss_critic_list_train, loss_gen_list_train = list(), list()

        total_time = 0

        print("Starting Training of the Gan")

        for epoch in range(1, n_epochs + 1):
            start_time = time.time() 

            epoch_loss_critic_train, epoch_loss_gen_train, epoch_extra_gen_loss = 0, 0, 0   # set loss to 0 at the start of every epoch 

            for X, cond in train_loader:
                X_real = X.to(device)
                cond = cond.to(device)
                batch_size = X_real.size(0) # better readability

                # *** Train discriminator ***
                for k in range(discriminator_num_steps):
                    z = torch.randn(batch_size, self.n_units_latent, device=device) 
                    noise_input = torch.cat([z, cond], dim=1)
                    X_fake = generator.forward(noise_input)

                    y_hat_real = discriminator(torch.cat([X_real, cond], dim=1))
                    y_hat_fake = discriminator(torch.cat([X_fake, cond], dim=1))  
                    loss_critic = - wasserstein_loss(y_hat_real, y_hat_fake)

                    gp = gradient_penalty(discriminator, torch.cat([X_real, cond], dim=1), torch.cat([X_fake, cond], dim=1), device)  
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
                    y_hat = discriminator(torch.cat([X_fake, cond_detached], dim=1))
                    loss_g = - torch.mean(y_hat)

                    extra_gen_loss = self.compute_extra_loss_generator(X_gen=X_fake, X_real=X_real, cond=cond)

                    total_loss = loss_g + extra_gen_loss

                    optim_generator.zero_grad()
                    total_loss.backward()
                    optim_generator.step()

                epoch_extra_gen_loss += extra_gen_loss.item() 
                epoch_loss_gen_train += total_loss.item()

            epoch_loss_critic_train /= len(train_loader)
            epoch_loss_gen_train /= len(train_loader)
            epoch_extra_gen_loss /= len(train_loader)

            loss_critic_list_train.append(epoch_loss_critic_train)
            loss_gen_list_train.append(epoch_loss_gen_train)
            
            epoch_time = time.time() - start_time
            total_time += epoch_time

            if epoch % n_iter_print == 0:   
                avg_time_per_epoch = total_time / epoch
                remaining_time = avg_time_per_epoch * (n_epochs - epoch)
                print(f"Epoch {epoch:4d} || Loss Critic: {epoch_loss_critic_train:7.4f} || Loss Gen: {epoch_loss_gen_train:7.4f} || Loss Gen Extra: {epoch_extra_gen_loss:7.4f}|| Avg Time/Epoch: {format_time(avg_time_per_epoch)} || Remaining Time: {format_time_with_h(remaining_time)}")

            if n_iter_checkpoint and epoch % n_iter_checkpoint == 0:
                self.save_checkpoint(epoch, checkpoint_dir=checkpoint_dir)

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

                z = torch.randn(batch_size, self.n_units_latent, device=device)  # Noise input for the generator
                noise_input = torch.cat([z, cond], dim=1)  # noise with condition
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
                X = X.to(device)
                cond = cond.to(device)
                batch_size = X.size(0)

                z = torch.randn(batch_size, self.n_units_latent, device=device) #Noise vec
                noise_input = torch.cat([z, cond], dim=1)
                X_fake = generator.forward(noise_input)

                y_hat = discriminator(torch.cat([X_fake, cond], dim=1))
                loss_g = -torch.mean(y_hat)

                # Calculate extra generator losses
                extra_gen_loss = self.compute_extra_loss_generator(X_fake, X, cond)

                total_loss = loss_g + extra_gen_loss
                total_loss_gen += total_loss.item()

        avg_loss_gen = total_loss_gen / len(test_loader)
        return avg_loss_gen
    
    def save_checkpoint(self, epoch, checkpoint_dir="checkpoints"):
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_filename = f"{epoch}_epoch_checkpoint.pt"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

        self.save(save_path= checkpoint_path)
        

    def save(self, save_path="wctgan_model.pt"):
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'data_encoder': self.data_encoder, 
            'config': {
                'n_units_latent': self.n_units_latent,
                'generator_params': {
                    'generator_n_units_in': self.n_units_latent,
                    'generator_n_units_conditional': self.n_units_conditional,
                    'generator_n_units_out': self.output_space,
                    'generator_n_layers_hidden': self.generator_n_layers_hidden,
                    'generator_n_units_hidden': self.generator_n_units_hidden,
                    'generator_nonlin': self.generator_nonlin,
                    'generator_nonlin_out': self.generator_nonlin_out,
                    'generator_batch_norm': self.generator_batch_norm,
                    'generator_dropout': self.generator_dropout,
                    'device': self.device
                },
                'discriminator_params': {
                    'discriminator_n_units_in': self.output_space,
                    'discriminator_units_conditional': self.n_units_conditional,
                    'discriminator_n_units_hidden': self.discriminator_n_units_hidden,
                    'discriminator_n_layers_hidden': self.discriminator_n_layers_hidden,
                    'discriminator_nonlin': self.discriminator_nonlin,
                    'discriminator_batch_norm': self.discriminator_batch_norm,
                    'discriminator_dropout': self.discriminator_dropout,
                    'device': self.device
                }
            }
        }, save_path)


    def load(self, load_path="wctgan_model.pt"):
        checkpoint = torch.load(load_path, map_location=self.device)

        config = checkpoint['config']
        self.n_units_latent = config['n_units_latent']

        self.generator = Generator(**config['generator_params']).to(self.device)
        self.discriminator = Discriminator(**config['discriminator_params']).to(self.device)

        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.data_encoder = checkpoint['data_encoder']
        print(f"Model loaded from {load_path}")


