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
from ..Data.dataset import CTGan_data_set
from .Classifier import Classifier
from .CTGan_utils import get_nolin_akt, get_loss_function, format_time, format_time_with_h, gradient_penalty, wasserstein_loss
from ..Data.encoder_condition import Cond_Encoder
from ..Data.encoder_data import Data_Encoder


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
        
        self.n_units_lat = generator_n_units_in

        self.device = device
        self.net = nn.Sequential()

        generator_nonlin = get_nolin_akt(generator_nonlin)
        generator_nonlin_out = get_nolin_akt(generator_nonlin_out)
        
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

    def generate(self, n_samples, condition): 
        self.eval()  
        with torch.no_grad():
            noise = torch.randn(n_samples, self.n_units_lat, device=self.device) 
            generator_input = torch.cat([noise, condition], dim=1) 
            gen_data = self.forward(generator_input) 
        return gen_data



class Discriminator(nn.Module):
    
    def __init__(
        self, 
        discriminator_n_units_in: int,
        discriminator_n_units_conditional: int,
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


    def forward(self, x):
        return self.net(x).view(-1)



class CTGan(nn.Module):
    """

        Classifier:  
            the classifier is used to predict the condition, it is trained on the real data and later used for classifiing the conditon for the generated data
            to get a nother loss
    
    
    
    
    """

    def __init__(
        self,

        n_units_latent: int = 100, 
        
        # Generator
        generator_n_layers_hidden: int = 2,
        generator_n_units_hidden: int = 250,
        generator_nonlin: str = "relu",
        generator_nonlin_out = "sigmoid", # This function given here should return a number between ]0; 1] because the data is processed and scaled between ]0; 1]
        generator_num_steps: int = 5,
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
        lambda_gradient_penalty: float = 1,
        lambda_condition_loss_weight: float = 1,
        
        n_epochs: int = 300,
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
        self.lambda_gradient_penalty = lambda_gradient_penalty # to tourn of gradient penalty set this to 0 
        self.lambda_condition_loss_weight = lambda_condition_loss_weight  # weighting of the conditional loss that is determit by the classifier

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
        conditiond_encoded = self.cond_encoder.get_cond_from_data(data)
        data_encoded = self.data_encoder(data)

        dataset = torch.utils.data.TensorDataset(
            data_encoded, 
            conditiond_encoded
        )

        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def preprocess_and_load_train_test_classifier(self, ctgan_data_set, batch_size, train_size = 0.8): #-> Tuple(torch.dataloader()): #TODO
        data = ctgan_data_set.dataframe()
        conditiond_encoded = self.cond_encoder.get_cond_from_data(data)
        data_encoded = self.data_encoder.transform_without_condition(data) 

        dataset = torch.utils.data.TensorDataset(
            data_encoded, 
            conditiond_encoded
        )
        train_dataset, test_dataset = random_split(dataset, [train_size, 1-train_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader =  DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        return train_loader, test_loader


    def train_classifier(self, dataset : CTGan_data_set, train_size:float = 0.8): 

        train_loader, test_loader = self.preprocess_and_load_train_test_classifier(ctgan_data_set=dataset, batch_size=self.classifier_batch_size, train_size=train_size)
        self.classifier.fit(train_loader=train_loader, test_loader=test_loader, lr=self.classifier_lr, opt_betas=self.classifier_opt_betas, epochs = self.classifier_n_iter, patience=self.classifier_patience) # loss=self.classifier_loss

        
    def fit(self, ctgan_data_set: CTGan_data_set):
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
            self.ord_col = ctgan_data_set.ord_cols()        # ordinal columns that should be given to the data_set

        
        # for better readability
        generator = self.generator
        discriminator = self.discriminator
        device = self.device

        self.train_classifier(ctgan_data_set)
        
        data_loader = self.preprocess_and_load_data(ctgan_data_set, self.batch_size)

        optim_generator = torch.optim.Adam(generator.net.parameters(), lr=self.generator_lr, betas=self.generator_opt_betas)
        optim_discriminator = torch.optim.Adam(discriminator.net.parameters(), lr=self.discriminator_lr, betas=self.discriminator_opt_betas)

        self.generator.train()
        self.discriminator.train()

        loss_critic_list = list()
        loss_gen_list = list()

        total_time = 0  # to track total elapsed time

        for epoch in range(1, self.n_epochs + 1):
            start_time = time.time()  # Start time of current epoch

            epoch_loss_critic = 0.0
            epoch_loss_gen = 0.0
            for X, cond in data_loader:
                X_real = X.to(device)
                cond = cond.to(device)
                batch_size = X_real.size(0)

                # *** Train discriminator ***
                for k in range(self.discriminator_num_steps):
                    z = torch.randn(batch_size, self.n_units_latent, device=device) 
                    noise_input = torch.cat([z, cond], dim=1)
                    X_fake = generator.forward(noise_input)

                    y_hat_real = discriminator(torch.cat([X_real, cond], dim=1))
                    y_hat_fake = discriminator(torch.cat([X_fake, cond], dim=1))
                    loss_critic = wasserstein_loss(y_hat_real, y_hat_fake)

                    # Add gradient penalty if necessary
                    gp = gradient_penalty(discriminator, torch.cat([X_real, cond], dim=1), torch.cat([X_fake, cond], dim=1), device)
                    loss_critic += self.lambda_gradient_penalty * gp
                    optim_discriminator.zero_grad()
                    loss_critic.backward()
                    optim_discriminator.step()

                epoch_loss_critic += loss_critic.item()

                # *** Train generator ***
                for k in range(self.generator_num_steps):
                    z = torch.randn(batch_size, self.n_units_latent, device=device)  # noise vec
                    noise_input = torch.cat([z, cond], dim=1)
                    X_fake = generator.forward(noise_input)

                    cond_detached = cond.detach()
                    y_hat = discriminator(torch.cat([X_fake, cond_detached], dim=1))
                    loss_g = -torch.mean(y_hat)

                    # Calculate condition loss and update total loss
                    loss_cond = self.compute_condition_loss(X_fake.detach().cpu().numpy(), cond)
                    total_loss = loss_g + self.lambda_condition_loss_weight * loss_cond

                    optim_generator.zero_grad()
                    total_loss.backward()
                    optim_generator.step()

                epoch_loss_gen += total_loss.item()

            epoch_loss_critic /= len(data_loader)
            epoch_loss_gen /= len(data_loader)

            loss_critic_list.append(epoch_loss_critic)
            loss_gen_list.append(epoch_loss_gen)

            epoch_time = time.time() - start_time
            total_time += epoch_time

            avg_time_per_epoch = total_time / epoch
            remaining_time = avg_time_per_epoch * (self.n_epochs - epoch)

            if epoch % self.n_iter_print == 0:
                print(f"Epoch {epoch:4d} || Loss Critic: {epoch_loss_critic:7.4f} || Loss Gen: {epoch_loss_gen:7.4f} || Avg Time/Epoch: {format_time(avg_time_per_epoch)} || Remaining Time: {format_time_with_h(remaining_time)}")

            if self.n_iter_checkpoint and epoch % self.n_iter_checkpoint == 0:
                # TODO: Add code for saving model checkpoints
                pass

        return loss_critic_list, loss_gen_list

    def compute_condition_loss(self, X, real_cond):
        """
        Predicts the condition using the classifier, calculates the difference between the real and predicted condition,
        and appends this difference to the base loss.

        Args:
        - X (Tensor): The gen data used for prediction.
        - real_condition (Tensor): The true condition.

        Returns:
        - loss_condition (Tensor): The classification loss (condition prediction loss).
        """
        x_transformed = self.data_encoder.inv_transform(X)

        x_classifier = x_transformed.drop(columns=self.cond_cols)
        x_classifier_tensor = torch.tensor(x_classifier.values, dtype=torch.float32)

        pred = self.classifier.predict(x_classifier_tensor)

        loss_condition = torch.nn.functional.binary_cross_entropy_with_logits(pred.float(), real_cond.float())
    
        return loss_condition

    
    def gen(self, num_samples=None, cond_df=pd.DataFrame()):
        """
        Generate samples using the generator model. Either `num_samples` or `cond_df` should be provided.

        Parameters:
        - num_samples: The number of samples to generate if not using a condition.
        - cond_df: A DataFrame containing the conditional columns for generation.

        Returns:
        - Generated data as a DataFrame.
        """
        
        # Check if both num_samples and cond_df are missing
        if num_samples is None and cond_df.empty:
            raise ValueError("Please provide either num_samples or cond_df to generate data.")
        
        # If cond_df is empty, generate random conditions
        if cond_df.empty:
            random_cond = self.cond_encoder.generate_random_condition(num_samples)
            gen_data_raw = self.generator.generate(num_samples, random_cond)
            gen_data = self.data_encoder.inv_transform(gen_data_raw)
            return gen_data
        
        else:
            # Check if cond_df contains all required columns (ignoring the order)
            if len(self.cond_cols) != len(cond_df.columns):
                raise ValueError(f"cond_df is missing a coloumn, following columns are required: {self.cond_cols}")

            cond_df = self.cond_encoder.transform(cond_df)

            if cond_df.shape[1] != self.n_units_conditional:
                raise ValueError(f"Expected {self.n_units_conditional} conditional columns, but got {cond_df.shape[1]}")
            
            gen_data_raw = self.generator.generate(n_samples=cond_df.shape[0], condition=cond_df)
            gen_data_np = gen_data_raw.detach().cpu().numpy() 
            gen_data_df = pd.DataFrame(gen_data_np) 

            gen_data = self.data_encoder.inv_transform(gen_data_df)
            
            return gen_data

    


    def save(self, save_path):
        pass #TODO https://pytorch.org/tutorials/beginner/saving_loading_models.html save multible state dicts append condition coloumns, self.cond_encoder somehow seld.data_ecoder also somehow

    def load(self, path):
        pass #TODO model.load_state_dict(torch.load(PATH, weights_only=True, map_location="cuda:0"))  # it is not so easy to load a model trained on cuda on your cpu 
