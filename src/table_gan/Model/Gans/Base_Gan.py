# stdlib
from typing import Any, Callable, List, Optional, Tuple

# third party
import torch
from torch import nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, random_split

# Projects imports
from table_gan.Data.dataset import CTGan_data_set
from table_gan.Model.Extra_Pen.Classifier import Classifier

# Default parameter
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' # If device is not set try to use cuda else cpu

# This project is inspired by CTGAN and is an independent implementation of similar ideas.

class Base_CTGan(nn.Module):
    def __init__(self):
        super().__init__()


    def preprocess_and_load_data(self, ctgan_data_set, batch_size)-> DataLoader:
        data = ctgan_data_set.dataframe()
        conditiond_encoded = self.data_encoder.get_cond_from_data(data)
        data_encoded = self.data_encoder.transform_data_df_without_condition(data)

        dataset = torch.utils.data.TensorDataset(
            data_encoded, 
            conditiond_encoded
        )

        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def preprocess_and_load_data_train_test(self, ctgan_data_set, batch_size, train_size = 0.8):
        data = ctgan_data_set.dataframe()
        conditiond_encoded = self.data_encoder.get_cond_from_data(data)
        data_encoded = self.data_encoder.transform_data_df_without_condition(data)

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


    def init_generator_extra_penalties(self, ctgan_data_set: CTGan_data_set):
        """
            takes the weights of the lambdas to decide if something should be init, not every extra gen pen needs to be init
        """
        
        if self.lambda_cond_classifier_loss_weight:
            # init classifier
            self.classifier = Classifier(
                classifier_n_units_in = self.output_space - self.n_units_conditional,
                classifier_n_units_out_per_category = self.data_encoder.get_units_per_cond_column(),
                classifier_n_layers_hidden =  self.classifier_n_layers_hidden, 
                classifier_n_units_hidden = self.classifier_n_units_hidden, 
                classifier_nonlin= self.classifier_nonlin, 
                classifier_batch_norm= self.classifier_batch_norm, 
                classifier_dropout= self.classifier_dropout,
                device= self.device,
                loss= self.classifier_loss,
            )
            
            self.train_classifier(ctgan_data_set) # train Classifier

        if self.lambda_cond_loss_weight:
            pass # nothing must be init for this loss

        if self.lambda_mean_loss_weight:
            data_df = ctgan_data_set.df()
            real_data_tensor = self.data_encoder.transform_data(data_df)

            self.real_data_mean = torch.mean(real_data_tensor, dim=0)

        if self.lambda_correlation_loss_weight:
            pass # nothing must be init for this loss

        if self.lambda_cov_weight:
            data_df = ctgan_data_set.df()
            real_data_tensor = self.data_encoder.transform_data(data_df)
            cov_real = torch.cov(torch.swapdims(real_data_tensor,0,1)) 
            self.lower_cov_real = torch.tril(cov_real) # get the lower triangolar elements of the cov matrix
        

    def preprocess_and_load_train_test_classifier(self, ctgan_data_set, batch_size, train_size = 0.8): 
        data = ctgan_data_set.dataframe()
        conditiond_encoded = self.data_encoder.get_cond_from_data(data)
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
    

    def train_classifier(self, ctgan_data_set : CTGan_data_set, train_size:float = 0.8): 
        train_loader, test_loader = self.preprocess_and_load_train_test_classifier(ctgan_data_set=ctgan_data_set, batch_size=self.classifier_batch_size, train_size=train_size)
        self.classifier.fit(train_loader=train_loader, test_loader=test_loader, lr=self.classifier_lr, opt_betas=self.classifier_opt_betas, epochs = self.classifier_n_iter, patience=self.classifier_patience) 


    def compute_extra_loss_generator(self, X_gen: torch.tensor, X_real:torch.tensor, cond:torch.tensor):
        total_extra_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        if self.lambda_cond_loss_weight:
            cond_loss = self.compute_condition_loss(X_gen, cond)
            total_extra_loss = total_extra_loss + self.lambda_cond_loss_weight * cond_loss
        
        if self.lambda_cond_classifier_loss_weight:
            cond_class_loss = self.compute_condition_classifier_loss(X_gen, cond)
            total_extra_loss = total_extra_loss + self.lambda_cond_classifier_loss_weight * cond_class_loss
        
        if self.lambda_mean_loss_weight:
            mean_loss = self.compute_mean_loss(X_gen)
            total_extra_loss = total_extra_loss + self.lambda_mean_loss_weight * mean_loss

        if self.lambda_correlation_loss_weight:
            corr_loss = self.compute_correlation_loss(X_gen, X_real)
            total_extra_loss = total_extra_loss + self.lambda_correlation_loss_weight * corr_loss

        if self.lambda_cov_weight:
            cov_loss = self.compute_cov_loss(X_gen)
            total_extra_loss = total_extra_loss + self.lambda_cov_weight * cov_loss

        return total_extra_loss

    # TODO this two implementations dosent work with CTGan encoding or GMM 
    def compute_condition_loss(self, gen_data:torch.tensor, real_cond:torch.tensor):
        cond_x_gen = self.data_encoder.get_condition_from_tensor(gen_data) 

        cond_x_gen = cond_x_gen.to(self.device)
        cond_real = real_cond.to(self.device)

        units_per_col = self.data_encoder.get_units_per_cond_column()    
        cum_units_per_col = [0] + np.cumsum(units_per_col).tolist()

        loss_condition = torch.tensor(0.0, device=self.device)

        for i in range(len(units_per_col)):
            cond_real_tensor = cond_real[:, cum_units_per_col[i]:cum_units_per_col[i+1]]
            cond_gen_tensor = cond_x_gen[:, cum_units_per_col[i]:cum_units_per_col[i+1]]
                
            if units_per_col[i]>= 0:
                loss_condition = loss_condition + torch.nn.functional.cross_entropy(cond_gen_tensor, torch.argmax(cond_real_tensor, dim = 1)) 
            else:
                loss_condition = loss_condition + torch.mean((cond_real_tensor - cond_gen_tensor)**2) # use mse if it is a num col

        return loss_condition

    # TODO this two implementations dosent work with CTGan encoding or GMM
    def compute_condition_classifier_loss(self, gen_data:torch.tensor, real_cond:torch.tensor):
        x_gen_data = self.data_encoder.get_only_data_from_tensor(gen_data) 

        cond_real = real_cond.to(self.device)
        pred_cond_x_gen = self.classifier.predict(x_gen_data)

        units_per_col = self.data_encoder.get_units_per_cond_column()    
        cum_units_per_col = [0] + np.cumsum(units_per_col).tolist()

        loss_condition_class = torch.tensor(0.0, device=self.device)

        for i in range(len(units_per_col)):
            cond_real_tensor = cond_real[:, cum_units_per_col[i]:cum_units_per_col[i+1]]
            cond_gen_tensor = pred_cond_x_gen[:, cum_units_per_col[i]:cum_units_per_col[i+1]]
                
            if units_per_col[i]>= 0:
                loss_condition_class = loss_condition_class + torch.nn.functional.cross_entropy(cond_gen_tensor, torch.argmax(cond_real_tensor, dim = 1)) 
            else:
                loss_condition_class = loss_condition_class + torch.mean((cond_real_tensor - cond_gen_tensor)**2) # use mse if it is a num col

        return loss_condition_class


    def compute_correlation_loss(self, gen_data: torch.tensor, real_data:torch.tensor) -> torch.tensor:
        # Source: https://arxiv.org/html/2405.16971v1
        mean_gen_data = torch.mean(gen_data, dim=0)
        mean_real_data = torch.mean(real_data, dim=0)

        dev_gen = gen_data - mean_gen_data
        dev_real = real_data - mean_real_data 

        numerator = torch.sum(dev_gen * dev_real, dim=0)
        
        std_gen = torch.sqrt(torch.sum(dev_gen ** 2, dim=0))
        std_real = torch.sqrt(torch.sum(dev_real ** 2, dim=0))
        
        denominator = std_gen * std_real
        
        correlation = numerator / (denominator + 1e-8)  
        loss_correlation = 1 - correlation.mean() 
        
        return loss_correlation


    def compute_mean_loss(self, gen_data: torch.tensor) -> torch.tensor:
        # Source: https://arxiv.org/html/2405.16971v1
        mean_gen_data = torch.mean(gen_data, dim=0)
        mean_real_data = self.real_data_mean.to(self.device)
        
        loss_mean = torch.mean(torch.abs(mean_gen_data - mean_real_data))
        
        return loss_mean
    

    def compute_cov_loss(self, gen_data:torch.tensor) -> torch.tensor:
        """
        
        """
        # TODO Problem when batch size < num_features
        conv_gen = torch.cov(torch.swapdims(gen_data, 0, 1)) # swap dims 
        lower_cov_gen = torch.tril(conv_gen, diagonal=-1) # get the lower triangolar elements of the cov matrix
        lower_cov_real = self.lower_cov_real.to(self.device)

        cov_loss = torch.sum((lower_cov_real - lower_cov_gen)**2)

        return cov_loss # todo check why this is so low 

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
            cond_df = self.data_encoder.generate_random_condition(num_samples)
            random_cond = self.data_encoder.transform_cond(cond_df)
            gen_data_raw = self.generator.generate(num_samples, random_cond)
            gen_data = self.data_encoder.inv_transform_data(gen_data_raw)
            return gen_data
        
        else:
            if len(self.cond_cols) != len(cond_df.columns): # todo set vergleichen da wir das ja mit dicts machen 
                raise ValueError(f"cond_df is missing a coloumn, following columns are required: {self.cond_cols}")

            cond_tensor = self.data_encoder.transform_cond(cond_df)

            if cond_tensor.shape[1] != self.n_units_conditional:
                raise ValueError(f"Expected {self.n_units_conditional} conditional columns, but got {cond_tensor.shape[1]}")

            gen_data_raw = self.generator.generate(n_samples=cond_tensor.shape[0], condition=cond_tensor)
            gen_data_np = gen_data_raw.detach().cpu().numpy() 

            gen_data = self.data_encoder.inv_transform_data(gen_data_np)
            
            return gen_data


    def save(self, save_path):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError
    
    def fit(self, ctgan_data_set: CTGan_data_set):
        raise NotImplementedError


    def evaluate_discriminator(self, data_loader: DataLoader):
        raise NotImplementedError


    def evaluate_generator(self, test_loader: DataLoader):
        raise NotImplementedError

