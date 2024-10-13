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
from ...Data.dataset import CTGan_data_set


# Default parameter
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' # If device is not set try to use cuda else cpu



class Base_CTGan(nn.Module):
    def __init__(self):
        super().__init__()

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


    def compute_condition_loss(self, gen_data:torch.tensor, real_cond:torch.tensor, test):
        x_gen = self.data_encoder.inv_transform(gen_data)
        x_gen_without_cond = self.data_encoder.transform_without_condition(x_gen)

        pred = self.classifier.forward(x_gen_without_cond)
        pred_tensor = torch.cat(pred, dim=1)

        loss_condition = torch.nn.functional.binary_cross_entropy_with_logits(pred_tensor.float(), real_cond.float()) 

        return loss_condition


    def calculate_cond_loss_weight(self, epoch):
        weight = (epoch / (self.n_epochs)) * self.lambda_condition_loss_weight

        return weight



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
            cond_df = self.cond_encoder.generate_random_condition(num_samples)
            random_cond = self.cond_encoder.transform(cond_df)
            gen_data_raw = self.generator.generate(num_samples, random_cond)
            gen_data = self.data_encoder.inv_transform(gen_data_raw)
            return gen_data
        
        else:
            if len(self.cond_cols) != len(cond_df.columns): # todo set vergleichen da wir das ja mit dicts machen 
                raise ValueError(f"cond_df is missing a coloumn, following columns are required: {self.cond_cols}")

            cond_tensor = self.cond_encoder.transform(cond_df)

            if cond_tensor.shape[1] != self.n_units_conditional:
                raise ValueError(f"Expected {self.n_units_conditional} conditional columns, but got {cond_tensor.shape[1]}")
            
            gen_data_raw = self.generator.generate(n_samples=cond_tensor.shape[0], condition=cond_tensor)
            gen_data_np = gen_data_raw.detach().cpu().numpy() 
            gen_data_df = pd.DataFrame(gen_data_np) 

            gen_data = self.data_encoder.inv_transform(gen_data_df)
            
            return gen_data


    def save(self, save_path):
        pass #TODO https://pytorch.org/tutorials/beginner/saving_loading_models.html save multible state dicts append condition coloumns, self.cond_encoder somehow seld.data_ecoder also somehow

    def load(self, path):
        pass #TODO model.load_state_dict(torch.load(PATH, weights_only=True, map_location="cuda:0"))  # it is not so easy to load a model trained on cuda on your cpu 

    
    def fit(self, ctgan_data_set: CTGan_data_set):
        pass


    def evaluate_discriminator(self, data_loader: DataLoader):
        pass


    def evaluate_generator(self, test_loader: DataLoader):
        pass

