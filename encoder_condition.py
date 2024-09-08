import torch
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import pandas as pd
import numpy as np

class Cond_Encoder:
    """
    Conditional encoder that handles the preprocessing of condition columns (categorical, ordinal, numeric)
    for GAN models, preserving the original order of columns and returning the number of output units per column.
    """

    def __init__(
            self, 
            condition_df: pd.DataFrame, 
            categorical_columns: list, 
            ordinal_columns: list, 
            numeric_columns: list
        ):
        
        self._columns_in_order = [col for col in condition_df.columns if col in categorical_columns + ordinal_columns + numeric_columns]
        self._categorical_columns = categorical_columns
        self._ordinal_columns = ordinal_columns
        self._numeric_columns = numeric_columns
        
        self._one_hot_encoders = {}
        self._min_max_scalers = {}
        self._units_per_column = [] 

        self._preprocess_conditions(condition_df)

    def _preprocess_conditions(self, condition_df: pd.DataFrame):
        """
        Prepares encoders and scalers based on the given DataFrame, but does not modify the DataFrame itself.
        Encodes categorical features with One-Hot-Encoding and scales numeric and ordinal features.
        """
        condition_dim = 0

        # Process each column in the order they appear in the DataFrame
        for column in self._columns_in_order:
            if column in self._categorical_columns:
                one_hot_encoder = OneHotEncoder(sparse_output=False)
                condition_df[column] = condition_df[column].astype(str)
                one_hot_encoder.fit(condition_df[[column]])  
                self._one_hot_encoders[column] = one_hot_encoder
                n_categories = len(one_hot_encoder.categories_[0])
                self._units_per_column.append(n_categories)  
                condition_dim += n_categories
            elif column in self._ordinal_columns + self._numeric_columns:
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaler.fit(condition_df[[column]])
                self._min_max_scalers[column] = scaler
                self._units_per_column.append(1)
                condition_dim += 1
        
        self._condition_dim = condition_dim  # Store the total condition dimensionality

    def transform(self, conditions: pd.DataFrame) -> torch.Tensor:
        """
        Encodes the given condition DataFrame into a tensor that can be used as input for the GAN,
        while preserving the original column order.
        """
        condition_encoded = conditions.copy()
        
        encoded_list = []
        for column in self._columns_in_order:
            if column in self._categorical_columns:
                one_hot_encoder = self._one_hot_encoders[column]
                condition_encoded[column] = condition_encoded[column].astype(str)
                encoded_data = one_hot_encoder.transform(condition_encoded[[column]])
                encoded_df = pd.DataFrame(encoded_data, columns=[f"{column}_{i}" for i in range(encoded_data.shape[1])])
                encoded_list.append(encoded_df)
            elif column in self._ordinal_columns + self._numeric_columns:
                scaler = self._min_max_scalers[column]
                scaled_data = scaler.transform(condition_encoded[[column]])
                encoded_df = pd.DataFrame(scaled_data, columns=[column])
                encoded_list.append(encoded_df)
        
        condition_encoded_final = pd.concat(encoded_list, axis=1)
        return torch.tensor(condition_encoded_final.values, dtype=torch.float32)

    def inverse_transform(self, condition_data: torch.Tensor) -> pd.DataFrame:
        """
        Reverts the encoding of condition data back to its original form, preserving the original column order.
        """
        df_sample = pd.DataFrame(condition_data.numpy(), columns=self._columns_in_order)

        for column in self._categorical_columns:
            one_hot_encoder = self._one_hot_encoders[column]
            col_indices = [col for col in df_sample.columns if col.startswith(f"{column}_")]
            one_hot_data = df_sample[col_indices].values
            original_data = one_hot_encoder.inverse_transform(one_hot_data)
            df_sample[column] = original_data
            df_sample = df_sample.drop(col_indices, axis=1)

        for column in self._ordinal_columns + self._numeric_columns:
            scaler = self._min_max_scalers[column]
            df_sample[column] = scaler.inverse_transform(df_sample[[column]])

        return df_sample

    def get_units_per_column(self):
        """
        Returns the number of units (dimensions) per encoded column as a list, maintaining the original column order.
        """
        return self._units_per_column

    def condition_dim(self):
        return self._condition_dim
    
    def __call__(self, data):
        return self.transform(data)