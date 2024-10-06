import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

class Data_Encoder(object):
    """
    Data preprocessing for CTGAN, handles numeric, ordinal, and categorical columns.
    """
    
    def __init__(
            self, 
            raw_df: pd.DataFrame, 
            categorical_columns: list, 
            ordinal_columns: list, 
            numeric_columns: list, 
            cond_cols: list,
            ordinal_cols_max_cluster: int = 50,
            random_state: int = None,
    ):
        self.categorical_columns = categorical_columns
        self.ordinal_columns = ordinal_columns
        self.numeric_columns = numeric_columns

        self.columns_in_order = [col for col in raw_df.columns if col in categorical_columns + ordinal_columns + numeric_columns]
        self.encoders_in_order = []
        self.units_in_order = []
        self.encoder_n_dim = 0
        self.cond_cols = cond_cols

        self.random_state = random_state
        self.ordinal_cols_max_cluster = ordinal_cols_max_cluster #todo ist nicht eingebaut
        
        self.preprocess_columns(raw_df)


    def preprocess_columns(self, df: pd.DataFrame):
          
        for column in self.columns_in_order:

            if column in self.categorical_columns:
                one_hot_encoder = OneHotEncoder(sparse_output=False)
                df[column] = df[column].astype(str)
                one_hot_encoder.fit(df[[column]])
                self.encoders_in_order.append(one_hot_encoder)
                n_categories = len(one_hot_encoder.categories_[0])
                self.units_in_order.append(n_categories)  
            
            elif column in self.ordinal_columns + self.numeric_columns:
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaler.fit(df[[column]])
                self.encoders_in_order.append(scaler)
                self.units_in_order.append(1)  
        
        self.encoder_n_dim = sum(self.units_in_order)


    def transform(self, df: pd.DataFrame):
        if list(df.columns) != list(self.columns_in_order):
            raise ValueError("The columns of the DataFrame are not in the expected order.")
        
        df_encoded = df.copy()
        df_encoded[self.categorical_columns] = df_encoded[self.categorical_columns].astype(str)
        encoded_list = []

        for encoder, column in zip(self.encoders_in_order, self.columns_in_order):
            encoded_data = encoder.transform(df_encoded[[column]])
            encoded_list.append(torch.tensor(encoded_data, dtype=torch.float32))

        data_encoded_final = torch.cat(encoded_list, dim=1)

        return data_encoded_final
    
    def transform_without_condition(self, df: pd.DataFrame):
        """
        Transforms all columns except the condition columns (self.cond_cols).
        """
        if list(df.columns) != list(self.columns_in_order):
            raise ValueError("The columns of the DataFrame are not in the expected order.")

        columns_to_transform = [col for col in self.columns_in_order if col not in self.cond_cols]

        df_encoded = df.copy()        
        categorical_to_transform = [col for col in self.categorical_columns if col in columns_to_transform]
        df_encoded[categorical_to_transform] = df_encoded[categorical_to_transform].astype(str)

        encoded_list = []
        for encoder, column in zip(self.encoders_in_order, self.columns_in_order):
            if column in columns_to_transform:  
                encoded_data = encoder.transform(df_encoded[[column]])
                encoded_list.append(torch.tensor(encoded_data, dtype=torch.float32))

        transformed_data = torch.cat(encoded_list, dim=1)

        return transformed_data


    def inv_transform(self, data):
        df = pd.DataFrame(data)

        if len(df.columns) != self.encoder_n_dim:
            raise ValueError(f"The number of columns in the input data ({len(df.columns)}) does not match the expected number of columns ({self.encoder_n_dim}) based on the encoder's configuration.")

        df_decoded = pd.DataFrame()
        index_units = 0 

        for encoder, units, column  in zip(self.encoders_in_order, self.units_in_order, self.columns_in_order):   
            data_segment = df.iloc[:, index_units:index_units + units]
            decoded_data = encoder.inverse_transform(data_segment)
            df_decoded[column] = decoded_data.flatten()
            index_units += units

        return df_decoded


    def encode_dim(self):
        return self.encoder_n_dim 
    

    def __call__(self, data):
        return self.transform(data) 