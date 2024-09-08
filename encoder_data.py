import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

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
            target_col: str = None,
            ordinal_cols_max_cluster: int = 50,
            random_state: int = None
    ):
        self.categorical_columns = categorical_columns
        self.ordinal_columns = ordinal_columns
        self.numeric_columns = numeric_columns
        self.target_col = target_col  # Optional target column
        self.random_state = random_state
        self.ordinal_cols_max_cluster = ordinal_cols_max_cluster
        self.one_hot_encoders = {}  # Dictionary for OneHotEncoders
        self.min_max_scalers = {}  # Dictionary for MinMaxScalers (for numeric and ordinal columns)
        self.preprocess_columns(raw_df)

    def preprocess_columns(self, df: pd.DataFrame):
        """
        Prepares encoders and scalers based on the given DataFrame, but does not modify the DataFrame itself.
        """
        encoded_dim = 0 

        for column in self.categorical_columns:
            one_hot_encoder = preprocessing.OneHotEncoder(sparse_output=False)
            df[column] = df[column].astype(str)
            one_hot_encoder.fit(df[[column]]) 
            self.one_hot_encoders[column] = one_hot_encoder
            encoded_dim += len(one_hot_encoder.get_feature_names_out())

        for column in self.ordinal_columns:
            if df[column].nunique() > self.ordinal_cols_max_cluster:
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaler.fit(df[[column]])  
                self.min_max_scalers[column] = scaler
                encoded_dim += 1
            else:
                one_hot_encoder = preprocessing.OneHotEncoder(sparse_output=False)
                one_hot_encoder.fit(df[[column]]) 
                self.one_hot_encoders[column] = one_hot_encoder
                encoded_dim += len(one_hot_encoder.get_feature_names_out())

        for column in self.numeric_columns:
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(df[[column]])
            self.min_max_scalers[column] = scaler
            encoded_dim += 1
        
        self.encoder_n_dim = encoded_dim

    def transform(self, new_df: pd.DataFrame):
        """
        Transforms a new DataFrame using the fitted encoders and scalers.
        """
        df_encoded = new_df.copy()
        
        for column in self.categorical_columns:
            one_hot_encoder = self.one_hot_encoders[column]
            df_encoded[column] = df_encoded[column].astype(str)
            encoded_data = one_hot_encoder.transform(df_encoded[[column]])
            encoded_df = pd.DataFrame(encoded_data, columns=[f"{column}_{i}" for i in range(encoded_data.shape[1])])
            df_encoded = pd.concat([df_encoded.drop(column, axis=1), encoded_df], axis=1)

        for column in self.ordinal_columns:
            if column in self.min_max_scalers:
                scaler = self.min_max_scalers[column]
                df_encoded[column] = scaler.transform(df_encoded[[column]])
            else:
                one_hot_encoder = self.one_hot_encoders[column]
                encoded_data = one_hot_encoder.transform(df_encoded[[column]])
                encoded_df = pd.DataFrame(encoded_data, columns=[f"{column}_{i}" for i in range(encoded_data.shape[1])])
                df_encoded = pd.concat([df_encoded.drop(column, axis=1), encoded_df], axis=1)

        for column in self.numeric_columns:
            scaler = self.min_max_scalers[column]
            df_encoded[column] = scaler.transform(df_encoded[[column]])

        return df_encoded

    def inverse_prep(self, data):
        """
        Inverse the transformations that were made during encoding to reconstruct the original data.
        """
        df_sample = pd.DataFrame(data, columns=self.df.columns)
        
        for column in self.categorical_columns:
            one_hot_encoder = self.one_hot_encoders[column]
            col_indices = [col for col in df_sample.columns if col.startswith(f"{column}_")]
            one_hot_data = df_sample[col_indices].values
            original_data = one_hot_encoder.inverse_transform(one_hot_data)
            df_sample[column] = original_data
            df_sample = df_sample.drop(col_indices, axis=1)
        
        for column in self.numeric_columns:
            scaler = self.min_max_scalers[column]
            df_sample[column] = scaler.inverse_transform(df_sample[[column]])

        for column in self.min_max_scalers.keys():
            if column in df_sample.columns:
                scaler = self.min_max_scalers[column]
                df_sample[column] = scaler.inverse_transform(df_sample[[column]])
        
        return df_sample

    def encodet_dim(self):
        return self.encoder_n_dim 
    
    def __call__(self, data):
        return self.transform(data)