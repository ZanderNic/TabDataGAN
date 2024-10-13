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
            df: pd.DataFrame, 
            categorical_columns: list, 
            ordinal_columns: list, 
            numeric_columns: list,
            cond_cols : list
        ):
        
        self.cond_cols = cond_cols

        self.categorical_columns = [col for col in categorical_columns if col in cond_cols]
        self.ordinal_columns = [col for col in ordinal_columns if col in cond_cols]
        self.numeric_columns = [col for col in numeric_columns if col in cond_cols]

        self.columns_in_order = [col for col in df.columns if col in self.categorical_columns + self.ordinal_columns + self.numeric_columns]
        self.encoders_in_order = []
        self.units_in_order = []
        self.encoder_n_dim = 0
        
        self.preprocess_conditions(df)


    def preprocess_conditions(self, condition_df: pd.DataFrame):
        """
        Prepares encoders and scalers based on the given DataFrame, but does not modify the DataFrame itself.
        Encodes categorical features with One-Hot-Encoding and scales numeric and ordinal features.
        """
        for column in self.columns_in_order:
            if column in self.categorical_columns:
                one_hot_encoder = OneHotEncoder(sparse_output=False)
                condition_df[column] = condition_df[column].astype(str)
                one_hot_encoder.fit(condition_df[[column]])  
                self.encoders_in_order.append(one_hot_encoder)
                n_categories = len(one_hot_encoder.categories_[0])
                self.units_in_order.append(n_categories)  
            
            elif column in self.ordinal_columns + self.numeric_columns:
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaler.fit(condition_df[[column]])
                self.encoders_in_order.append(scaler)
                self.units_in_order.append(1)
        
        self.encoder_n_dim = sum(self.units_in_order)


    def get_cond_from_data(self, df:pd.DataFrame):
        """
            This funktion retourns the cond loaded if the howle data frame is provided
        """
        cond_df = df[self.cond_cols]
        cond_df_transformed = self.transform(cond_df)
        return cond_df_transformed


    def transform(self, cond_df: pd.DataFrame):
        df_encoded = cond_df.copy()
        df_encoded[self.categorical_columns] = df_encoded[self.categorical_columns].astype(str)
        encoded_list = []

        for encoder, column in zip(self.encoders_in_order, self.columns_in_order):
            if column in df_encoded.columns:
                encoded_data = encoder.transform(df_encoded[[column]])
                encoded_list.append(torch.tensor(encoded_data, dtype=torch.float32))
            else:
                raise ValueError(f"Column {column} is not found in the input DataFrame.")

        condition_encoded_final = torch.cat(encoded_list, dim=1)
        
        return condition_encoded_final
        

    def inv_transform(self, data):
        df = pd.DataFrame(data)
        
        if len(df.columns) != self.encoder_n_dim:
            raise ValueError(f"The number of columns in the input data ({len(df.columns)}) does not match the expected number of columns ({self.encoder_n_dim}) based on the encoder's configuration.")

        df_decoded = pd.DataFrame()
        index_units = 0 

        for encoder, units, column in zip(self.encoders_in_order, self.units_in_order, self.columns_in_order):   
            data_segment = df.iloc[:, index_units:index_units + units]
            decoded_data = encoder.inverse_transform(data_segment)
            df_decoded[column] = decoded_data.flatten()
            index_units += units

        return df_decoded
    

    def generate_random_condition(self, num_samples: int = 1) -> pd.DataFrame:
        random_conditions = {}
        for column in self.columns_in_order:
            if column in self.categorical_columns:
                categories = self.encoders_in_order[self.columns_in_order.index(column)].categories_[0]
                random_conditions[column] = np.random.choice(categories, num_samples)
            elif column in self.ordinal_columns + self.numeric_columns:
                random_conditions[column] = np.random.uniform(0, 1, num_samples)

        random_conditions_df = pd.DataFrame(random_conditions)
       
        return random_conditions_df


    def get_units_per_column(self):
        """
        Returns the number of units (dimensions) per encoded column as a list, maintaining the original column order.
        """
        return self.units_in_order


    def condition_dim(self):
        return self.encoder_n_dim

    def cond_cols(self):
        return self.columns_in_order
