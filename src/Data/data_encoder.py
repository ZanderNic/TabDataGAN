import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

class DataEncoder(object):
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
            random_state: int = None,
    ):
        self.categorical_columns = categorical_columns
        self.ordinal_columns = ordinal_columns
        self.numeric_columns = numeric_columns

        self.categorical_cond_columns = [col for col in self.categorical_columns if col in cond_cols]

        self.columns_in_order = [col for col in raw_df.columns if col in categorical_columns + ordinal_columns + numeric_columns]
        self.cond_cols_in_order = [col for col in raw_df.columns if col in categorical_columns + ordinal_columns + numeric_columns and col in cond_cols]
        self.d_types_in_order = raw_df[self.columns_in_order].dtypes.tolist()

        self.encoders_in_order = []
        self.index_cond = [] 
        self.units_in_order = []
        self.encoder_n_dim = 0
        self.cond_cols = cond_cols

        self.random_state = random_state #todo use in generate random cond

        self.preprocess_columns(raw_df)


    def preprocess_columns(self, df: pd.DataFrame):
        df = df.copy()

        for index, column in enumerate(self.columns_in_order):
            if column in self.cond_cols:
                self.index_cond.append(index) 

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
        self.encode_n_cond_dim = sum([self.units_in_order[i] for i in self.index_cond])

    def transform_data(self, df: pd.DataFrame):
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
    

    def transform_cond(self, cond_df: pd.DataFrame):
        """
        Transforms cond columns from either only cond df or whole df.
        """
        cond_df_encoded = cond_df.copy()
        cond_df_encoded[self.categorical_cond_columns] = cond_df_encoded[self.categorical_cond_columns].astype(str) 
        encoded_list = []

        for column in self.cond_cols_in_order:
            encoder = self.encoders_in_order[self.columns_in_order.index(column)]
            encoded_data = encoder.transform(cond_df_encoded[[column]])
            encoded_list.append(torch.tensor(encoded_data, dtype=torch.float32))
        
        data_cond_encoded_final = torch.cat(encoded_list, dim=1)

        return data_cond_encoded_final
            

    def inv_transform_data(self, data):
        df = pd.DataFrame(data)

        if len(df.columns) != self.encoder_n_dim:
            raise ValueError(f"The number of columns in the input data ({len(df.columns)}) does not match the expected number of columns ({self.encoder_n_dim}) based on the encoder's configuration.")

        df_decoded = pd.DataFrame()
        index_units = 0 

        for encoder, units, column, dtype in zip(self.encoders_in_order, self.units_in_order, self.columns_in_order, self.d_types_in_order):   
            data_segment = df.iloc[:, index_units:index_units + units]
            decoded_data = encoder.inverse_transform(data_segment)
            df_decoded[column] = decoded_data.flatten().astype(dtype)
            index_units += units

        return df_decoded

    def inv_transform_cond(self, data):
        df = pd.DataFrame(data)
        
        if len(df.columns) != self.encode_n_cond_dim:  # Correctly check against cond dim
            raise ValueError(f"The number of columns in the input data ({len(df.columns)}) does not match the expected number of conditional columns ({self.encode_n_cond_dim}) based on the encoder's configuration.")

        df_decoded = pd.DataFrame()
        index_units = 0 

        for encoder, units, column in zip([self.encoders_in_order[i] for i in self.index_cond], [self.units_in_order[i] for i in self.index_cond], self.cond_cols_in_order):   
            data_segment = df.iloc[:, index_units:index_units + units]
            decoded_data = encoder.inverse_transform(data_segment)
            df_decoded[column] = decoded_data.flatten()
            index_units += units

        return df_decoded


    def transform_data_df_without_condition(self, df: pd.DataFrame):
        """
        Transforms a dataframe except the condition columns (self.cond_cols).
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


    def generate_random_condition(self, num_samples: int = 1) -> pd.DataFrame:
        random_conditions = {}
        for column in self.cond_cols_in_order:
            if column in self.categorical_columns:
                categories = self.encoders_in_order[self.columns_in_order.index(column)].categories_[0]
                random_conditions[column] = np.random.choice(categories, num_samples)
            elif column in self.ordinal_columns + self.numeric_columns:
                random_conditions[column] = np.random.uniform(0, 1, num_samples)

        random_conditions_df = pd.DataFrame(random_conditions)
       
        return random_conditions_df


    def get_condition_from_tensor(self, data_tensor: torch.tensor):
        """
        Extracts the condition from the input data tensor without detaching the gradients.
        """
        cond_tensors = [data_tensor[:, i:(i + self.units_in_order[i])] for i in self.index_cond]
        return torch.cat(cond_tensors, dim=1)  

    def get_only_data_from_tensor(self, data_tensor: torch.tensor):
        """
        Extracts only the data (excluding the condition) from the input data tensor without detaching the gradients.
        """
        data_tensor_no_cond = [data_tensor[:, i:(i + self.units_in_order[i])] for i in range(len(self.units_in_order)) if i not in self.index_cond]
        return torch.cat(data_tensor_no_cond, dim=1)  


    def encode_dim(self):
        return self.encoder_n_dim 
    
    def encode_cond_dim(self):
        return self.encode_n_cond_dim

    
    def get_units_per_column(self):
        """
        Returns the number of units (dimensions) per encoded column as a list, maintaining the original column order.
        """
        return self.units_in_order

    def get_units_per_cond_column(self):
        """
        Returns the number of units (dimensions) per encoded cond column as a list, maintaining the original column order.
        """
        return [self.units_in_order[i] for i in self.index_cond]


    def cols(self):
        return self.columns_in_order

    def cond_cols(self):
        return self.cond_cols_in_order
