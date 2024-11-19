import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.mixture import BayesianGaussianMixture


class DataEncoder(object):
    """
        Data preprocessing for GANs for tabular data, handling numeric, ordinal, and categorical columns.

        The `DataEncoder` provides three different numerical feature encoding methods:
        
        Min-Max Scaling:
            - Transforms numerical data to the range [0, 1] using min-max normalization.

        Gaussian Mixture Model (GMM) Encoding
            - Applies a Bayesian Gaussian Mixture Model to decompose numerical data into probabilistic responsibilities for multiple Gaussian components.
            - Encodes the data as probabilities across GMM components.
            - Often struggles with dominant responsibilities (one component dominating others so the Generator struggels to learn something)

        CTGAN-style Encoding
            - Based on the CTGAN paper's methodology.
            - GMM responsibilities are sampled to create one-hot encoded representations.
            - These are then scaled with an "alpha" value to maintain a balance between data density and variety.
            - Suitable for capturing complex distributions while ensuring efective GAN training.
     """

    def __init__(
            self,
            raw_df: pd.DataFrame,
            categorical_columns: list,
            ordinal_columns: list,
            numeric_columns: list,
            cond_cols: list,
            random_state: int = None,
            cont_transform_method: str = "CTGAN",  # "min_max", "GMM", or "CTGAN"
            gmm_n_components: int = 30  # Number of GMM components for GMM and CTGAN transformation
    ):
        self.cont_transform_method = cont_transform_method

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
        self.encode_n_cond_dim = 0
        self.cond_cols = cond_cols

        self.random_state = random_state

        self.gmm_n_components = gmm_n_components

        self.preprocess_columns(raw_df)

    def preprocess_columns(self, df: pd.DataFrame):
        df = df.copy()

        if self.cont_transform_method == "GMM" or self.cont_transform_method == "CTGAN":
            print("Start Transformation with GMM this could take a while")

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
                if self.cont_transform_method == "min_max":
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    scaler.fit(df[[column]])
                    self.encoders_in_order.append(scaler)
                    self.units_in_order.append(1)

                elif self.cont_transform_method in ["GMM", "CTGAN"]:
                    gmm = BayesianGaussianMixture(
                        n_components=self.gmm_n_components,
                        max_iter=100,
                        weight_concentration_prior=0.0001,
                        random_state=self.random_state
                    )
                    gmm.fit(df[[column]].values)
                    self.encoders_in_order.append(gmm)
                    if self.cont_transform_method == "CTGAN":
                        self.units_in_order.append(self.gmm_n_components + 1)  # num components + Alpha
                    else:
                        self.units_in_order.append(self.gmm_n_components)
                else:
                    raise ValueError(f"The cont_transform_method '{self.cont_transform_method}' doesn't exist. Please use 'min_max', 'GMM', or 'CTGAN'.")

        self.encoder_n_dim = sum(self.units_in_order)
        self.encode_n_cond_dim = sum([self.units_in_order[i] for i in self.index_cond])
        self.units_accumulate = [0] + np.cumsum(self.units_in_order).tolist()


    def transform_data(self, df: pd.DataFrame):
        if list(df.columns) != list(self.columns_in_order):
            raise ValueError("The columns of the DataFrame are not in the expected order.")

        df_encoded = df.copy()
        df_encoded[self.categorical_columns] = df_encoded[self.categorical_columns].astype(str)
        encoded_list = []

        for encoder, column in zip(self.encoders_in_order, self.columns_in_order):
            if column in self.categorical_columns:
                encoded_data = encoder.transform(df_encoded[[column]])
                encoded_list.append(torch.tensor(encoded_data, dtype=torch.float32))
            elif column in self.ordinal_columns + self.numeric_columns:
                if self.cont_transform_method == "min_max":
                    encoded_data = encoder.transform(df_encoded[[column]])
                    encoded_list.append(torch.tensor(encoded_data, dtype=torch.float32))
                elif self.cont_transform_method == "GMM":
                    responsibilities = encoder.predict_proba(df_encoded[[column]].values)
                    encoded_list.append(torch.tensor(responsibilities, dtype=torch.float32))
                elif self.cont_transform_method == "CTGAN":
                    # as introduced by the CTGAN paper 
                    responsibilities = encoder.predict_proba(df_encoded[[column]].values)
                    sampled_components = np.array([np.random.choice(len(r), p=r) for r in responsibilities])
                    one_hot_encoded = np.eye(self.gmm_n_components)[sampled_components]
                    alpha = df_encoded[[column]].values.flatten() - encoder.means_[sampled_components].flatten()
                    alpha = alpha.reshape(-1, 1)
                    transformed = np.hstack([one_hot_encoded, alpha])
                    encoded_list.append(torch.tensor(transformed, dtype=torch.float32))

        data_encoded_final = torch.cat(encoded_list, dim=1)

        return data_encoded_final


    def transform_cond(self, cond_df: pd.DataFrame):
        cond_df_encoded = cond_df.copy()
        cond_df_encoded[self.categorical_cond_columns] = cond_df_encoded[self.categorical_cond_columns].astype(str)
        encoded_list = []

        for column in self.cond_cols_in_order:
            encoder = self.encoders_in_order[self.columns_in_order.index(column)]
            if column in self.categorical_columns:
                encoded_data = encoder.transform(cond_df_encoded[[column]])
                encoded_list.append(torch.tensor(encoded_data, dtype=torch.float32))
            elif column in self.ordinal_columns + self.numeric_columns:
                if self.cont_transform_method == "min_max":
                    encoded_data = encoder.transform(cond_df_encoded[[column]])
                    encoded_list.append(torch.tensor(encoded_data, dtype=torch.float32))
                elif self.cont_transform_method == "GMM":
                    responsibilities = encoder.predict_proba(cond_df_encoded[[column]].values)
                    encoded_list.append(torch.tensor(responsibilities, dtype=torch.float32))
                elif self.cont_transform_method == "CTGAN":
                    responsibilities = encoder.predict_proba(cond_df_encoded[[column]].values)
                    sampled_components = np.array([np.random.choice(len(r), p=r) for r in responsibilities])
                    one_hot_encoded = np.eye(self.gmm_n_components)[sampled_components]
                    alpha = cond_df_encoded[[column]].values.flatten() - encoder.means_[sampled_components].flatten()
                    alpha = alpha.reshape(-1, 1)
                    transformed = np.hstack([one_hot_encoded, alpha])
                    encoded_list.append(torch.tensor(transformed, dtype=torch.float32))

        data_cond_encoded_final = torch.cat(encoded_list, dim=1)

   
        return data_cond_encoded_final

    def inv_transform_data(self, data):
        if isinstance(data, torch.Tensor):
            data = data.cpu().detach().numpy()

        df = pd.DataFrame(data)

        if len(df.columns) != self.encoder_n_dim:
            raise ValueError(f"The number of columns in the input data ({len(df.columns)}) does not match the expected number of columns ({self.encoder_n_dim}) based on the encoder's configuration.")

        df_decoded = pd.DataFrame()
        index_units = 0

        for encoder, units, column, dtype in zip(self.encoders_in_order, self.units_in_order, self.columns_in_order, self.d_types_in_order):
            data_segment = df.iloc[:, index_units:index_units + units].values
            if column in self.categorical_columns:
                decoded_data = encoder.inverse_transform(data_segment)
                df_decoded[column] = decoded_data.flatten()
            elif column in self.ordinal_columns + self.numeric_columns:
                if self.cont_transform_method == "min_max":
                    decoded_data = encoder.inverse_transform(data_segment)
                    df_decoded[column] = decoded_data.flatten().astype(dtype)
                elif self.cont_transform_method == "GMM":
                    means = encoder.means_.flatten()
                    decoded_values = np.dot(data_segment, means)
                    df_decoded[column] = decoded_values.astype(dtype)
                elif self.cont_transform_method == "CTGAN":
                    one_hot_part = data_segment[:, :-1]
                    alpha_part = data_segment[:, -1]
                    sampled_components = np.argmax(one_hot_part, axis=1)
                    decoded_values = encoder.means_[sampled_components].flatten() + alpha_part
                    df_decoded[column] = decoded_values.astype(dtype)
            index_units += units

        return df_decoded


    def inv_transform_cond(self, data):
        df = pd.DataFrame(data.cpu().detach().numpy())

        if len(df.columns) != self.encode_n_cond_dim:
            raise ValueError(f"The number of columns in the input data ({len(df.columns)}) does not match the expected number of conditional columns ({self.encode_n_cond_dim}) based on the encoder's configuration.")

        df_decoded = pd.DataFrame()
        index_units = 0

        cond_dtypes = [self.d_types_in_order[i] for i in self.index_cond]

        for encoder, units, column, dtype in zip(
                [self.encoders_in_order[i] for i in self.index_cond],
                [self.units_in_order[i] for i in self.index_cond],
                self.cond_cols_in_order,
                cond_dtypes):
            data_segment = df.iloc[:, index_units:index_units + units].values
            if column in self.categorical_columns:
                decoded_data = encoder.inverse_transform(data_segment)
                df_decoded[column] = decoded_data.flatten()
            elif column in self.ordinal_columns + self.numeric_columns:
                if self.cont_transform_method == "min_max":
                    decoded_data = encoder.inverse_transform(data_segment)
                    df_decoded[column] = decoded_data.flatten().astype(dtype)
                elif self.cont_transform_method == "GMM":
                    means = encoder.means_.flatten()
                    decoded_values = np.dot(data_segment, means)
                    df_decoded[column] = decoded_values.astype(dtype)
                elif self.cont_transform_method == "CTGAN":
                    one_hot_part = data_segment[:, :-1]
                    alpha_part = data_segment[:, -1]
                    sampled_components = np.argmax(one_hot_part, axis=1)
                    decoded_values = encoder.means_[sampled_components].flatten() + alpha_part
                    df_decoded[column] = decoded_values.astype(dtype)
            index_units += units

        return df_decoded


    def generate_random_condition(self, num_samples: int = 1) -> pd.DataFrame:
        if self.random_state is not None:
            np.random.seed(self.random_state)

        random_conditions = {}
        for column in self.cond_cols_in_order:
            if column in self.categorical_columns:
                categories = self.encoders_in_order[self.columns_in_order.index(column)].categories_[0]
                random_conditions[column] = np.random.choice(categories, num_samples)
            elif column in self.ordinal_columns + self.numeric_columns:
                if self.cont_transform_method == "min_max":
                    random_uniform = np.random.uniform(0, 1, num_samples)
                    scaler = self.encoders_in_order[self.columns_in_order.index(column)]
                    random_conditions[column] = scaler.inverse_transform(random_uniform.reshape(-1, 1)).flatten()
                elif self.cont_transform_method in ["GMM", "CTGAN"]:
                    gmm = self.encoders_in_order[self.columns_in_order.index(column)]
                    sampled_components = gmm.sample(num_samples)[1]
                    if self.cont_transform_method == "CTGAN":
                        alpha = np.random.uniform(-1, 1, num_samples)
                        sampled_values = gmm.means_[sampled_components].flatten() + alpha
                        random_conditions[column] = sampled_values
                    else:
                        random_conditions[column] = gmm.sample(num_samples)[0].flatten()

        random_conditions_df = pd.DataFrame(random_conditions)

        return random_conditions_df
    

    def get_condition_from_tensor(self, data_tensor: torch.tensor):
        """
        Extracts the conditions from the input data tensor without detaching gradients.
        """
        cond_tensors = [data_tensor[:, self.units_accumulate[i]: self.units_accumulate[i + 1]] for i in self.index_cond]
        return torch.cat(cond_tensors, dim=1)

    def get_only_data_from_tensor(self, data_tensor: torch.tensor):
        """
        Extracts only the data (excluding conditions) from the input data tensor without detaching gradients.
        """
        index_not_cond = [i for i in range(len(self.cols())) if i not in self.index_cond]
        data_tensor_no_cond = [data_tensor[:, self.units_accumulate[i]: self.units_accumulate[i + 1]] for i in index_not_cond]
        return torch.cat(data_tensor_no_cond, dim=1)

    def get_units_per_column(self):
        """
        Returns the number of units (dimensions) per encoded column as a list, in the original column order.
        """
        return self.units_in_order

    def get_units_per_cond_column(self):
        """
        Returns the number of units (dimensions) per encoded condition column as a list, in the original column order.
        """
        return [self.units_in_order[i] for i in self.index_cond]

    def cols(self):
        return self.columns_in_order

    def cond_cols(self):
        return self.cond_cols_in_order

    def get_units_accumulate(self):
        return self.units_accumulate

    def encode_dim(self):
        return self.encoder_n_dim

    def encode_cond_dim(self):
        return self.encode_n_cond_dim