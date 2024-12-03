import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor

from typing import List


class Benchmark:
    
    def mean_rfc(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, n_estimators=100, k_folds=10, plot=False, top_n=8, plot_path: str = "mean_rfc") -> float:
        """
            Evaluates the separability between real and synthetic data using random forest.
        """
        
        classifier = RandomForestClassifier(n_estimators=n_estimators)
        combined_data = pd.concat([real_data, synthetic_data], ignore_index=True)
        labels = np.concatenate([np.ones(len(real_data)), np.zeros(len(synthetic_data))])
   
        scores = cross_val_score(classifier, combined_data, labels, cv=k_folds, scoring='accuracy')

        if plot:
            if plot_path is None:
                raise ValueError("You must provide a valid plot_path if plot=True.")
            
            classifier.fit(combined_data, labels)   
            feature_importance = classifier.feature_importances_

            feature_names = real_data.columns.tolist()  
            feature_data = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importance
            })

            feature_data = feature_data.sort_values(by='Importance', ascending=False)

            top_features = feature_data.head(top_n)

            plt.figure(figsize=(10, 6))
            plt.barh(top_features['Feature'], top_features['Importance'], color='skyblue')
            plt.xlabel('Feature Importance')
            plt.title(f'Top {top_n} Most Important Features')
            plt.gca().invert_yaxis() 
            plt.savefig(plot_path)
            plt.close()

        return np.mean(scores)
    

    def mean_lr(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, k_folds=10) -> float:
        """
        Evaluates the separability between real and synthetic data using Logistic Regression.
        """
        
        classifier = LogisticRegression(max_iter=1000)
        
        combined_data = pd.concat([real_data, synthetic_data], ignore_index=True)
        labels = np.concatenate([np.ones(len(real_data)), np.zeros(len(synthetic_data))])

        scores = cross_val_score(classifier, combined_data, labels, cv=k_folds, scoring='accuracy')

        return np.mean(scores)
    

    def pca_visualization(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, plot_path: str = "pca_visualization.png"):
        """
        Performs PCA on the combined dataset and plots the first two principal components
        """
        combined_data = pd.concat([real_data, synthetic_data], ignore_index=True)
        labels = np.concatenate([np.ones(len(real_data)), np.zeros(len(synthetic_data))])

        for column in combined_data.columns:
            if combined_data[column].dtype == 'object':
                le = LabelEncoder()
                combined_data[column] = le.fit_transform(combined_data[column])

        scaler = MinMaxScaler()
        combined_data_scaled = scaler.fit_transform(combined_data)

        pca = PCA(n_components=2)
        components = pca.fit_transform(combined_data_scaled)

        plt.figure(figsize=(10, 6))
        plt.scatter(components[labels == 1, 0], components[labels == 1, 1], label='Real Data', alpha=0.5)
        plt.scatter(components[labels == 0, 0], components[labels == 0, 1], label='Synthetic Data', alpha=0.5)
        plt.legend()
        plt.title('PCA Visualization')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.savefig(plot_path)
        plt.close()


    def mlEfficiency(self, real_data_df, gen_data_df, target_column, task_type="classification", model=None, epochs=20, batch_size=32, learning_rate=0.001):
        """
        Evaluates the ML efficiency of generated data using a neural network (default) or a custom torch model that has implementet the fit and the eval method with a df.
        It trains the model on the gen_data and evals it after this on the real data ret the evaluation on the real data

        Args:
            real_data_df (pd.DataFrame): Real data, including the target column.
            gen_data_df (pd.DataFrame): Generated data, including the target column.
            target_column (str): The name of the target column.
            task_type (str): "classification" or "regression".
            model (torch.nn.Module, optional): Custom PyTorch model implementing `fit` and `evaluate` methods.
            epochs (int): Number of epochs for training the default model.
            batch_size (int): Batch size for training.
            learning_rate (float): Learning rate for the default model.
        Returns:
            dict: Evaluation results for real and generated data, including metrics.
        """
        X_test = real_data_df.drop(columns=[target_column]).values
        y_test = real_data_df[target_column].values
        X_train_gen = gen_data_df.drop(columns=[target_column]).values
        y_train_gen = gen_data_df[target_column].values

        # Train-test split
        X_train, X_val, y_train, y_val = train_test_split(X_train_gen, y_train_gen, test_size=0.2)

        if model is None:  
            if task_type == "classification":
                model = MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=epochs, learning_rate_init=learning_rate)
                metric = accuracy_score
                metric_name = "Accuracy"
            elif task_type == "regression":
                model = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=epochs, learning_rate_init=learning_rate)
                metric = mean_squared_error
                metric_name = "MSE"
            else:
                raise ValueError("Invalid task_type. Use 'classification' or 'regression'.")

        model.fit(X_train, y_train)
        y_pred_val = model.predict(X_val)
        val_metric = metric(y_val, y_pred_val)

        y_pred_test = model.predict(X_test)
        test_metric = metric(y_train_gen, y_pred_test)

        return {
            f"{metric_name} (Validation Set (20% of the generated data))": val_metric,
            f"{metric_name} (Test Set (100% of the real data)": test_metric,
        }
    

    def absCovPlot(self, real_data_df: pd.DataFrame, gen_data_df: pd.DataFrame, plot_path=None, cat_cols: List[str] = []):
        """
            This Function will plot a cov plot that reprecents the absolut diverence betwen the real data cov and the gen data cov path please ceap in mind how covarince works
            Args:
                real_data_df (pd.DataFrame): Real data, including the target column.
                gen_data_df (pd.DataFrame): Generated data, including the target column.
                cat_cols (list of str) : categorical cols, all colls will be selcetet that are categorical, but if the df type is numeric but it should be tretet as categorical add it here
                plot_path (str) : the plot will be saved to the specivic path that is providet here
                only_syn_data : the plot will only contain the synthetic covPlot
            Returns:
                fig (matplotlib.figure.Figure): The matplotlib figure object for the plot. 
        """
        if list(real_data_df.columns) != list(gen_data_df.columns):
            raise ValueError("The columns of the generated data and the real data don't match")


        if cat_cols:    # if cat cols cast the cols to category so they will not be selected 
            for col in cat_cols:
                if col in real_data_df.columns and real_data_df[col].dtype != 'category':
                    real_data_df[col] = real_data_df[col].astype('category')
                if col in gen_data_df.columns and gen_data_df[col].dtype != 'category':
                    gen_data_df[col] = gen_data_df[col].astype('category')


        numeric_cols = real_data_df.select_dtypes(include=[np.number]).columns

        scaler = StandardScaler()
        real_data_scaled = scaler.fit_transform(real_data_df[numeric_cols])
        gen_data_scaled = scaler.transform(gen_data_df[numeric_cols])

        real_data_scaled_df = pd.DataFrame(real_data_scaled, columns=numeric_cols)
        gen_data_scaled_df = pd.DataFrame(gen_data_scaled, columns=numeric_cols)

        real_cov = real_data_scaled_df.cov()
        gen_cov = gen_data_scaled_df.cov()

        abs_diff_cov = np.abs(real_cov - gen_cov)

        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(abs_diff_cov, annot=False, cmap="Blues", cbar=True, ax=ax, vmin=0, vmax=1)
        ax.set_title("Absolute Difference of Covariance Matrices\n(Real and Generated Data)")
        plt.tight_layout()

        if plot_path:
            plt.savefig(plot_path)

        return fig 
    

    def absCorrPlot(self, real_data_df: pd.DataFrame, gen_data_df: pd.DataFrame, plot_path=None, cat_cols: List[str] = []):
        """
        Plots the absolute difference between the pairwise column correlations of real and generated data.

        Args:
            real_data_df (pd.DataFrame): Real data, including the target column.
            gen_data_df (pd.DataFrame): Generated data, including the target column.
            cat_cols (list of str): List of categorical columns to exclude.
            plot_path (str): Path to save the plot.

        Returns:
            fig (matplotlib.figure.Figure): The matplotlib figure object for the plot.
        """
        if list(real_data_df.columns) != list(gen_data_df.columns):
            raise ValueError("The columns of the generated data and the real data don't match")

        if cat_cols:    
            numeric_cols = real_data_df.select_dtypes(include=[np.number]).columns.difference(cat_cols)
        else:
            numeric_cols = real_data_df.select_dtypes(include=[np.number]).columns

        
        real_corr = real_data_df[numeric_cols].corr()
        gen_corr = gen_data_df[numeric_cols].corr()

        abs_diff_corr = np.abs(real_corr - gen_corr)


        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(abs_diff_corr, annot=False, cmap="Blues", cbar=True, ax=ax, vmin=0, vmax=1)
        ax.set_title("Absolute Difference of Pairwise Correlations\n(Real and Generated Data)")
        plt.tight_layout()

        if plot_path:
            plt.savefig(plot_path)
        return fig
