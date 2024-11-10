import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt


class Benchmark:
    
    
    def mean_rfc(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, n_estimators=100, k_folds=10, plot=False, top_n=8, plot_path: str = "mean_rfc") -> float:
        classifier = RandomForestClassifier(n_estimators=n_estimators)
        combined_data = pd.concat([real_data, synthetic_data], ignore_index=True)
        labels = np.concatenate([np.ones(len(real_data)), np.zeros(len(synthetic_data))])
   
        scores = cross_val_score(classifier, combined_data, labels, cv=k_folds, scoring='accuracy')

        classifier.fit(combined_data, labels)
        feature_importance = classifier.feature_importances_

        if plot:
            if plot_path is None:
                raise ValueError("You must provide a valid plot_path if plot=True.")
            
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

