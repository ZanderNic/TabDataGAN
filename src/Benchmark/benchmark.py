import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
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

            # Plotten und speichern
            plt.figure(figsize=(10, 6))
            plt.barh(top_features['Feature'], top_features['Importance'], color='skyblue')
            plt.xlabel('Feature Importance')
            plt.title(f'Top {top_n} Most Important Features')
            plt.gca().invert_yaxis()  # Umkehren der y-Achse, damit die wichtigsten Features oben sind
            plt.savefig(plot_path)
            plt.close()

        return np.mean(scores)