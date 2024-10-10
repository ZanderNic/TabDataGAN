import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import pandas as pd

class Benchmark:
    def __init__(self):
        pass


    def mean_rfc(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, n_estimators=100, k_folds=10) -> float:
        classifier = RandomForestClassifier(n_estimators=n_estimators)
        combined_data = pd.concat([real_data, synthetic_data], ignore_index=True)
        labels = np.concatenate([np.ones(len(real_data)), np.zeros(len(synthetic_data))])
        scores = cross_val_score(classifier, combined_data, labels, cv=k_folds, scoring='accuracy')

        return np.mean(scores)