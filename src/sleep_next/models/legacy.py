from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils import class_weight
import numpy as np
from typing import Self
from sleep_next.models.base import BaseClassifierStrategy

def get_class_weights(y: np.ndarray) -> dict:
    classes = np.unique(y)
    weights = class_weight.compute_class_weight('balanced', classes=classes, y=y)
    return {c: float(w) for c, w in zip(classes, weights)}

class RandomForestStrategy(BaseClassifierStrategy):
    def __init__(self, **kwargs):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_features=1.0,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=32,
            bootstrap=True,
            **kwargs
        )
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Self:
        self.model.class_weight = get_class_weights(y)
        self.model.fit(X, y)
        return self
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
        
    @classmethod
    def get_hyperparameter_grid(cls) -> dict:
        return {'max_depth': [10, 50, 100]}

class LogisticRegressionStrategy(BaseClassifierStrategy):
    def __init__(self, **kwargs):
        self.model = LogisticRegression(
            penalty='l1',
            solver='liblinear',
            **kwargs
        )
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Self:
        self.model.class_weight = get_class_weights(y)
        self.model.fit(X, y)
        return self
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
        
    @classmethod
    def get_hyperparameter_grid(cls) -> dict:
        return {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2']}

class KNNStrategy(BaseClassifierStrategy):
    def __init__(self, **kwargs):
        self.model = KNeighborsClassifier(
            weights='distance',
            **kwargs
        )
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Self:
        self.model.fit(X, y)
        return self
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
        
    @classmethod
    def get_hyperparameter_grid(cls) -> dict:
        return {'n_neighbors': [500, 1000]}

class NeuralNetStrategy(BaseClassifierStrategy):
    def __init__(self, **kwargs):
        self.model = MLPClassifier(
            activation='relu',
            hidden_layer_sizes=(15, 15, 15),
            max_iter=2000,
            alpha=0.01,
            solver='adam',
            verbose=False,
            n_iter_no_change=20,
            early_stopping=True,
            **kwargs
        )
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Self:
        self.model.fit(X, y)
        return self
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
        
    @classmethod
    def get_hyperparameter_grid(cls) -> dict:
        return {'alpha': [0.1, 0.01, 0.001]}
