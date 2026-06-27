import xgboost as xgb
import lightgbm as lgb
from sklearn.utils import class_weight
import numpy as np
from typing import Self
from sleep_next.models.base import BaseClassifierStrategy

def get_sample_weights(y: np.ndarray) -> np.ndarray:
    classes = np.unique(y)
    weights = class_weight.compute_class_weight('balanced', classes=classes, y=y)
    weight_dict = {c: w for c, w in zip(classes, weights)}
    return np.array([weight_dict[val] for val in y], dtype=np.float32)

class XGBoostStrategy(BaseClassifierStrategy):
    def __init__(self, **kwargs):
        # Pass random_state for reproducibility
        self.model = xgb.XGBClassifier(
            random_state=42,
            n_jobs=-1,
            eval_metric="mlogloss",
            **kwargs
        )
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Self:
        sample_weight = get_sample_weights(y)
        self.model.fit(X, y, sample_weight=sample_weight)
        return self
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
        
    @classmethod
    def get_hyperparameter_grid(cls) -> dict:
        return {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.05, 0.1]
        }

class LightGBMStrategy(BaseClassifierStrategy):
    def __init__(self, **kwargs):
        self.model = lgb.LGBMClassifier(
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
            **kwargs
        )
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Self:
        sample_weight = get_sample_weights(y)
        self.model.fit(X, y, sample_weight=sample_weight)
        return self
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
        
    @classmethod
    def get_hyperparameter_grid(cls) -> dict:
        return {
            'num_leaves': [15, 31, 63],
            'learning_rate': [0.05, 0.1]
        }
