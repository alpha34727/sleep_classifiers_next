from abc import ABC, abstractmethod
from typing import Self
import numpy as np

class BaseClassifierStrategy(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Self:
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        pass
    
    @classmethod
    @abstractmethod
    def get_hyperparameter_grid(cls) -> dict:
        pass
