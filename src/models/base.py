from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np

class BaseSleepClassifier(ABC):
    """
    所有睡眠分期預測模型的抽象基類，強制規定統一的 scikit-learn 風格調用介面。
    """
    @property
    @abstractmethod
    def name(self) -> str:
        """模型名稱"""
        pass

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseSleepClassifier":
        """訓練模型"""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """預測睡眠類別"""
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """預測類別概率"""
        pass

    @abstractmethod
    def save_model(self, path: Path) -> None:
        """將模型儲存至硬碟"""
        pass

    @abstractmethod
    def load_model(self, path: Path) -> "BaseSleepClassifier":
        """從硬碟載入模型"""
        pass
