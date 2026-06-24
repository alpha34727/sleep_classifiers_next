import pickle
import warnings
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from src.models.base import BaseSleepClassifier

from sklearn.multiclass import OneVsRestClassifier

# 壓制 scikit-learn 庫內部的 API 轉型期警告與無關的特徵名稱 UserWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


class LogisticRegressionClassifier(BaseSleepClassifier):
    """L1 邏輯斯迴歸分類器 (對齊原著參數)"""
    def __init__(self, **kwargs):
        # 預設參數對齊原著：L1 懲罰項、liblinear 求解器
        self.defaults = {
            "penalty": "l1",
            "solver": "liblinear",
            "random_state": 42
        }
        self.defaults.update(kwargs)
        self._model = None

    @property
    def name(self) -> str:
        return "Logistic Regression"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionClassifier":
        unique_classes = np.unique(y)
        n_classes = len(unique_classes)

        if n_classes >= 3 and self.defaults.get("solver") == "liblinear":
            # 新版 scikit-learn 中 liblinear 不再直接支援多分類，須用 OneVsRestClassifier 封裝以對齊舊版 OVR 行為
            base_lr = LogisticRegression(**self.defaults)
            self._model = OneVsRestClassifier(base_lr)
        else:
            self._model = LogisticRegression(**self.defaults)

        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise ValueError("Model must be fitted before making predictions.")
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise ValueError("Model must be fitted before making predictions.")
        return self._model.predict_proba(X)

    def save_model(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump((self.defaults, self._model), f)

    def load_model(self, path: Path) -> "LogisticRegressionClassifier":
        with open(path, "rb") as f:
            self.defaults, self._model = pickle.load(f)
        return self



class RandomForestClassifierWrapper(BaseSleepClassifier):
    """隨機森林分類器 (對齊原著參數)"""
    def __init__(self, **kwargs):
        # 預設參數對齊原著
        defaults = {
            "n_estimators": 100,
            "max_features": 1.0,
            "max_depth": 10,
            "min_samples_split": 10,
            "min_samples_leaf": 32,
            "bootstrap": True,
            "random_state": 42,
            "n_jobs": -1
        }
        defaults.update(kwargs)
        self._model = RandomForestClassifier(**defaults)

    @property
    def name(self) -> str:
        return "Random Forest"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifierWrapper":
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)

    def save_model(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self._model, f)

    def load_model(self, path: Path) -> "RandomForestClassifierWrapper":
        with open(path, "rb") as f:
            self._model = pickle.load(f)
        return self


class MLPClassifierWrapper(BaseSleepClassifier):
    """多層感知器 (MLP) 深度學習分類器 (對齊原著參數)"""
    def __init__(self, **kwargs):
        # 預設參數對齊原著
        defaults = {
            "activation": "relu",
            "hidden_layer_sizes": (15, 15, 15),
            "max_iter": 2000,
            "alpha": 0.01,
            "solver": "adam",
            "n_iter_no_change": 20,
            "random_state": 42
        }
        defaults.update(kwargs)
        self._model = MLPClassifier(**defaults)

    @property
    def name(self) -> str:
        return "Neural Net"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPClassifierWrapper":
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)

    def save_model(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self._model, f)

    def load_model(self, path: Path) -> "MLPClassifierWrapper":
        with open(path, "rb") as f:
            self._model = pickle.load(f)
        return self
