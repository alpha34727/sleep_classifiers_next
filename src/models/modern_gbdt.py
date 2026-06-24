import pickle
import warnings
from pathlib import Path
import numpy as np
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from src.models.base import BaseSleepClassifier

# 壓制 LightGBM 預測時特徵名稱不匹配的無關 UserWarning
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


class LightGBMClassifier(BaseSleepClassifier):
    """LightGBM 梯度提升樹分類器"""
    def __init__(self, **kwargs):
        # 預設的一些常用睡眠分類優化參數
        self.defaults = {
            "n_estimators": 100,
            "learning_rate": 0.05,
            "max_depth": 6,
            "num_leaves": 31,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1
        }
        self.defaults.update(kwargs)
        self._model = None

    @property
    def name(self) -> str:
        return "LightGBM"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LightGBMClassifier":
        unique_classes = np.unique(y)
        n_classes = len(unique_classes)

        params = self.defaults.copy()
        if n_classes == 2:
            params["objective"] = "binary"
        else:
            params["objective"] = "multiclass"
            params["num_class"] = n_classes

        # 動態初始化模型
        self._model = LGBMClassifier(**params)
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

    def load_model(self, path: Path) -> "LightGBMClassifier":
        with open(path, "rb") as f:
            self.defaults, self._model = pickle.load(f)
        return self


class XGBoostClassifier(BaseSleepClassifier):
    """XGBoost 梯度提升樹分類器"""
    def __init__(self, **kwargs):
        # 預設的一些常用睡眠分類優化參數
        self.defaults = {
            "n_estimators": 100,
            "learning_rate": 0.05,
            "max_depth": 6,
            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "logloss"
        }
        self.defaults.update(kwargs)
        self.defaults.pop("class_weight", None)
        self._model = None

    @property
    def name(self) -> str:
        return "XGBoost"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostClassifier":
        unique_classes = np.unique(y)
        n_classes = len(unique_classes)

        params = self.defaults.copy()
        if n_classes == 2:
            params["objective"] = "binary:logistic"
        else:
            params["objective"] = "multi:softprob"
            params["num_class"] = n_classes

        # 動態初始化模型
        self._model = XGBClassifier(**params)
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

    def load_model(self, path: Path) -> "XGBoostClassifier":
        with open(path, "rb") as f:
            self.defaults, self._model = pickle.load(f)
        return self
