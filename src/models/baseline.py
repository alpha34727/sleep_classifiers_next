import pickle
import warnings
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

from src.models.base import BaseSleepClassifier

from sklearn.multiclass import OneVsRestClassifier

# 壓制 scikit-learn 庫內部的 API 轉型期警告與無關的特徵名稱 UserWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


class LogisticRegressionClassifier(BaseSleepClassifier):
    """L1 邏輯斯迴歸分類器 (對齊原著參數)"""

    # 對齊 legacy ParameterSearch.parameter_dictionary
    PARAM_GRID = {"C": [0.001, 0.01, 0.1, 1, 10, 100], "penalty": ["l1", "l2"]}

    def __init__(self, **kwargs):
        # 預設參數對齊原著：L1 懲罰項、liblinear 求解器；class_weight='balanced' 對齊 legacy get_class_weights
        self.defaults = {
            "penalty": "l1",
            "solver": "liblinear",
            "class_weight": "balanced",
            "random_state": 42,
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

    # 對齊 legacy ParameterSearch.parameter_dictionary
    PARAM_GRID = {"max_depth": [10, 50, 100]}

    def __init__(self, **kwargs):
        # 預設參數對齊原著；class_weight='balanced' 對齊 legacy get_class_weights
        defaults = {
            "n_estimators": 100,
            "max_features": 1.0,
            "max_depth": 10,
            "min_samples_split": 10,
            "min_samples_leaf": 32,
            "bootstrap": True,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
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

    # 對齊 legacy ParameterSearch.parameter_dictionary
    PARAM_GRID = {"alpha": [0.1, 0.01, 0.001, 0.0001, 0.00001]}

    def __init__(self, **kwargs):
        # 預設參數對齊原著
        defaults = {
            "activation": "relu",
            "hidden_layer_sizes": (15, 15, 15),
            "max_iter": 2000,
            "alpha": 0.01,
            "solver": "adam",
            "n_iter_no_change": 20,
            "random_state": 42,
        }
        defaults.update(kwargs)
        defaults.pop("n_jobs", None)
        defaults.pop("class_weight", None)
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


class KNNClassifierWrapper(BaseSleepClassifier):
    """K-Nearest Neighbors 分類器 (K-NN)"""

    # 對齊 legacy ParameterSearch：每次 MC fold 中由 GridSearchCV 從 [500, 1000] 選最優 n_neighbors
    PARAM_GRID = {"n_neighbors": [500, 1000]}

    def __init__(self, **kwargs):
        # weights='distance' 對齊 legacy utils.get_classifiers()
        # KNN 不支援 class_weight，legacy 雖然會設定但對 KNN 無效
        self.defaults = {
            "n_neighbors": 500,
            "weights": "distance",
            "metric": "minkowski",
            "p": 2,
            "n_jobs": -1,
        }
        self.defaults.update(kwargs)
        self.defaults.pop("class_weight", None)
        self._model = None

    @property
    def name(self) -> str:
        return "k-Nearest Neighbors"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifierWrapper":
        n_samples = X.shape[0]
        params = self.defaults.copy()
        # 確保 n_neighbors 不會大於樣本數
        params["n_neighbors"] = min(self.defaults["n_neighbors"], n_samples)
        
        self._model = KNeighborsClassifier(**params)
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

    def load_model(self, path: Path) -> "KNNClassifierWrapper":
        with open(path, "rb") as f:
            self.defaults, self._model = pickle.load(f)
        return self
