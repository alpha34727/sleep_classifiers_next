import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sleep_classifiers_next.models.base import BaseSleepClassifier

class RandomForestSleepClassifier(BaseSleepClassifier):
    def __init__(self):
        clf = RandomForestClassifier(
            n_estimators=100,
            max_features=1.0,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=32,
            bootstrap=True,
            random_state=42
        )
        param_grid = {"max_depth": [10, 50, 100]}
        super().__init__(name="Random Forest", classifier=clf, param_grid=param_grid)

class LogisticRegressionSleepClassifier(BaseSleepClassifier):
    def __init__(self):
        clf = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            verbose=0,
            random_state=42
        )
        param_grid = {"C": [0.001, 0.01, 0.1, 1, 10, 100], "penalty": ["l1", "l2"]}
        super().__init__(name="Logistic Regression", classifier=clf, param_grid=param_grid)

class KNNSleepClassifier(BaseSleepClassifier):
    def __init__(self):
        clf = KNeighborsClassifier(weights="distance")
        param_grid = {"n_neighbors": [500, 1000]}
        super().__init__(name="k-Nearest Neighbors", classifier=clf, param_grid=param_grid)

    def train(self, X: np.ndarray, y: np.ndarray, scoring: str = "roc_auc") -> None:
        n_samples = len(X)
        valid_neighbors = [k for k in [500, 1000] if k < n_samples]
        if not valid_neighbors:
            valid_neighbors = [max(1, min(5, n_samples - 1))]
        self.param_grid = {"n_neighbors": valid_neighbors}
        super().train(X, y, scoring=scoring)

class MLPSleepClassifier(BaseSleepClassifier):
    def __init__(self, solver: str = "adam", max_iter: int = 2000):
        clf = MLPClassifier(
            activation="relu",
            hidden_layer_sizes=(15, 15, 15),
            max_iter=max_iter,
            alpha=0.01,
            solver=solver,
            verbose=False,
            n_iter_no_change=20,
            random_state=42
        )
        param_grid = {"alpha": [0.1, 0.01, 0.001, 0.0001, 0.00001]}
        super().__init__(name="Neural Net", classifier=clf, param_grid=param_grid)
