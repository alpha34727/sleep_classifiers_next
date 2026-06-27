from abc import ABC, abstractmethod
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.utils import class_weight

class BaseSleepClassifier(ABC):
    def __init__(self, name: str, classifier, param_grid: dict):
        self.name = name
        self.classifier = classifier
        self.param_grid = param_grid
        self.best_params = None

    def train(self, X: np.ndarray, y: np.ndarray, scoring: str = "roc_auc") -> None:
        """
        Calculates balanced class weights, runs 3-fold Grid Search CV over self.param_grid,
        configures the classifier with best parameters, and fits it on training data.
        """
        # Calculate balanced class weights
        classes = np.unique(y)
        weights = class_weight.compute_class_weight("balanced", classes=classes, y=y)
        weight_dict = {c: w for c, w in zip(classes, weights)}
        
        # Apply class weights to classifier if it supports them
        if hasattr(self.classifier, "class_weight"):
            self.classifier.class_weight = weight_dict
        
        # Grid Search
        grid_search = GridSearchCV(self.classifier, self.param_grid, scoring=scoring, cv=3)
        grid_search.fit(X, y)
        self.best_params = grid_search.best_params_
        
        # Set parameters and refit
        self.classifier.set_params(**self.best_params)
        self.classifier.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classifier.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.classifier.predict_proba(X)
