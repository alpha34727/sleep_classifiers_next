import random
import gc
import numpy as np
import polars as pl
from pathlib import Path
from pydantic import BaseModel
from sleep_classifiers_next.config import Settings
from sleep_classifiers_next.models.base import BaseSleepClassifier

class DataSplit(BaseModel):
    training_set: list[str]
    testing_set: list[str]

class CrossValidationService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.paths = settings.paths

    def get_loso_splits(self, subject_ids: list[str]) -> list[DataSplit]:
        splits = []
        for i in range(len(subject_ids)):
            test_subject = subject_ids[i]
            train_subjects = [sid for j, sid in enumerate(subject_ids) if j != i]
            splits.append(DataSplit(training_set=train_subjects, testing_set=[test_subject]))
        return splits

    def get_monte_carlo_splits(self, subject_ids: list[str], test_fraction: float = 0.3, num_splits: int = 20, seed: int = 42) -> list[DataSplit]:
        random.seed(seed)
        splits = []
        num_test = int(np.round(len(subject_ids) * test_fraction))
        for _ in range(num_splits):
            shuffled = list(subject_ids)
            random.shuffle(shuffled)
            test_set = shuffled[:num_test]
            train_set = shuffled[num_test:]
            splits.append(DataSplit(training_set=train_set, testing_set=test_set))
        return splits

    def load_features_and_labels(self, subject_ids: list[str], feature_cols: list[str], is_three_class: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        Loads pre-computed features from Parquet files for the specified subjects.
        Returns X (features) and y (mapped labels).
        """
        dfs = []
        for sid in subject_ids:
            path = self.paths.features_dir / f"{sid}_features.parquet"
            if path.exists():
                dfs.append(pl.read_parquet(path))
            else:
                raise FileNotFoundError(f"Parquet file not found for subject {sid}: {path}")

        if not dfs:
            return np.empty((0, len(feature_cols))), np.empty((0,))

        # Concatenate subject dataframes
        df_all = pl.concat(dfs)
        X = df_all.select(feature_cols).to_numpy()
        raw_labels = df_all.select("label").to_numpy().flatten()

        # Map labels
        if is_three_class:
            # 0=Wake (raw label 0), 1=NREM (raw label 1,2,3,4), 2=REM (raw label 5)
            y = np.zeros(len(raw_labels), dtype=int)
            y[raw_labels == 0] = 0
            y[(raw_labels >= 1) & (raw_labels <= 4)] = 1
            y[raw_labels == 5] = 2
        else:
            # Sleep-Wake (0=Wake, 1=Sleep)
            y = np.where(raw_labels > 0, 1, 0)

        return X, y

    def run_cross_validation(
        self,
        classifier: BaseSleepClassifier,
        splits: list[DataSplit],
        feature_cols: list[str],
        is_three_class: bool = False,
        scoring: str = "roc_auc"
    ) -> list[dict]:
        """
        Runs cross-validation over the splits.
        Returns a list of result dictionaries containing true labels and predicted probabilities for each split.
        """
        results = []
        for idx, split in enumerate(splits):
            if self.settings.verbose:
                print(f"[{classifier.name}] Running fold {idx+1}/{len(splits)}...")

            # 1. Load train inputs
            X_train, y_train = self.load_features_and_labels(split.training_set, feature_cols, is_three_class)
            
            # 2. Train model (with grid search & class weighting)
            classifier.train(X_train, y_train, scoring=scoring)
            
            # Free training variables and GC
            del X_train, y_train
            gc.collect()

            # 3. Evaluate on test set sequentially
            # We predict subject by subject to preserve memory and collect results
            fold_true_labels = []
            fold_probs = []
            
            for test_sid in split.testing_set:
                X_test, y_test = self.load_features_and_labels([test_sid], feature_cols, is_three_class)
                if len(y_test) == 0:
                    continue
                probs = classifier.predict_proba(X_test)
                fold_true_labels.append(y_test)
                fold_probs.append(probs)

            if fold_true_labels:
                results.append({
                    "true_labels": np.concatenate(fold_true_labels),
                    "probabilities": np.concatenate(fold_probs),
                    "testing_subjects": split.testing_set
                })
            
            gc.collect()

        return results
