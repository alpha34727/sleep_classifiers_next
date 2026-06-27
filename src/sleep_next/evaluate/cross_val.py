import gc
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import polars as pl
from typing import List, Tuple, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.model_selection import GridSearchCV
from sleep_next.config import settings
from sleep_next.evaluate.metrics import compute_sleep_wake_performance, SleepWakePerformance

class RawPerformance:
    def __init__(self, true_labels: np.ndarray, class_probabilities: np.ndarray):
        self.true_labels = true_labels
        self.class_probabilities = class_probabilities

def load_fold_data(subject_ids: List[str], feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    lazy_frames = []
    for sid in subject_ids:
        path = settings.FEATURE_DIR / f"{sid}_features.parquet"
        if path.is_file():
            lazy_frames.append(pl.scan_parquet(path))
            
    if not lazy_frames:
        raise ValueError(f"No feature files found for subjects: {subject_ids}")
        
    df = pl.concat(lazy_frames).select(feature_cols + ["label"]).collect()
    X = df.select(feature_cols).to_numpy().astype(np.float32)
    y = df.select("label").to_numpy().flatten()
    return X, y

def run_single_fold(
    train_ids: List[str],
    test_ids: List[str],
    strategy_cls,
    strategy_kwargs: dict,
    feature_cols: List[str],
    classification_type: str, # "sleep_wake" or "three_class"
    scoring: str,
    compute_shap: bool = False,
    shap_output_path: str = ""
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Worker function executed in parallel.
    Loads data on-demand, trains strategy with GridSearchCV, and returns predictions.
    """
    # Load train and test subsets
    X_train, y_train = load_fold_data(train_ids, feature_cols)
    X_test, y_test = load_fold_data(test_ids, feature_cols)
    
    # Map labels:
    # y=0 is wake. y>0 is sleep.
    if classification_type == "sleep_wake":
        # Convert to 0/1: sleep (value > 0) -> 1, wake (0) -> 0
        y_train = (y_train > 0).astype(np.int32)
        y_test = (y_test > 0).astype(np.int32)
    elif classification_type == "three_class":
        # Convert: wake (0) -> 0, NREM (1,2,3,4) -> 1, REM (5) -> 2
        y_train_mapped = []
        for val in y_train:
            if val == 0:
                y_train_mapped.append(0)
            elif val == 5:
                y_train_mapped.append(2)
            else:
                y_train_mapped.append(1)
        y_train = np.array(y_train_mapped, dtype=np.int32)
        
        y_test_mapped = []
        for val in y_test:
            if val == 0:
                y_test_mapped.append(0)
            elif val == 5:
                y_test_mapped.append(2)
            else:
                y_test_mapped.append(1)
        y_test = np.array(y_test_mapped, dtype=np.int32)
        
    # Instantiate strategy
    strategy = strategy_cls(**strategy_kwargs)
    
    if "Logistic" in strategy_cls.__name__ and classification_type == "three_class":
        strategy.model.set_params(solver='saga')
        
    # Custom sample weighting if model supports it (like XGBoost / LightGBM)
    is_tree_model = "XGB" in strategy_cls.__name__ or "LightGBM" in strategy_cls.__name__
    
    # Grid search cross-validation
    param_grid = strategy.get_hyperparameter_grid()
    
    if is_tree_model:
        # compute sample weights
        from sklearn.utils import class_weight
        classes = np.unique(y_train)
        weights = class_weight.compute_class_weight('balanced', classes=classes, y=y_train)
        weight_dict = {c: w for c, w in zip(classes, weights)}
        sample_weight = np.array([weight_dict[val] for val in y_train], dtype=np.float32)
        
        grid_search = GridSearchCV(strategy.model, param_grid, scoring=scoring, cv=3)
        grid_search.fit(X_train, y_train, sample_weight=sample_weight)
        strategy.model.set_params(**grid_search.best_params_)
        strategy.fit(X_train, y_train)
    else:
        # If standard Sklearn classifier
        grid_search = GridSearchCV(strategy.model, param_grid, scoring=scoring, cv=3)
        grid_search.fit(X_train, y_train)
        strategy.model.set_params(**grid_search.best_params_)
        strategy.fit(X_train, y_train)
        
    class_probs = strategy.predict_proba(X_test)
    
    # Compute SHAP if requested and it is a tree model
    if compute_shap and is_tree_model:
        from sleep_next.visualize.plotters import plot_tree_shap_summary
        plot_tree_shap_summary(strategy.model, X_test, feature_cols, shap_output_path)
    
    # Free memory
    del X_train, y_train, X_test
    gc.collect()
    
    return y_test, class_probs

def run_cross_validation(
    subject_ids: List[str],
    strategy_cls,
    strategy_kwargs: dict,
    feature_cols: List[str],
    cv_type: str, # "mc" (Monte Carlo) or "loo" (Leave-One-Out)
    classification_type: str, # "sleep_wake" or "three_class"
    number_of_splits: int = 20,
    test_fraction: float = 0.3,
    scoring: str = "roc_auc",
    compute_shap_last_fold: bool = False,
    shap_output_path: str = ""
) -> List[RawPerformance]:
    from tqdm import tqdm
    
    splits = []
    
    if cv_type == "loo":
        for index in range(len(subject_ids)):
            train_set = subject_ids.copy()
            test_set = [train_set.pop(index)]
            splits.append((train_set, test_set))
    elif cv_type == "mc":
        # Shuffle deterministically to match trial expectations but maintain reproducibility per run
        rnd = random.Random(42)
        test_index = int(np.round(test_fraction * len(subject_ids)))
        for _ in range(number_of_splits):
            shuffled = subject_ids.copy()
            rnd.shuffle(shuffled)
            test_set = shuffled[:test_index]
            train_set = shuffled[test_index:]
            splits.append((train_set, test_set))
            
    raw_performances = []
    
    # Process folds in parallel using ProcessPoolExecutor
    with ProcessPoolExecutor() as executor:
        futures = []
        for idx, (train_ids, test_ids) in enumerate(splits):
            # Check if this is the last fold
            is_last_fold = (idx == len(splits) - 1)
            comp_shap = compute_shap_last_fold and is_last_fold
            
            futures.append(
                executor.submit(
                    run_single_fold,
                    train_ids=train_ids,
                    test_ids=test_ids,
                    strategy_cls=strategy_cls,
                    strategy_kwargs=strategy_kwargs,
                    feature_cols=feature_cols,
                    classification_type=classification_type,
                    scoring=scoring,
                    compute_shap=comp_shap,
                    shap_output_path=shap_output_path
                )
            )
            
        desc = f"[{strategy_cls.__name__[:12]} | MC {len(splits)} folds]"
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc, leave=False):
            try:
                y_test, class_probs = future.result()
                raw_performances.append(RawPerformance(y_test, class_probs))
            except Exception as e:
                print(f"Fold execution failed: {e}")
                
    return raw_performances
