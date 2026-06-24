from typing import Any, Dict, List
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV
from sklearn.utils.class_weight import compute_class_weight
from joblib import Parallel, delayed

from src.models.base import BaseSleepClassifier

class SleepValidator:
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, 
                          y_pred: np.ndarray, 
                          y_prob: np.ndarray, 
                          n_classes: int = 2) -> Dict[str, float]:
        """
        計算 Accuracy, Sensitivity, Specificity, AUC 與 Cohen's Kappa 指標。
        支援二分類與三分類。
        """
        metrics = {}
        
        # 1. 整體指標
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["cohen_kappa"] = float(cohen_kappa_score(y_true, y_pred))
        
        # 2. AUC-ROC
        try:
            if n_classes == 2:
                # 傳入正類機率 (即 class 1)
                if y_prob.ndim == 2:
                    metrics["auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
                else:
                    metrics["auc"] = float(roc_auc_score(y_true, y_prob))
            else:
                metrics["auc"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
        except Exception:
            metrics["auc"] = 0.0

        # 3. Sensitivity 與 Specificity
        if n_classes == 2:
            # 二分類下：Sleep (1) 為陽性，Wake (0) 為陰性
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            # cm 結構: [[TN, FP], [FN, TP]]
            tn, fp, fn, tp = cm.ravel()
            
            # Sensitivity = TP / (TP + FN) = Recall of Class 1
            metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            # Specificity = TN / (TN + FP) = Recall of Class 0
            metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        else:
            # 三分類下 (0=Wake, 1=NREM, 2=REM)：使用 One-Vs-Rest 方式為每個類別計算
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
            sensitivities = []
            specificities = []
            
            for c in range(3):
                tp = cm[c, c]
                fn = np.sum(cm[c, :]) - tp
                fp = np.sum(cm[:, c]) - tp
                tn = np.sum(cm) - tp - fn - fp
                
                sens_c = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
                spec_c = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
                
                sensitivities.append(sens_c)
                specificities.append(spec_c)
                
                # 記錄單一類別的指標
                metrics[f"sensitivity_class_{c}"] = sens_c
                metrics[f"specificity_class_{c}"] = spec_c
            
            # 記錄 Macro 平均指標
            metrics["macro_sensitivity"] = float(np.mean(sensitivities))
            metrics["macro_specificity"] = float(np.mean(specificities))
            
        return metrics

    @staticmethod
    def run_loocv(model_cls: Any, 
                  X: np.ndarray, 
                  y: np.ndarray, 
                  groups: np.ndarray, 
                  n_classes: int = 2,
                  n_jobs: int = -1,
                  **model_kwargs) -> Dict[str, Any]:
        """
        執行 Leave-One-Group-Out (LOOCV) 交叉驗證。以受試者 (subject) 為 Group。
        """
        # 如果平行處理，為避免內部模型搶奪執行緒，將 internal n_jobs 設為 1 (如果未指定且 outer n_jobs != 1)
        if "n_jobs" not in model_kwargs and n_jobs != 1:
            model_kwargs["n_jobs"] = 1

        logo = LeaveOneGroupOut()
        folds = list(logo.split(X, y, groups=groups))
        
        results = Parallel(n_jobs=n_jobs)(
            delayed(_run_single_loocv_fold)(
                fold=fold,
                model_cls=model_cls,
                X=X,
                y=y,
                train_idx=train_idx,
                test_idx=test_idx,
                n_classes=n_classes,
                model_kwargs=model_kwargs
            )
            for fold, (train_idx, test_idx) in enumerate(folds)
        )
        
        subject_metrics = [res["metrics"] for res in results]
        all_true = [res["y_test"] for res in results]
        all_pred = [res["preds"] for res in results]
        all_prob = [res["probs"] for res in results]
        
        all_true = np.concatenate(all_true)
        all_pred = np.concatenate(all_pred)
        all_prob = np.concatenate(all_prob)
        
        # 1. 計算全域合併指標
        global_metrics = SleepValidator.calculate_metrics(all_true, all_pred, all_prob, n_classes)
        
        # 2. 計算受試者級別指標的 Mean ± Std
        subj_summary = {}
        metric_keys = list(subject_metrics[0].keys())
        for key in metric_keys:
            vals = [m[key] for m in subject_metrics]
            subj_summary[f"{key}_mean"] = float(np.mean(vals))
            subj_summary[f"{key}_std"] = float(np.std(vals))
            
        return {
            "global_metrics": global_metrics,
            "subject_summary": subj_summary,
            "raw_subject_metrics": subject_metrics
        }

    @staticmethod
    def run_mccv(model_cls: Any, 
                 X: np.ndarray, 
                 y: np.ndarray, 
                 groups: np.ndarray, 
                 n_splits: int = 20, 
                 train_size: float = 0.7, 
                 n_classes: int = 2,
                 n_jobs: int = -1,
                 **model_kwargs) -> Dict[str, Any]:
        """
        執行以受試者（Subject）為分組單位的 Monte Carlo 交叉驗證。
        每次隨機選擇 70% 的受試者作訓練，30% 的受試者作測試。
        """
        # 如果平行處理，為避免內部模型搶奪執行緒，將 internal n_jobs 設為 1 (如果未指定且 outer n_jobs != 1)
        if "n_jobs" not in model_kwargs and n_jobs != 1:
            model_kwargs["n_jobs"] = 1

        results = Parallel(n_jobs=n_jobs)(
            delayed(_run_single_mccv_fold)(
                trial=trial,
                model_cls=model_cls,
                X=X,
                y=y,
                groups=groups,
                train_size=train_size,
                n_classes=n_classes,
                model_kwargs=model_kwargs
            )
            for trial in range(n_splits)
        )
        
        trial_metrics = [res["metrics"] for res in results]
        all_y_test = [res["y_test"] for res in results]
        all_y_prob = [res["y_prob"] for res in results]
        
        # 彙總多個 trial 指標的 Mean ± Std
        mccv_summary = {}
        metric_keys = list(trial_metrics[0].keys())
        for key in metric_keys:
            vals = [m[key] for m in trial_metrics]
            mccv_summary[f"{key}_mean"] = float(np.mean(vals))
            mccv_summary[f"{key}_std"] = float(np.std(vals))
            
        return {
            "mccv_summary": mccv_summary,
            "raw_trial_metrics": trial_metrics,
            "all_y_test": all_y_test,
            "all_y_prob": all_y_prob
        }


def _run_single_loocv_fold(fold: int,
                           model_cls: Any,
                           X: np.ndarray,
                           y: np.ndarray,
                           train_idx: np.ndarray,
                           test_idx: np.ndarray,
                           n_classes: int,
                           model_kwargs: dict) -> dict:
    """單折 LOOCV 計算輔助函數 (放置於模組頂層以利多行程序列化)"""
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    model = model_cls(**model_kwargs)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)
    
    metrics = SleepValidator.calculate_metrics(y_test, preds, probs, n_classes)
    
    return {
        "metrics": metrics,
        "y_test": y_test,
        "preds": preds,
        "probs": probs
    }


def _run_single_mccv_fold(trial: int,
                          model_cls: Any,
                          X: np.ndarray,
                          y: np.ndarray,
                          groups: np.ndarray,
                          train_size: float,
                          n_classes: int,
                          model_kwargs: dict) -> dict:
    """單次 MCCV 計算輔助函數 (放置於模組頂層以利多行程序列化)

    對齊 legacy ClassifierService.train_classifier 行為：
    1. 若模型類別定義了 PARAM_GRID，執行 GridSearchCV(cv=3, scoring='roc_auc') 選出每折最佳超參數。
    2. 計算 balanced class_weight 並注入給支援此參數的模型（LR, RF, MLP）；KNN 不支援，跳過。
    """
    # ── 1. 受試者級別切分 ──────────────────────────────────────────────────────
    unique_subjects = np.unique(groups)
    n_subjects = len(unique_subjects)
    n_train_subjects = int(np.round(train_size * n_subjects))

    rng = np.random.default_rng(seed=42 + trial)
    shuffled_subjs = rng.permutation(unique_subjects)

    train_subjs = shuffled_subjs[:n_train_subjects]
    test_subjs = shuffled_subjs[n_train_subjects:]

    train_mask = np.isin(groups, train_subjs)
    test_mask = np.isin(groups, test_subjs)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    # ── 2. 計算 Balanced Class Weights (對齊 legacy get_class_weights) ────────
    # 僅特定分類器支援 class_weight 參數以控制注入，避免 KNN, MLP, XGBoost 發生建構子錯誤
    _SUPPORTS_CLASS_WEIGHT = model_cls.__name__ in {
        "LogisticRegressionClassifier",
        "RandomForestClassifierWrapper",
        "LightGBMClassifier",
    }
    extra_kwargs = model_kwargs.copy()

    if _SUPPORTS_CLASS_WEIGHT:
        unique_classes = np.unique(y_train)
        cw_array = compute_class_weight(
            class_weight="balanced",
            classes=unique_classes,
            y=y_train,
        )
        class_weight_dict = dict(zip(unique_classes.tolist(), cw_array.tolist()))
        extra_kwargs["class_weight"] = class_weight_dict

    # ── 3. GridSearchCV 超參數搜尋 (對齊 legacy ParameterSearch.run_search) ───
    # 策略：從 wrapper 的 defaults 直接建構內部 sklearn estimator，
    # 執行 GridSearchCV，取得 best_params_，再以 best_params_ 初始化最終 wrapper。
    # 不需要 probe fit，避免 KNN n_neighbors > n_samples 的邊界錯誤。
    param_grid = getattr(model_cls, "PARAM_GRID", None)
    if param_grid is not None:
        # 建立一個「無狀態」wrapper 實例，只取其 defaults 字典
        _probe = model_cls(**extra_kwargs)
        inner_model = _make_inner_estimator(_probe)

        # 對 inner sklearn estimator 執行 3-fold GridSearchCV
        grid_search = GridSearchCV(
            inner_model,
            param_grid,
            scoring="roc_auc",
            cv=3,
            n_jobs=1,  # 外層已平行化，內層設 1 避免資源競爭
            refit=False,  # 不需要 refit，我們自己用 best_params_ 重建
        )
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_

        # c. 用 best_params 建立最終模型並在完整訓練集 fit
        final_kwargs = extra_kwargs.copy()
        final_kwargs.update(best_params)
        model = model_cls(**final_kwargs)
        model.fit(X_train, y_train)
    else:
        # 無 PARAM_GRID 的現代模型 (LightGBM, XGBoost) 直接 fit
        model = model_cls(**extra_kwargs)
        model.fit(X_train, y_train)

    # ── 4. 推論 ────────────────────────────────────────────────────────────────
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)

    metrics = SleepValidator.calculate_metrics(y_test, preds, probs, n_classes)

    return {
        "metrics": metrics,
        "y_test": y_test,
        "y_prob": probs[:, 1] if n_classes == 2 else probs,
    }


def _make_inner_estimator(wrapper):
    """
    從 wrapper 實例的儲存參數建構對應的內部 sklearn estimator（不執行 fit）。

    支援的 wrapper 型別：
      - KNNClassifierWrapper       → KNeighborsClassifier
      - LogisticRegressionClassifier → LogisticRegression
      - RandomForestClassifierWrapper → RandomForestClassifier (clone)
      - MLPClassifierWrapper        → MLPClassifier (clone)

    目的：讓 GridSearchCV 可操作真正的 sklearn estimator，
    繞開 BaseSleepClassifier 不繼承 BaseEstimator（缺 get_params/set_params）的限制。
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.base import clone

    cls_name = type(wrapper).__name__

    if cls_name == "KNNClassifierWrapper":
        # KNN 不接受 class_weight，只取 KNN 合法參數
        params = wrapper.defaults.copy()
        knn_params = {k: v for k, v in params.items()
                      if k in ("n_neighbors", "weights", "metric", "p", "n_jobs")}
        return KNeighborsClassifier(**knn_params)

    elif cls_name == "LogisticRegressionClassifier":
        params = wrapper.defaults.copy()
        lr_params = {k: v for k, v in params.items()
                     if k in ("penalty", "C", "solver", "class_weight",
                               "random_state", "max_iter", "multi_class")}
        lr_params.setdefault("solver", "liblinear")
        lr_params.setdefault("max_iter", 1000)
        return LogisticRegression(**lr_params)

    elif cls_name in ("RandomForestClassifierWrapper", "MLPClassifierWrapper"):
        # RF 與 MLP 在 __init__ 時就建立了 _model，直接 clone 即可
        return clone(wrapper._model)

    else:
        raise TypeError(
            f"_make_inner_estimator: unsupported wrapper type '{cls_name}'. "
            "Add a branch here or remove PARAM_GRID from this model class."
        )
