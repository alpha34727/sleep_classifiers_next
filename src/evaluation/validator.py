from typing import Any, Dict, List
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut

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
                  **model_kwargs) -> Dict[str, Any]:
        """
        執行 Leave-One-Group-Out (LOOCV) 交叉驗證。以受試者 (subject) 為 Group。
        """
        logo = LeaveOneGroupOut()
        subject_metrics = []
        
        # 收集所有折的預測結果以計算全域合併指標
        all_true = []
        all_pred = []
        all_prob = []
        
        for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=groups)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            
            # 訓練與預測
            model = model_cls(**model_kwargs)
            model.fit(X_train, y_train)
            
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)
            
            # 收集結果
            all_true.append(y_test)
            all_pred.append(preds)
            all_prob.append(probs)
            
            # 計算此受試者的獨立指標
            fold_metrics = SleepValidator.calculate_metrics(y_test, preds, probs, n_classes)
            subject_metrics.append(fold_metrics)

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
                 **model_kwargs) -> Dict[str, Any]:
        """
        執行以受試者（Subject）為分組單位的 Monte Carlo 交叉驗證。
        每次隨機選擇 70% 的受試者作訓練，30% 的受試者作測試。
        """
        unique_subjects = np.unique(groups)
        n_subjects = len(unique_subjects)
        n_train_subjects = int(np.round(train_size * n_subjects))
        
        trial_metrics = []
        
        for trial in range(n_splits):
            # 隨機打亂受試者 ID
            rng = np.random.default_rng(seed=42 + trial)
            shuffled_subjs = rng.permutation(unique_subjects)
            
            train_subjs = shuffled_subjs[:n_train_subjects]
            test_subjs = shuffled_subjs[n_train_subjects:]
            
            # 根據受試者 ID 過濾數據索引
            train_mask = np.isin(groups, train_subjs)
            test_mask = np.isin(groups, test_subjs)
            
            X_train, y_train = X[train_mask], y[train_mask]
            X_test, y_test = X[test_mask], y[test_mask]
            
            # 訓練與預測
            model = model_cls(**model_kwargs)
            model.fit(X_train, y_train)
            
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)
            
            # 計算該 trial 指標
            t_metrics = SleepValidator.calculate_metrics(y_test, preds, probs, n_classes)
            trial_metrics.append(t_metrics)
            
        # 彙總多個 trial 指標的 Mean ± Std
        mccv_summary = {}
        metric_keys = list(trial_metrics[0].keys())
        for key in metric_keys:
            vals = [m[key] for m in trial_metrics]
            mccv_summary[f"{key}_mean"] = float(np.mean(vals))
            mccv_summary[f"{key}_std"] = float(np.std(vals))
            
        return {
            "mccv_summary": mccv_summary,
            "raw_trial_metrics": trial_metrics
        }
