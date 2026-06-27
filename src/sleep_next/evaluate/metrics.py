import numpy as np
from sklearn.metrics import roc_curve, auc, cohen_kappa_score, accuracy_score, recall_score, precision_score
from sleep_next.config import settings

def get_tst(labels: np.ndarray) -> float:
    # labels > 0 represent sleep
    sleep_epochs = np.sum(labels > 0)
    return float(sleep_epochs * 30.0 / 60.0)

def get_wake_after_sleep_onset(labels: np.ndarray) -> float:
    sleep_indices = np.where(labels > 0)[0]
    if len(sleep_indices) > 0:
        sol_index = np.amin(sleep_indices)
        wake_after_onset = np.sum((labels == 0) & (np.arange(len(labels)) > sol_index))
        return float(wake_after_onset * 30.0 / 60.0)
    else:
        return float(len(labels) * 30.0 / 60.0)

def get_sleep_efficiency(labels: np.ndarray) -> float:
    if len(labels) == 0:
        return 0.0
    sleep_epochs = np.sum(labels > 0)
    return float(sleep_epochs / len(labels))

def get_sleep_onset_latency(labels: np.ndarray) -> float:
    sleep_indices = np.where(labels > 0)[0]
    if len(sleep_indices) > 0:
        return float(np.amin(sleep_indices) * 30.0 / 60.0)
    else:
        return float(len(labels) * 30.0 / 60.0)

def get_time_in_rem(labels: np.ndarray) -> float:
    # 2 represents REM in ThreeClassLabel
    rem_epochs = np.sum(labels == 2)
    return float(rem_epochs * 30.0 / 60.0)

def get_time_in_nrem(labels: np.ndarray) -> float:
    # 1 represents NREM in ThreeClassLabel
    nrem_epochs = np.sum(labels == 1)
    return float(nrem_epochs * 30.0 / 60.0)

class SleepWakePerformance:
    def __init__(self, accuracy, wake_correct, sleep_correct, kappa, auc_val, sleep_pv, wake_pv):
        self.accuracy = accuracy
        self.wake_correct = wake_correct  # Specificity
        self.sleep_correct = sleep_correct  # Sensitivity
        self.kappa = kappa
        self.auc = auc_val
        self.sleep_predictive_value = sleep_pv
        self.wake_predictive_value = wake_pv

class ThreeClassPerformance:
    def __init__(self, accuracy, wake_correct, rem_correct, nrem_correct, kappa):
        self.accuracy = accuracy
        self.wake_correct = wake_correct
        self.rem_correct = rem_correct
        self.nrem_correct = nrem_correct
        self.kappa = kappa

def compute_sleep_wake_performance(true_labels: np.ndarray, class_probs: np.ndarray, sleep_threshold: float) -> SleepWakePerformance:
    # Two-class evaluation. true_labels: 0=wake, 1=sleep
    fpr, tpr, thresholds = roc_curve(true_labels, class_probs[:, 1], pos_label=1, drop_intermediate=False)
    auc_val = auc(fpr, tpr)
    
    predicted = (class_probs[:, 1] >= sleep_threshold).astype(np.int32)
    
    accuracy = accuracy_score(true_labels, predicted)
    kappa = cohen_kappa_score(true_labels, predicted)
    
    wake_correct = recall_score(true_labels, predicted, pos_label=0, zero_division=0)
    sleep_correct = recall_score(true_labels, predicted, pos_label=1, zero_division=0)
    
    sleep_pv = precision_score(true_labels, predicted, pos_label=1, zero_division=0)
    wake_pv = precision_score(true_labels, predicted, pos_label=0, zero_division=0)
    
    return SleepWakePerformance(accuracy, wake_correct, sleep_correct, kappa, auc_val, sleep_pv, wake_pv)
