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

def equalize_three_class_thresholds(
    true_labels: np.ndarray,
    class_probabilities: np.ndarray,
    wake_scored_as_sleep_interpolation_point: float = 0.4
) -> ThreeClassPerformance:
    # true_labels: 0=wake, 1=nrem, 2=rem
    # class_probabilities: shape (N, 3), cols: P_wake, P_nrem, P_rem
    
    number_of_wake_scored_as_sleep_bins = 20
    false_positive_buffer = 0.001
    max_attempts_binary_search_wake = 50
    rem_nrem_accuracy_tolerance = 1e-2
    max_attempts_binary_search_rem_nrem = 15
    
    goal_fraction_wake_scored_as_sleep_spread = np.linspace(0.0, 0.95, number_of_wake_scored_as_sleep_bins)
    
    wake_scored_as_sleep_spread = []
    sleep_accuracy_spread = []
    accuracies = []
    kappas = []
    nrem_class_accuracies = []
    rem_class_accuracies = []
    
    true_wake_indices = np.where(true_labels == 0)[0]
    true_nrem_indices = np.where(true_labels == 1)[0]
    true_rem_indices = np.where(true_labels == 2)[0]
    
    for goal_fraction_wake_scored_as_sleep in goal_fraction_wake_scored_as_sleep_spread:
        fraction_wake_scored_as_sleep = -1.0
        binary_search_counter = 0

        # --- FIX E1 ---
        # The legacy code (curve_performance_builder.py:124-127) resets these to
        # 0.5 / 0.25 at the *start of every goal-fraction's* binary search (guarded
        # by `if binary_search_counter == 0`).  Having them initialised only once
        # outside the loop caused each bin to warm-start from the previous bin's
        # converged threshold ("bleed"), exhausting the 50-iteration budget for
        # distant bins and silently skipping them, which corrupted the final
        # interp-derived REM/NREM accuracy values.
        threshold_for_sleep = 0.5
        threshold_delta = 0.25

        while (fraction_wake_scored_as_sleep < goal_fraction_wake_scored_as_sleep - false_positive_buffer
               or fraction_wake_scored_as_sleep >= goal_fraction_wake_scored_as_sleep + false_positive_buffer) \
               and binary_search_counter < max_attempts_binary_search_wake:

            
            if binary_search_counter > 0:
                if fraction_wake_scored_as_sleep < goal_fraction_wake_scored_as_sleep - false_positive_buffer:
                    threshold_for_sleep -= threshold_delta
                    threshold_delta /= 2.0
                else:
                    threshold_for_sleep += threshold_delta
                    threshold_delta /= 2.0
                    
            if goal_fraction_wake_scored_as_sleep == 1.0:
                threshold_for_sleep = 0.0
            elif goal_fraction_wake_scored_as_sleep == 0.0:
                threshold_for_sleep = 1.0
                
            predicted_sleep_indices = np.where(1.0 - class_probabilities[:, 0] >= threshold_for_sleep)[0]
            predicted_labels = np.zeros_like(true_labels)
            predicted_labels[predicted_sleep_indices] = 1
            predicted_labels_at_true_wake = predicted_labels[true_wake_indices]
            
            number_wake_correct = len(true_wake_indices) - np.count_nonzero(predicted_labels_at_true_wake)
            fraction_wake_correct = number_wake_correct / max(1.0, len(true_wake_indices))
            fraction_wake_scored_as_sleep = 1.0 - fraction_wake_correct
            
            binary_search_counter += 1
            
        if binary_search_counter < max_attempts_binary_search_wake:
            smallest_accuracy_difference = 2.0
            sleep_accuracy = 0.0
            rem_accuracy = 0.0
            nrem_accuracy = 0.0
            best_accuracy = -1.0
            kappa_at_best_accuracy = -1.0
            
            count_thresh = 0
            threshold_for_rem = 0.5
            threshold_delta_rem = 0.5
            
            temp_predicted = np.zeros_like(true_labels)
            
            while count_thresh < max_attempts_binary_search_rem_nrem and \
                    smallest_accuracy_difference > rem_nrem_accuracy_tolerance:
                count_thresh += 1
                
                temp_predicted[:] = 0
                rem_mask = class_probabilities[predicted_sleep_indices, 2] > threshold_for_rem
                temp_predicted[predicted_sleep_indices[rem_mask]] = 2
                temp_predicted[predicted_sleep_indices[~rem_mask]] = 1
                
                accuracy = accuracy_score(temp_predicted, true_labels)
                kappa = cohen_kappa_score(temp_predicted, true_labels)
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    kappa_at_best_accuracy = kappa
                    
                predicted_nrem_indices = np.where(temp_predicted == 1)[0]
                predicted_rem_indices = np.where(temp_predicted == 2)[0]
                
                correct_nrem_indices = np.intersect1d(predicted_nrem_indices, true_nrem_indices)
                correct_rem_indices = np.intersect1d(predicted_rem_indices, true_rem_indices)
                
                nrem_accuracy = len(correct_nrem_indices) / max(1.0, len(true_nrem_indices))
                rem_accuracy = len(correct_rem_indices) / max(1.0, len(true_rem_indices))
                
                sleep_accuracy = (len(correct_nrem_indices) + len(correct_rem_indices)) / max(
                    1.0, len(true_nrem_indices) + len(true_rem_indices)
                )
                
                smallest_accuracy_difference = np.abs(nrem_accuracy - rem_accuracy)
                
                if rem_accuracy < nrem_accuracy:
                    threshold_for_rem -= threshold_delta_rem / 2.0
                else:
                    threshold_for_rem += threshold_delta_rem / 2.0
                threshold_delta_rem /= 2.0
                
            wake_scored_as_sleep_spread.append(fraction_wake_scored_as_sleep)
            sleep_accuracy_spread.append(sleep_accuracy)
            nrem_class_accuracies.append(nrem_accuracy)
            rem_class_accuracies.append(rem_accuracy)
            accuracies.append(best_accuracy)
            kappas.append(kappa_at_best_accuracy)
            
    wake_scored_as_sleep_spread = np.array(wake_scored_as_sleep_spread)
    sleep_accuracy_spread = np.array(sleep_accuracy_spread)
    nrem_class_accuracies = np.array(nrem_class_accuracies)
    rem_class_accuracies = np.array(rem_class_accuracies)
    
    wake_scored_as_sleep_spread = np.insert(wake_scored_as_sleep_spread, 0, 0.0)
    sleep_accuracy_spread = np.insert(sleep_accuracy_spread, 0, 0.0)
    nrem_class_accuracies = np.insert(nrem_class_accuracies, 0, 0.0)
    rem_class_accuracies = np.insert(rem_class_accuracies, 0, 0.0)
    
    idx_best = np.argmax(accuracies) if len(accuracies) > 0 else 0
    accuracy = accuracies[idx_best] if len(accuracies) > 0 else 0.0
    kappa = kappas[idx_best] if len(kappas) > 0 else 0.0
    
    rem_correct = float(np.interp(wake_scored_as_sleep_interpolation_point, wake_scored_as_sleep_spread, rem_class_accuracies))
    nrem_correct = float(np.interp(wake_scored_as_sleep_interpolation_point, wake_scored_as_sleep_spread, nrem_class_accuracies))
    
    return ThreeClassPerformance(
        accuracy=accuracy,
        wake_correct=1.0 - wake_scored_as_sleep_interpolation_point,
        rem_correct=rem_correct,
        nrem_correct=nrem_correct,
        kappa=kappa
    )
