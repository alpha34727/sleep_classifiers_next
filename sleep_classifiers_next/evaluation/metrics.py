import numpy as np
from sklearn.metrics import roc_curve, auc, cohen_kappa_score, accuracy_score, recall_score, precision_score
from pydantic import BaseModel

class SleepWakePerformance(BaseModel):
    accuracy: float
    wake_correct: float
    sleep_correct: float
    kappa: float
    auc: float
    sleep_predictive_value: float
    wake_predictive_value: float

class ThreeClassPerformance(BaseModel):
    accuracy: float
    wake_correct: float
    rem_correct: float
    nrem_correct: float
    kappa: float

class MetricsCalculator:
    @staticmethod
    def calculate_sleep_wake(true_labels: np.ndarray, probabilities: np.ndarray, sleep_threshold: float = 0.5) -> SleepWakePerformance:
        # probabilities shape is (N, 2), where column 0 is Wake prob, column 1 is Sleep prob
        # If probabilities shape is (N, 3), convert it to 2-class probabilities
        if probabilities.shape[1] == 3:
            # Sleep probability = NREM + REM probability (columns 1 + 2)
            probs_2class = np.zeros((len(probabilities), 2))
            probs_2class[:, 0] = probabilities[:, 0]
            probs_2class[:, 1] = probabilities[:, 1] + probabilities[:, 2]
            probabilities = probs_2class

        # Calculate ROC and AUC
        fpr, tpr, thresholds = roc_curve(true_labels, probabilities[:, 1], pos_label=1, drop_intermediate=False)
        auc_val = float(auc(fpr, tpr))

        # Predict labels using threshold
        predicted_labels = np.where(probabilities[:, 1] >= sleep_threshold, 1, 0)

        # Compute metrics
        accuracy = float(accuracy_score(true_labels, predicted_labels))
        kappa = float(cohen_kappa_score(true_labels, predicted_labels))
        
        wake_correct = float(recall_score(true_labels, predicted_labels, pos_label=0))
        sleep_correct = float(recall_score(true_labels, predicted_labels, pos_label=1))
        
        sleep_predictive_value = float(precision_score(true_labels, predicted_labels, pos_label=1, zero_division=0.0))
        wake_predictive_value = float(precision_score(true_labels, predicted_labels, pos_label=0, zero_division=0.0))

        return SleepWakePerformance(
            accuracy=accuracy,
            wake_correct=wake_correct,
            sleep_correct=sleep_correct,
            kappa=kappa,
            auc=auc_val,
            sleep_predictive_value=sleep_predictive_value,
            wake_predictive_value=wake_predictive_value
        )

    @staticmethod
    def calculate_three_class(true_labels: np.ndarray, probabilities: np.ndarray, wake_threshold: float = 0.5, rem_threshold: float = 0.35) -> ThreeClassPerformance:
        # probabilities shape is (N, 3), where column 0 is Wake, column 1 is NREM, column 2 is REM
        predicted_labels = []
        for prob in probabilities:
            if prob[0] >= wake_threshold:
                predicted_labels.append(0)  # Wake
            elif prob[2] >= rem_threshold:
                predicted_labels.append(2)  # REM
            else:
                predicted_labels.append(1)  # NREM
        
        predicted_labels = np.array(predicted_labels)

        # Compute metrics
        accuracy = float(accuracy_score(true_labels, predicted_labels))
        kappa = float(cohen_kappa_score(true_labels, predicted_labels))
        
        recalls = recall_score(true_labels, predicted_labels, average=None, labels=[0, 1, 2], zero_division=0.0)
        wake_correct = float(recalls[0])
        nrem_correct = float(recalls[1])
        rem_correct = float(recalls[2])

        return ThreeClassPerformance(
            accuracy=accuracy,
            wake_correct=wake_correct,
            rem_correct=rem_correct,
            nrem_correct=nrem_correct,
            kappa=kappa
        )
