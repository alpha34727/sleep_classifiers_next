from lightgbm import LGBMClassifier
from sleep_classifiers_next.models.base import BaseSleepClassifier

class LightGBMSleepClassifier(BaseSleepClassifier):
    def __init__(self):
        clf = LGBMClassifier(
            class_weight="balanced",
            random_state=42,
            verbosity=-1,
            n_estimators=100
        )
        param_grid = {
            "learning_rate": [0.01, 0.05, 0.1],
            "num_leaves": [15, 31, 63]
        }
        super().__init__(name="LightGBM", classifier=clf, param_grid=param_grid)
