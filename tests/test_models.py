import unittest
import numpy as np
import polars as pl
import tempfile
import shutil
from pathlib import Path

from sleep_classifiers_next.config import Settings
from sleep_classifiers_next.models import (
    RandomForestSleepClassifier,
    LogisticRegressionSleepClassifier,
    KNNSleepClassifier,
    MLPSleepClassifier,
    LightGBMSleepClassifier,
)
from sleep_classifiers_next.evaluation import (
    CrossValidationService,
    MetricsCalculator,
)

class TestModelsAndEvaluation(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.settings = Settings()
        self.settings.paths.output_dir = self.tmp_dir
        self.settings.paths.make_dirs()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_model_strategies_dummy_fit(self):
        # Generate simple dummy training dataset: 50 samples, 4 features
        np.random.seed(42)
        X = np.random.randn(50, 4)
        y = np.random.randint(0, 2, size=50)

        classifiers = [
            RandomForestSleepClassifier(),
            LogisticRegressionSleepClassifier(),
            KNNSleepClassifier(),
            MLPSleepClassifier(max_iter=50),  # low max_iter for speed
            LightGBMSleepClassifier(),
        ]

        for clf in classifiers:
            # Low max_iter or simple settings to run quickly
            clf.train(X, y, scoring="roc_auc")
            
            # Predict
            preds = clf.predict(X)
            probs = clf.predict_proba(X)
            
            self.assertEqual(len(preds), 50)
            self.assertEqual(probs.shape, (50, 2))
            self.assertTrue(np.all((preds == 0) | (preds == 1)))

    def test_metrics_calculator_sleep_wake(self):
        true_labels = np.array([0, 1, 0, 1, 0, 1])
        probs = np.array([
            [0.9, 0.1],
            [0.2, 0.8],
            [0.4, 0.6],  # will predict 1 at threshold 0.5
            [0.1, 0.9],
            [0.8, 0.2],
            [0.3, 0.7]
        ])
        
        perf = MetricsCalculator.calculate_sleep_wake(true_labels, probs, sleep_threshold=0.5)
        self.assertGreaterEqual(perf.accuracy, 0.0)
        self.assertLessEqual(perf.accuracy, 1.0)
        self.assertTrue(0.0 <= perf.auc <= 1.0)
        self.assertTrue(-1.0 <= perf.kappa <= 1.0)

    def test_metrics_calculator_three_class(self):
        true_labels = np.array([0, 1, 2, 0, 1, 2])
        probs = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
            [0.2, 0.7, 0.1],
            [0.2, 0.1, 0.7]
        ])

        perf = MetricsCalculator.calculate_three_class(true_labels, probs, wake_threshold=0.5, rem_threshold=0.35)
        self.assertEqual(perf.accuracy, 1.0)
        self.assertEqual(perf.kappa, 1.0)
        self.assertEqual(perf.wake_correct, 1.0)
        self.assertEqual(perf.nrem_correct, 1.0)
        self.assertEqual(perf.rem_correct, 1.0)

    def test_cross_validation_service(self):
        # Create 3 dummy subject Parquet files in tmp_dir/features/
        # subject names: subj1, subj2, subj3
        subjects = ["subj1", "subj2", "subj3"]
        features_dir = self.settings.paths.features_dir
        
        np.random.seed(42)
        for sid in subjects:
            df = pl.DataFrame({
                "subject_id": [sid] * 20,
                "timestamp": np.arange(20) * 30.0,
                "motion_count": np.random.randn(20),
                "heart_rate_std": np.random.randn(20),
                "cosine_proxy": np.random.randn(20),
                "circadian_proxy": np.random.randn(20),
                "time_proxy": np.random.randn(20),
                "label": np.random.choice([0, 1, 2, 5], size=20)  # mix of Wake, N1/N2, REM
            })
            df.write_parquet(features_dir / f"{sid}_features.parquet")

        cv_service = CrossValidationService(self.settings)
        splits = cv_service.get_loso_splits(subjects)
        self.assertEqual(len(splits), 3)
        self.assertEqual(splits[0].testing_set, ["subj1"])
        self.assertEqual(splits[0].training_set, ["subj2", "subj3"])

        # Run LOSO cross-validation with LogisticRegression classifier
        clf = LogisticRegressionSleepClassifier()
        feature_cols = ["motion_count", "heart_rate_std", "cosine_proxy"]
        
        results = cv_service.run_cross_validation(
            clf, splits, feature_cols, is_three_class=False, scoring="roc_auc"
        )
        
        self.assertEqual(len(results), 3)
        for res in results:
            self.assertIn("true_labels", res)
            self.assertIn("probabilities", res)
            self.assertEqual(res["probabilities"].shape[1], 2)
