import sys
import unittest
from pathlib import Path
import numpy as np
import math

# Monkey-patch numpy.math for compatibility with legacy code on modern numpy 2.0+
np.math = math
import polars as pl

# Add legacy project to sys.path so we can import it for equivalence testing
legacy_path = Path("C:/Users/alpha/Desktop/sleep_classifiers")
if str(legacy_path) not in sys.path:
    sys.path.insert(0, str(legacy_path))

class TestEquivalence(unittest.TestCase):
    def setUp(self):
        # Initialize our new ETL settings
        from sleep_classifiers_next.config import Settings
        self.settings = Settings()
        self.settings.paths.data_dir = Path("c:/Users/alpha/Desktop/sleep_classifiers_next/data")
        # Direct new outputs to a test-specific directory to keep workspace clean
        self.settings.paths.output_dir = Path("c:/Users/alpha/Desktop/sleep_classifiers_next/outputs_test")
        self.settings.paths.make_dirs()
        self.settings.features.legacy_compatibility = True

        # Direct legacy outputs to the same test-specific directory
        from source.constants import Constants
        Constants.CROPPED_FILE_PATH = self.settings.paths.cropped_dir
        Constants.FEATURE_FILE_PATH = self.settings.paths.features_dir
        Constants.FIGURE_FILE_PATH = self.settings.paths.figures_dir
        Constants.INCLUDE_CIRCADIAN = False  # Keep circadian False as legacy MATLAB isn't available

    def test_subject_equivalence(self):
        subject_id = "5383425"

        # 1. Run legacy preprocessing and feature building
        from source.preprocessing.raw_data_processor import RawDataProcessor as LegacyRawDataProcessor
        from source.preprocessing.feature_builder import FeatureBuilder as LegacyFeatureBuilder

        print("\n[Test] Running legacy ETL pipeline...")
        LegacyRawDataProcessor.crop_all(subject_id)
        LegacyFeatureBuilder.build(subject_id)

        # Load legacy outputs
        from source.preprocessing.activity_count.activity_count_feature_service import ActivityCountFeatureService
        from source.preprocessing.heart_rate.heart_rate_feature_service import HeartRateFeatureService
        from source.preprocessing.time.time_based_feature_service import TimeBasedFeatureService
        from source.preprocessing.psg.psg_label_service import PSGLabelService

        legacy_counts = ActivityCountFeatureService.load(subject_id).flatten()
        legacy_hr = HeartRateFeatureService.load(subject_id).flatten()
        legacy_time = TimeBasedFeatureService.load_time(subject_id).flatten()
        legacy_cosine = TimeBasedFeatureService.load_cosine(subject_id).flatten()
        legacy_labels = PSGLabelService.load(subject_id).flatten()

        # 2. Run new ETL pipeline
        from sleep_classifiers_next.etl.feature_pipeline import FeaturePipeline
        print("[Test] Running new ETL pipeline...")
        pipeline = FeaturePipeline(self.settings)
        new_df = pipeline.process_subject(subject_id)

        # Extract features from new pipeline output
        new_counts = new_df["motion_count"].to_numpy()
        new_hr = new_df["heart_rate_std"].to_numpy()
        new_time = new_df["time_proxy"].to_numpy()
        new_cosine = new_df["cosine_proxy"].to_numpy()
        new_labels = new_df["label"].to_numpy()

        # 3. Assert shapes and decimal-level parity after accounting for legacy double header-drop bug
        # Legacy loaded features (length N-1 relative to legacy_valid_epochs) correspond to new features starting at index 1 (timestamp 60.0)
        print("[Test] Comparing lengths...")
        self.assertEqual(len(legacy_labels), len(new_labels) - 1)
        self.assertEqual(len(legacy_counts), len(new_counts) - 1)
        self.assertEqual(len(legacy_hr), len(new_hr) - 1)

        print("[Test] Asserting decimal-level parity...")
        # Assert within 1e-5 margin of error
        np.testing.assert_allclose(new_counts[1:], legacy_counts, rtol=1e-5, atol=1e-5, err_msg="Motion counts differ")
        np.testing.assert_allclose(new_hr[1:], legacy_hr, rtol=1e-5, atol=1e-5, err_msg="Heart rate standard deviations differ")
        np.testing.assert_allclose(new_time[1:], legacy_time, rtol=1e-5, atol=1e-5, err_msg="Time proxy values differ")
        np.testing.assert_allclose(new_cosine[1:], legacy_cosine, rtol=1e-5, atol=1e-5, err_msg="Cosine proxy values differ")
        
        # Assert labels match exactly
        np.testing.assert_array_equal(new_labels[1:], legacy_labels, err_msg="Labels do not match exactly")
        print("[Test] Equivalence test passed successfully!")

if __name__ == "__main__":
    unittest.main()
