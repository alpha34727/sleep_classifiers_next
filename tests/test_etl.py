import unittest
import numpy as np
from pathlib import Path
from sleep_classifiers_next.config import Settings
from sleep_classifiers_next.etl.raw_processor import RawDataProcessor
from sleep_classifiers_next.etl.activity_count import ActivityCountProcessor
from sleep_classifiers_next.etl.circadian import CircadianProcessor

class TestETLComponents(unittest.TestCase):
    def setUp(self):
        self.settings = Settings()
        self.settings.paths.data_dir = Path("c:/Users/alpha/Desktop/sleep_classifiers_next/data")
        self.settings.paths.output_dir = Path("c:/Users/alpha/Desktop/sleep_classifiers_next/outputs_test")

    def test_activity_count_scaling(self):
        processor = ActivityCountProcessor(self.settings)
        # Create a mock motion data: 10 seconds at 50Hz (500 samples)
        # columns: timestamp, x, y, z
        timestamps = np.linspace(0, 10, 500)
        x = np.zeros(500)
        y = np.zeros(500)
        # z-axis containing some signals
        z = np.sin(2 * np.pi * 5 * timestamps)  # 5Hz signal
        mock_motion = np.column_stack((timestamps, x, y, z))

        counts_df = processor.compute_activity_counts(mock_motion)
        # Check shapes
        self.assertEqual(counts_df.ndim, 2)
        self.assertEqual(counts_df.shape[1], 2)
        # Values should be non-negative
        self.assertTrue(np.all(counts_df[:, 1] >= 0.0))

    def test_raw_processor_cropping(self):
        processor = RawDataProcessor(self.settings)
        # 100 samples from timestamp 0 to 99
        data = np.column_stack((np.arange(100), np.random.rand(100)))
        cropped = processor.crop_to_interval(data, 10.0, 90.0)
        self.assertEqual(np.min(cropped[:, 0]), 10.0)
        self.assertEqual(np.max(cropped[:, 0]), 89.0)

    def test_circadian_clock_ode(self):
        processor = CircadianProcessor(self.settings)
        # Test the simple ode calculation
        t = 12.0
        y = np.array([0.1, 0.2, 0.3])
        u_time = np.array([0.0, 24.0])
        u_light = np.array([0.01, 0.01])
        dydt = processor._simple_ode(t, y, u_time, u_light)
        
        self.assertEqual(len(dydt), 3)
        self.assertTrue(np.all(np.isfinite(dydt)))

    def test_circadian_model_run(self):
        processor = CircadianProcessor(self.settings)
        # 1 day of steps data at 1-minute steps (1440 points)
        steps_ts = np.arange(0, 86400, 60.0)
        steps_vals = np.random.randint(0, 50, len(steps_ts))
        steps_data = np.column_stack((steps_ts, steps_vals))

        # PSG epochs (30s) spanning 9 hours (1080 epochs)
        psg_ts = np.arange(0, 9 * 3600, 30.0)

        circadian_feature = processor.simulate_circadian_model(steps_data, psg_ts)
        self.assertEqual(circadian_feature.shape, (len(psg_ts), 1))
        # Value must be bounded above lower bound
        self.assertTrue(np.all(circadian_feature >= -0.2))

if __name__ == "__main__":
    unittest.main()
