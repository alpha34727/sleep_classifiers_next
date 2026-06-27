import numpy as np
import polars as pl
from pathlib import Path
from sleep_classifiers_next.config import Settings

class RawDataProcessor:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.cfg = settings.features
        self.paths = settings.paths

    def load_raw_motion(self, subject_id: str) -> np.ndarray:
        path = self.paths.data_dir / "motion" / f"{subject_id}_acceleration.txt"
        df = pl.read_csv(
            path, 
            has_header=False, 
            separator=" ", 
            new_columns=["timestamp", "x", "y", "z"],
            schema_overrides={"timestamp": pl.Float64, "x": pl.Float64, "y": pl.Float64, "z": pl.Float64}
        )
        return df.to_numpy()

    def load_raw_heart_rate(self, subject_id: str) -> np.ndarray:
        path = self.paths.data_dir / "heart_rate" / f"{subject_id}_heartrate.txt"
        df = pl.read_csv(
            path, 
            has_header=False, 
            separator=",", 
            new_columns=["timestamp", "heart_rate"],
            schema_overrides={"timestamp": pl.Float64, "heart_rate": pl.Float64}
        )
        return df.to_numpy()

    def load_raw_labels(self, subject_id: str) -> np.ndarray:
        path = self.paths.data_dir / "labels" / f"{subject_id}_labeled_sleep.txt"
        df = pl.read_csv(
            path, 
            has_header=False, 
            separator=" ", 
            new_columns=["timestamp", "stage"],
            schema_overrides={"timestamp": pl.Float64, "stage": pl.Int64}
        )
        return df.to_numpy()

    def load_raw_steps(self, subject_id: str) -> np.ndarray:
        path = self.paths.data_dir / "steps" / f"{subject_id}_steps.txt"
        df = pl.read_csv(
            path, 
            has_header=False, 
            separator=",", 
            new_columns=["timestamp", "steps"],
            schema_overrides={"timestamp": pl.Float64, "steps": pl.Float64}
        )
        return df.to_numpy()

    def crop_to_interval(self, data: np.ndarray, start_time: float, end_time: float) -> np.ndarray:
        valid_mask = (data[:, 0] >= start_time) & (data[:, 0] < end_time)
        return data[valid_mask]

    def process_subject_raw(self, subject_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
        """
        Loads all raw data for a subject and crops them to their overlapping intersecting interval.
        Returns:
            cropped_psg: np.ndarray of shape (N, 2)
            cropped_motion: np.ndarray of shape (M, 4)
            cropped_hr: np.ndarray of shape (K, 2)
            raw_steps: np.ndarray of shape (L, 2) (kept raw for circadian input)
            start_time: float
            end_time: float
        """
        psg = self.load_raw_labels(subject_id)
        motion = self.load_raw_motion(subject_id)
        hr = self.load_raw_heart_rate(subject_id)
        steps = self.load_raw_steps(subject_id)

        # Remove repeats (similar to utils.remove_repeats)
        motion = self._remove_repeats(motion)
        hr = self._remove_repeats(hr)

        # Get overlapping interval
        start_time = max(psg[0, 0], motion[0, 0], hr[0, 0])
        end_time = min(psg[-1, 0], motion[-1, 0], hr[-1, 0])

        if start_time >= end_time:
            raise ValueError(f"No overlapping time interval found for subject {subject_id} "
                             f"(start_time={start_time}, end_time={end_time})")

        cropped_psg = self.crop_to_interval(psg, start_time, end_time)
        cropped_motion = self.crop_to_interval(motion, start_time, end_time)
        cropped_hr = self.crop_to_interval(hr, start_time, end_time)

        return cropped_psg, cropped_motion, cropped_hr, steps, start_time, end_time

    def get_valid_epochs(self, cropped_psg: np.ndarray, cropped_motion: np.ndarray, cropped_hr: np.ndarray) -> np.ndarray:
        """
        Filters epochs that have at least one motion sample and one heart rate sample, and are not unscored.
        Returns a boolean mask of shape (num_psg_epochs,) indicating validity.
        """
        start_time = cropped_psg[0, 0]
        epoch_dur = self.cfg.epoch_duration

        # Compute valid epoch dictionary key sets
        motion_floored = cropped_motion[:, 0] - np.mod(cropped_motion[:, 0] - start_time, epoch_dur)
        hr_floored = cropped_hr[:, 0] - np.mod(cropped_hr[:, 0] - start_time, epoch_dur)

        motion_set = set(motion_floored)
        hr_set = set(hr_floored)

        valid_mask = []
        for row in cropped_psg:
            ts = row[0]
            stage = int(row[1])
            is_valid = (ts in motion_set) and (ts in hr_set) and (0 <= stage <= 5)
            valid_mask.append(is_valid)

        return np.array(valid_mask, dtype=bool)

    def _remove_repeats(self, array: np.ndarray) -> np.ndarray:
        # Replicates utils.remove_repeats logic
        _, unique_indices = np.unique(array, axis=0, return_index=True)
        unique_array = array[np.sort(unique_indices)]
        unique_array = unique_array[np.argsort(unique_array[:, 0])]
        return unique_array
