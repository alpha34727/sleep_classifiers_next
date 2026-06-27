import gc
import numpy as np
import polars as pl
from pathlib import Path
from sleep_classifiers_next.config import Settings
from sleep_classifiers_next.etl.raw_processor import RawDataProcessor
from sleep_classifiers_next.etl.activity_count import ActivityCountProcessor
from sleep_classifiers_next.etl.circadian import CircadianProcessor

class FeaturePipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.cfg = settings.features
        self.paths = settings.paths
        self.raw_processor = RawDataProcessor(settings)
        self.count_processor = ActivityCountProcessor(settings)
        self.circadian_processor = CircadianProcessor(settings)
        self.paths.make_dirs()

    def process_subject(self, subject_id: str, is_mesa: bool = False) -> pl.DataFrame:
        """
        Runs the complete subject-by-subject ETL pipeline.
        Extracts features, aligns them with PSG labels, and returns a Polars DataFrame.
        """
        if self.settings.verbose:
            print(f"[{subject_id}] Loading and cropping raw datasets...")

        # 1. Load and crop raw data to intersecting interval
        cropped_psg, cropped_motion, cropped_hr, raw_steps, start_t, end_t = \
            self.raw_processor.process_subject_raw(subject_id)

        # Apply legacy compatibility drop if requested
        if self.cfg.legacy_compatibility:
            cropped_psg_valid = cropped_psg[1:]
            cropped_motion_valid = cropped_motion[1:]
            cropped_hr_valid = cropped_hr[1:]
        else:
            cropped_psg_valid = cropped_psg
            cropped_motion_valid = cropped_motion
            cropped_hr_valid = cropped_hr

        # 2. Get valid epoch mask
        valid_mask = self.raw_processor.get_valid_epochs(
            cropped_psg_valid, cropped_motion_valid, cropped_hr_valid
        )
        valid_psg = cropped_psg_valid[valid_mask]
        
        if len(valid_psg) == 0:
            raise ValueError(f"No valid epochs found for subject {subject_id}")

        valid_epochs_ts = valid_psg[:, 0]
        first_valid_ts = valid_epochs_ts[0]

        # 3. Compute activity counts
        if self.settings.verbose:
            print(f"[{subject_id}] Computing activity counts...")
        activity_counts = self.count_processor.compute_activity_counts(cropped_motion)
        
        if self.cfg.legacy_compatibility:
            activity_counts = activity_counts[1:]

        # 4. Interpolate activity counts on a 1-second grid
        count_ts = activity_counts[:, 0]
        count_vals = activity_counts[:, 1]
        t_grid_start = np.amin(count_ts)
        t_grid_end = np.amax(count_ts)
        interpolated_ts = np.arange(t_grid_start, t_grid_end, 1.0)
        interpolated_counts = np.interp(interpolated_ts, count_ts, count_vals)

        # 5. Interpolate and normalize heart rate
        if self.settings.verbose:
            print(f"[{subject_id}] Interpolating and filtering heart rate...")
        
        hr_ts = cropped_hr_valid[:, 0]
        hr_vals = cropped_hr_valid[:, 1]
        hr_grid_start = np.amin(hr_ts)
        hr_grid_end = np.amax(hr_ts)
        
        # Linearly interpolate HR to 1s grid
        hr_interpolated_ts = np.arange(hr_grid_start, hr_grid_end, 1.0)
        hr_interpolated_vals = np.interp(hr_interpolated_ts, hr_ts, hr_vals)

        # Convolve HR with Difference of Gaussians (DoG) filter
        hr_filtered = self._convolve_with_dog(hr_interpolated_vals, self.cfg.hr_window_size)
        
        # Normalize by the 90th percentile of absolute values
        hr_scale = np.percentile(np.abs(hr_filtered), 90.0)
        if hr_scale == 0:
            hr_scale = 1.0
        hr_normalized = hr_filtered / hr_scale

        # 6. Compute features for each valid epoch
        if self.settings.verbose:
            print(f"[{subject_id}] Building local window features...")

        motion_features = []
        hr_features = []
        cosine_features = []
        time_features = []

        for ts in valid_epochs_ts:
            # Motion Counts slice using strict math-matched indices
            m_start_idx, m_end_idx = self._get_window_slice(ts, t_grid_start, len(interpolated_counts))
            m_window = interpolated_counts[m_start_idx : m_end_idx]
            motion_features.append(self._smooth_gauss(m_window, self.cfg.motion_gaussian_sigma))

            # HR slice using strict math-matched indices
            h_start_idx, h_end_idx = self._get_window_slice(ts, hr_grid_start, len(hr_normalized))
            h_window = hr_normalized[h_start_idx : h_end_idx]
            hr_features.append(np.std(h_window))

            # Cosine proxy
            ts_elapsed = ts - first_valid_ts
            cosine_val = -1.0 * np.cos((ts_elapsed - 5.0 * self.cfg.seconds_per_hour) * 2.0 * np.pi / self.cfg.seconds_per_day)
            cosine_features.append(cosine_val)

            # Elapsed time feature (in hours)
            time_features.append(ts_elapsed / 3600.0)

        # 7. Compute circadian clock proxy feature
        if self.settings.verbose:
            print(f"[{subject_id}] Simulating circadian clock proxy...")
        circadian_features = self.circadian_processor.simulate_circadian_model(
            raw_steps, valid_epochs_ts, is_mesa=is_mesa
        ).flatten()

        # 8. Construct final Polars DataFrame
        labels = valid_psg[:, 1]
        df = pl.DataFrame({
            "subject_id": [subject_id] * len(valid_epochs_ts),
            "timestamp": valid_epochs_ts,
            "motion_count": motion_features,
            "heart_rate_std": hr_features,
            "cosine_proxy": cosine_features,
            "circadian_proxy": circadian_features,
            "time_proxy": time_features,
            "label": labels
        })

        # Save to disk as Parquet
        parquet_path = self.paths.features_dir / f"{subject_id}_features.parquet"
        df.write_parquet(parquet_path)

        if self.settings.verbose:
            print(f"[{subject_id}] Parquet saved successfully to {parquet_path}")

        # Clean memory and call Garbage Collection immediately
        del cropped_psg, cropped_motion, cropped_hr, raw_steps, activity_counts
        del interpolated_counts, hr_interpolated_vals, hr_filtered, hr_normalized
        gc.collect()

        return df

    def run_all_etl(self, subject_ids: list[str]) -> None:
        """
        Sequentially runs ETL for all subjects to guarantee minimal RAM usage.
        """
        for sid in subject_ids:
            try:
                self.process_subject(sid)
            except Exception as e:
                print(f"Error processing subject {sid}: {e}")

    def _get_window_slice(self, ts: float, t_grid_start: float, grid_length: int) -> tuple[int, int]:
        # Legacy window is strictly between ts - 285 and ts + 315
        start_time = ts - 285.0
        end_time = ts + 315.0
        
        first_idx = int(np.floor(start_time - t_grid_start)) + 1
        last_idx = int(np.ceil(end_time - t_grid_start)) - 1
        
        start_idx = max(0, first_idx)
        end_idx = min(grid_length, last_idx + 1)
        return start_idx, end_idx

    def _smooth_gauss(self, y: np.ndarray, sigma: float) -> float:
        box_pts = len(y)
        if box_pts == 0:
            return 0.0
        mu = box_pts // 2
        box = np.exp(-0.5 * (((np.arange(box_pts) - mu) / sigma) ** 2))
        box_sum = np.sum(box)
        if box_sum == 0:
            box_sum = 1.0
        box = box / box_sum
        return float(np.dot(box, y))

    def _convolve_with_dog(self, y: np.ndarray, box_pts: int) -> np.ndarray:
        y_centered = y - np.mean(y)
        mu = box_pts // 2
        idx = np.arange(box_pts)
        
        box = np.exp(-0.5 * (((idx - mu) / 120.0) ** 2)) - 0.75 * np.exp(-0.5 * (((idx - mu) / 600.0) ** 2))
        
        # Replicate legacy padding insertion
        pad_size = box_pts // 2
        y_padded = np.insert(y_centered, 0, np.flip(y_centered[0:pad_size]))
        y_padded = np.insert(y_padded, len(y_padded) - 1, np.flip(y_centered[-pad_size:]))
        
        # Convolve
        y_smooth = np.convolve(y_padded, box, mode='valid')
        return y_smooth
